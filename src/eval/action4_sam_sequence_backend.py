"""Explicit continuous SAM3.1 stream adapter; reuse the verified native loader."""
from eval.action4_sam_image_eval import NativeBackend,validate_live_authority,validate_source_snapshot

SCHEMA="action4.sam.b0_video_sequence.v1"

class SequenceBackend(NativeBackend):
    kind="sam31_official_sequence"
    def __init__(self,config):
        if config.get("schema")!=SCHEMA:
            raise ValueError("Explicit video-sequence configuration required")
        sequence=config.get("sequence",{})
        if (type(sequence.get("frame_count")) is not int or sequence["frame_count"]<2
                or sequence.get("direction") not in ("both","forward") or sequence.get("output_prob_thresh")!=.5
                or sequence.get("evict_cached_frame_outputs") is not False):
            raise ValueError("Frozen whole-region sequence policy required")
        from eval.action4_sam_b0_sequence import sequence_direction
        sequence_direction(config,allow_face=config.get("sequence_consumer")=="A0_face_native_pilot")
        super().__init__(config)
        fixed=config.get("fixed_forward_instance_ids")
        if fixed is not None:
            if (config.get("causal_forward_corrections") is not True or not isinstance(fixed,list)
                or not 1<=len(fixed)<=2 or any(type(i) is not int or i<0 for i in fixed) or len(set(fixed))!=len(fixed)):
                raise ValueError("Explicit fixed forward object IDs required")
        self._active_stream=None
        self.history_observations=[]
        self.tracking_score_observations=[]

    def _load(self):
        super()._load()
        if self.config.get("input_tensor_cache") is not None:
            from model.action4_sam_cached_input import install_pinned_cached_input
            from eval.action4_sam_image_eval import atomic_json,contained
            receipt=install_pinned_cached_input(self.predictor,cache_spec=self.config["input_tensor_cache"],
                official_root=self.config["native"]["source_root"],
                expected_source_sha256=self.load_receipt["compat"]["official_method_source_sha256"],
                snapshot_manifest=contained(self.config["source_snapshot"])/"SNAPSHOT_MANIFEST.json")
            atomic_json(contained(self.config["output"])/"CACHED_INPUT_BRIDGE.json",receipt)
        if self.config.get("fixed_forward_instance_ids") is not None:
            from model.action4_sam_fixed_forward import install_fixed_forward
            from eval.action4_sam_image_eval import atomic_json,contained
            receipt=install_fixed_forward(self.predictor,self.config["native"]["source_root"])
            atomic_json(contained(self.config["output"])/"FIXED_FORWARD_CONTRACT.json",receipt)
        retention=self.config.get("history_window")
        if retention is not None:
            if retention!={"ordinary_frames":15,"max_conditioning_anchors":2} or self.config.get("seed_mode")!="instance_points":
                raise ValueError("Explicit reviewed 15-frame instance history policy required")
            from model.action4_sam_history_window import install_history_window
            from eval.action4_sam_image_eval import atomic_json,contained
            receipt=install_history_window(self.predictor,self.config["native"]["source_root"],window=15,max_anchors=2)
            atomic_json(contained(self.config["output"])/"HISTORY_WINDOW.json",receipt)
        offload=self.config.get("tracker_state_cpu_offload",False)
        if type(offload) is not bool:raise ValueError("Explicit CPU state-storage flag required")
        if offload:
            from model.action4_sam_state_offload import install_state_offload
            from eval.action4_sam_image_eval import atomic_json,contained
            receipt=install_state_offload(self.predictor,self.config["native"]["source_root"])
            atomic_json(contained(self.config["output"])/"STATE_STORAGE_BRIDGE.json",receipt)
        if self.config.get("instance_output_bridge", False):
            if self.config.get("seed_mode") != "instance_points":
                raise ValueError("Instance output bridge requires explicit point-instance mode")
            from model.action4_sam_seed_output import install_seed_output_bridge
            from eval.action4_sam_image_eval import atomic_json, contained
            receipt = install_seed_output_bridge(self.predictor, self.config["native"]["source_root"])
            atomic_json(contained(self.config["output"]) / "INSTANCE_OUTPUT_BRIDGE.json", receipt)

    def call(self,request):
        if self._active_stream is not None:
            raise RuntimeError("Close stream before another native operation")
        # The original image/job API remains restricted; streaming is separate.
        return super().call(request)

    def retire_previous_anchor(self,session_id,*,obj_id,initial_frame,old_frame,new_frame,binding):
        if self._active_stream is not None:
            raise RuntimeError("Close native stream before rotating conditioning anchors")
        if self.config.get("causal_forward_corrections") is not True or self.config.get("history_window") is None:
            raise ValueError("Explicit bounded correction mode required")
        if self.predictor is None or session_id not in self.predictor._all_inference_states:
            raise ValueError("Unknown native session")
        from model.action4_sam_anchor_rotation import retire_previous_anchor
        state=self.predictor._all_inference_states[session_id]["state"]
        with self.torch.inference_mode():
            return retire_previous_anchor(state,self.predictor.model.tracker,obj_id=obj_id,
                initial_frame=initial_frame,old_frame=old_frame,new_frame=new_frame,binding=binding)

    def propagate(self,request):
        if self.config["sequence"]["direction"] == "forward":
            raise ValueError("Causal sequence cannot enter bidirectional propagation")
        return self._propagate(request,direction="both")

    def propagate_forward(self,request):
        if (self.config.get("causal_forward_corrections") is not True
                and not (self.config["sequence"]["direction"] == "forward"
                         and (self.config.get("temporal_mode") == "causal_forward_B0_assist"
                              or (self.config.get("temporal_mode") == "causal_forward_A0_face_hypothesis"
                                  and self.config.get("sequence_consumer")=="A0_face_native_pilot")))):
            raise ValueError("Forward correction stream requires explicit candidate configuration")
        return self._propagate(request,direction="forward")

    def _propagate(self,request,*,direction):
        allowed={"type","session_id","propagation_direction","start_frame_index"}
        if not isinstance(request,dict) or set(request)-allowed or request.get("type")!="propagate_in_video":
            raise ValueError("Explicit full-region propagation request required; no chunking or eviction override")
        if request.get("propagation_direction",direction)!=direction:
            raise ValueError("Propagation direction differs from explicit entry point")
        index=request.get("start_frame_index")
        if index is not None and (type(index) is not int or not 0<=index<self.config["sequence"]["frame_count"]):
            raise ValueError("Propagation start must be a local frame index")
        if self.predictor is None or self.torch is None:raise RuntimeError("Native model not loaded")
        sid=request.get("session_id")
        if not isinstance(sid,str) or not sid or sid not in self.predictor._all_inference_states:
            raise ValueError("Unknown native session")
        if self._active_stream is not None:raise RuntimeError("One serial native operation per worker")
        validate_live_authority(self.config)
        validate_source_snapshot(self.config,require_execution=True)
        effective=dict(request,propagation_direction=direction,output_prob_thresh=.5,evict_cached_frame_outputs=False)
        self._active_stream=sid
        stream=None;forward_state=None;marker_owned=False
        try:
            fixed=self.config.get("fixed_forward_instance_ids")
            if direction=="forward" and fixed is not None:
                forward_state=self.predictor._all_inference_states[sid]["state"]
                if "_action4_fixed_forward_ids" in forward_state:raise RuntimeError("Stale forward-operation marker")
                forward_state["_action4_fixed_forward_ids"]=tuple(fixed);marker_owned=True
            with self.torch.inference_mode(),self.torch.autocast(device_type="cuda",dtype=self.torch.bfloat16):
                stream=self.predictor.handle_stream_request(effective)
                for raw in stream:
                    self.torch.cuda.synchronize()
                    if self.config.get("history_window") is not None:
                        from model.action4_sam_history_window import prune_history
                        state=self.predictor._all_inference_states[sid]["state"]
                        frame=raw["frame_index"]
                        if state.get("_action4_history_frame")!=frame:
                            raise RuntimeError("Native frame did not pass bounded memory-read hook")
                        proof=prune_history(state,frame,reverse=state["_action4_history_reverse"],
                                            window=15,after_frame=True,max_anchors=2)
                        counts=[len(c["output_dict"]["non_cond_frame_outputs"]) for c in state["sam2_inference_states"]]
                        if any(n>15 for n in counts):raise RuntimeError("Native ordinary history exceeded15")
                        self.history_observations.append({"frame":frame,"counts":counts,
                            "public_cache_count":len(state["cached_frame_outputs"]),"anchors":proof["anchors"],
                            "reverse":proof["reverse"],"allocated_mib":self.torch.cuda.memory_allocated()/2**20})
                    if self.config.get("record_tracking_scores",False):
                        from model.action4_sam_tracking_scores import tracking_scores
                        state=self.predictor._all_inference_states[sid]["state"]
                        evidence=tracking_scores(state,raw["frame_index"])
                        evidence["session_id"]=sid
                        self.tracking_score_observations.append(evidence)
                    # Response/obj/frame identity is validated by the existing
                    # ResponseEnvelope/RoundCollector bridge, not silently repaired here.
                    yield raw
        finally:
            try:
                if stream is not None and hasattr(stream,"close"):stream.close()
            finally:
                if marker_owned:forward_state.pop("_action4_fixed_forward_ids",None)
                self._active_stream=None
