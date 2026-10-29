"""Authenticate completed reanchor generations and reuse the typed video ROI API."""
import copy,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from eval.action4_sam_full_review import ReceiptVerifier,read_json,checked_path,ROOT
from data.action4_video_roi_context import (
    VideoROIContext,VideoSAMSupportReceipt,VideoSAMSupportAdmission,rgb_digest)
from data.action4_sam_roi_inputs import array_digest

MAPPING={"track_00":0,"track_01":1}

def generation_owners(initial_session,events):
    if not isinstance(initial_session,str) or not initial_session:raise ValueError("Initial session required")
    owners=[{"generation":0,"start":4,"session_id":initial_session}]
    for event in events:
        prior=owners[-1];g=prior["generation"]+1;f=event["trigger_frame"]
        binding=event.get("binding",{})
        close=event.get("previous_close",{})
        if (event.get("outcome")!="native_bound" or type(f) is not int or f<=prior["start"]
                or type(event.get("seed_frame")) is not int or not 0<=f-event["seed_frame"]<=15
                or event.get("source_generation")!=prior["generation"] or event.get("proposed_generation")!=g
                or event.get("source_session_id")!=prior["session_id"]
                or event.get("native_obj_map")!=MAPPING
                or binding!={"session_id":event.get("new_session"),"generation":g,"frame_id":f,
                             "native_obj_map":MAPPING,"status":"native_binding_validated"}
                or not event.get("new_session") or any(o["session_id"]==event["new_session"] for o in owners)
                or not (close.get("is_success") is True or
                        g==1 and close.get("historical_closed_session")==initial_session)):
            raise ValueError("Unclosed, mixed or nonmonotonic generation transition")
        owners.append({"generation":g,"start":f,"session_id":event["new_session"]})
    return owners

def check_frame_owner(row,owners):
    frame=row["frame"]["global_frame_id"]
    selected=[o for o in owners if o["start"]<=frame]
    if not selected:raise ValueError("Preseed frame has no SAM owner")
    owner=selected[-1]
    if row.get("generation")!=owner["generation"] or row.get("session_id")!=owner["session_id"]:
        raise ValueError("Selected frame uses stale session/generation")
    return owner

def prediction_calibration_provenance(row,rows):
    current=row["frame"]["global_frame_id"]
    byframe=rows if isinstance(rows,dict) else {r["frame"]["global_frame_id"]:r["frame"] for r in rows}
    result={}
    for track in row["fused"]["tracks"]:
        history=track.get("calibration_frames",[])
        if history:
            last=history[-1]
            if type(last) is not int or last>current or last not in byframe:
                raise ValueError("Invalid prediction calibration frame")
            result[track["track_id"]]={"frame_id":last,"pts":byframe[last]["pts"],
                "evidence_kind":"B0_SAM_prediction_consistency_calibration",
                "verified_full_body_extent":False}
    return result


class ReanchorROIProvider:
    def __init__(self,*,run_root,completion_ref,predictions_ref,source_manifest_ref,b0_sha256,register_project_bodies=False):
        if type(register_project_bodies) is not bool:raise ValueError("Explicit project registration flag required")
        self.register_project_bodies=register_project_bodies
        self.root=checked_path(run_root);self.verifier=ReceiptVerifier();v=self.verifier
        if checked_path(completion_ref["path"])!=self.root/"run/COMPLETE.json" or checked_path(predictions_ref["path"])!=self.root/"run/FUSED_BOXES.jsonl" or checked_path(source_manifest_ref["path"])!=self.root/"source/SNAPSHOT_MANIFEST.json":
            raise ValueError("Explicit references belong to another run")
        done=v.bound_json(completion_ref);manifest=v.bound_json(source_manifest_ref)
        if (done.get("status")!="FULL2850_REANCHOR_CANDIDATE_COMPLETE_NOT_ACCURACY"
                or done.get("closed") is not True or done.get("frames")!=2850
                or done.get("source_identity")!=manifest["identity"]
                or read_json(self.root/"run/CLEANUP.json").get("failed") is not False
                or (self.root/"run/FAILED.json").exists()):
            raise ValueError("Only complete closed reanchor output may feed A0")
        cfg=read_json(v.verify(self.root/"source/config.json",manifest["files_sha256"]["config.json"]))
        if cfg["b0"]["sha256"]!=b0_sha256:raise ValueError("B0 counterpart mismatch")
        old=checked_path(cfg["reanchor"]["baseline_job"])
        oldmanifest=v.bound_json(cfg["reanchor"]["baseline_manifest"])
        oldcfg=read_json(v.verify(old/"source/config.json",oldmanifest["files_sha256"]["config.json"]))
        if oldcfg["b0"]!=cfg["b0"]:raise ValueError("Initial B0 identity changed")
        initial=read_json(old/"run/COMPLETE.json");ledger=read_json(old/"run/SESSION.json")
        if initial.get("closed") is not True or initial["single_session"]!=ledger["session_id"] or ledger["joint_obj_map"]!={"0":"track_00","1":"track_01"}:
            raise ValueError("Initial joint session not authenticated")
        self.owners=generation_owners(initial["single_session"],done["events"])
        def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
        self.owners[0].update(ledger_sha=digest(old/"run/SESSION.json"),
            source_sha=cfg["reanchor"]["baseline_manifest"]["sha256"],native_root=old)
        for owner,event in zip(self.owners[1:],done["events"]):
            eventpath=self.root/"run/events"/event["event_id"].replace(":","_")/"COMMIT.json"
            if read_json(eventpath)!=event:raise ValueError("Generation receipt differs from completion")
            owner.update(ledger_sha=digest(eventpath),source_sha=source_manifest_ref["sha256"],native_root=self.root)
        self.region=v.bound_json(cfg["region"]);self.b0_sha=b0_sha256;self.replay_sha=predictions_ref["sha256"]
        self.rows=[json.loads(line) for line in v.verify(predictions_ref["path"],predictions_ref["sha256"]).open()]
        if len(self.rows)!=2850 or [r["frame"]["global_frame_id"] for r in self.rows]!=list(range(2850)):
            raise ValueError("Missing/reordered video frames")
        for row in self.rows[4:]:check_frame_owner(row,self.owners)
        self.frame_by_id={r["frame"]["global_frame_id"]:r["frame"] for r in self.rows}
        if register_project_bodies:
            self._register_initial_bodies(old,oldcfg,oldmanifest,initial,ledger,manifest)


    def _register_initial_bodies(self,old,oldcfg,oldmanifest,initial,ledger,producer):
        from model.action4_sam_session_contracts import RegionSpec,SeedBinding
        from data.action4_video_roi_context import validate_project_body_scope
        def existing(p):
            p=checked_path(p);ref={"path":str(p.relative_to(ROOT)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
            return self.verifier.bound_json(ref),ref
        close,_=existing(old/"run/CLOSE.json")
        if (initial.get("status")!="FULL2850_COVERAGE_JOINT_NATIVE_BASELINE_NOT_ACCURACY"
            or initial.get("original_frames")!=2850 or initial.get("first_frame")!=4
            or initial.get("last_frame")!=2849 or initial.get("propagated_frames")!=2846
            or initial.get("preseed_unconfirmed")!=[0,1,2,3] or initial.get("causal_forward") is not True
            or initial.get("joint_obj_map")!={"0":"track_00","1":"track_01"}
            or ledger.get("start_frame")!=4 or ledger.get("generation")!=0 or close.get("is_success") is not True):
            raise ValueError("Initial full-video joint creation domain differs")
        for manifest in (oldmanifest,producer):
            if manifest["identity"]!=_canonical({k:v for k,v in manifest.items() if k!="identity"}):
                raise ValueError("Frozen producer manifest identity differs")
        region=self.region
        spec=RegionSpec(region["region_id"],tuple(x["global_frame_id"] for x in region["frames"]),
            tuple(x["pts"] for x in region["frames"]),tuple(region["time_base"]),
            oldcfg["region"]["sha256"],region["pixel_space_id"],region["width"],region["height"])
        prior_region=self.verifier.bound_json(oldcfg["region"])
        if prior_region!=region:raise ValueError("Baseline/reanchor source region differs")
        refs=oldcfg["instance_seed_sources"];seeds_ref=refs["track_00"]["seeds"]
        if refs["track_01"]["seeds"]!=seeds_ref:raise ValueError("Joint project seeds use different sources")
        selected=self.verifier.bound_json(seeds_ref)["seeds"]
        if [s["track_id"] for s in selected]!=list(MAPPING):raise ValueError("Initial project seed ordering differs")
        project={}
        for seed in selected:
            tid=seed["track_id"];oid=MAPPING[tid]
            if seed["local_index"]!=4 or seed["global_frame_id"]!=4:
                raise ValueError("Unexpected initial project creation frame")
            receipt,ref=existing(old/"run"/("seed_obj%d"%oid)/"RECEIPT.json")
            saved=receipt["seed"]
            if (receipt["session_id"]!=ledger["session_id"] or saved.get("obj_id")!=oid
                or saved.get("track_id")!=tid or saved.get("frame")!=4 or saved.get("GT_used") is not False
                or saved.get("protocol")!="action4.sam.pvs_box_corners.v1"
                or saved.get("candidate_id")!=seed["candidate_id"] or saved.get("bbox_native")!=seed["bbox_native"]):
                raise ValueError("Actual initial joint object/seed binding differs")
            raw=self.verifier.raw(receipt["raw"],old/"run"/("seed_obj%d"%oid))
            if native_presence(raw,4,(region["height"],region["width"]))[tid]!="target_present":
                raise ValueError("Initial actual native seed support absent")
            binding=SeedBinding(tid,spec.identity,seed["candidate_id"],4,4,tuple(seed["box_xywh01"]),
                tuple(seed["evidence_ids"]),"frozen_b0")
            project[tid]={"seed_fingerprint":binding.fingerprint,"creation_global_frame":4,
                "creation_pts":spec.frame(4)[1],"seed_binding_ref":ref,"seeds_ref":copy.deepcopy(seeds_ref)}
        scope={"clip_id":"full_original_video_development2850","region_identity":spec.identity,
               "producer_identity":producer["identity"]}
        self.body_scope=dict(scope,schema="action4.project_body_scope.v1",sequence_identity=_canonical(scope))
        validate_project_body_scope(self.body_scope)
        self.project_seeds=project
        self.body_events=project_body_events(self.body_scope,project,self.replay_sha,
            region["time_base"],region["pixel_space_id"])

    def current_native_masks(self,frame_id):
        """Read authenticated current native masks without decoding RGB again."""
        if type(frame_id) is not int or not 0<=frame_id<len(self.rows):raise ValueError("Frame outside fixed replay")
        row=self.rows[frame_id];f=row["frame"]
        base={"frame_id":frame_id,"pts":f["pts"],"RGB_sha256":f["sha256"],
              "pixel_space_id":self.region["pixel_space_id"],"time_base":self.region["time_base"],
              "replay_identity":self.replay_sha}
        if frame_id<4:return {},dict(base,status="initializing_no_native_session")
        owner=check_frame_owner(row,self.owners);status_path=checked_path(row["native_status"])
        expected=(owner["native_root"]/"run"/f"frame_{frame_id:06d}"/"STATUS.json" if owner["generation"]==0 else
                  owner["native_root"]/"run"/f"generation_{owner['generation']:03d}"/f"frame_{frame_id:06d}"/"STATUS.json")
        if status_path!=expected:raise ValueError("Current mask source escaped registered generation")
        status=read_json(status_path)
        for key,value in [("global_frame_id",frame_id),("pts",f["pts"]),("session_id",owner["session_id"]),
                          ("generation",owner["generation"]),("RGB_sha256",f["sha256"]),
                          ("time_base",self.region["time_base"]),("pixel_space_id",self.region["pixel_space_id"])]:
            if status.get(key)!=value:raise ValueError("Current mask identity differs: "+key)
        receipt=dict(base,native_status_ref=str(status_path.relative_to(ROOT)),session_id=owner["session_id"],
                     generation=owner["generation"],source_sha256=owner["source_sha"],raw=status.get("raw"))
        if status.get("raw") is None:return {},dict(receipt,status="missing_current_native_response")
        raw=self.verifier.raw(status["raw"],status_path.parent)
        ids=np.asarray(raw["outputs"]["out_obj_ids"]);masks=np.asarray(raw["outputs"]["out_binary_masks"])
        probs=np.asarray(raw["outputs"]["out_probs"])
        if (raw["frame_index"]!=frame_id or len(set(ids.tolist()))!=len(ids) or not set(ids.tolist())<=set(MAPPING.values())
            or masks.dtype!=np.bool_ or masks.shape!=(len(ids),self.region["height"],self.region["width"])
            or probs.shape!=(len(ids),) or not np.isfinite(probs).all()):
            raise ValueError("Current native masks/IDs/probabilities malformed")
        if owner["generation"] and status.get("native_obj_map")!=MAPPING:raise ValueError("Native obj mapping changed")
        present={};states={}
        for tid,oid in MAPPING.items():
            if oid not in ids:states[tid]="object_missing";continue
            i=ids.tolist().index(oid)
            if probs[i]<0 or not masks[i].any():states[tid]="removed_or_visibility_unknown";continue
            present[tid]=masks[i];states[tid]="current_mask_present"
        return present,dict(receipt,status="current_native_response",per_track_status=states,
                            mask_sha256={tid:array_digest(m) for tid,m in present.items()},
                            quality_is_not_probability=True)

    def load(self,frame_id,K):
        """Decode the bound original RGB once, then construct authenticated ROI context."""
        import torch
        if type(frame_id) is not int or not 0<=frame_id<len(self.rows):raise ValueError("Frame out of range")
        row=copy.deepcopy(self.rows[frame_id]);f=row["frame"]
        row["time_base"]=self.region["time_base"]
        # Existing tracker calibration is predictive evidence, not verified amodal extent.
        # Preserve it explicitly without promoting its timestamp to trusted full-body evidence.
        row["_roi_last_trusted_full"]={}
        row["_roi_last_prediction_calibration"]=prediction_calibration_provenance(row,self.frame_by_id)
        file=self.verifier.verify(f["image"],f["sha256"])
        with Image.open(file) as im:rgb=torch.from_numpy(np.array(im.convert("RGB"),copy=True)).permute(2,0,1)
        self.verifier.verify(f["image"],f["sha256"])  # stat-only after prior hash; detect decode-time changes
        if tuple(rgb.shape[1:])!=(self.region["height"],self.region["width"]):raise ValueError("Native RGB geometry differs")
        if frame_id<4:
            return rgb,VideoROIContext.from_replay(row,rgb,K,b0_sha256=self.b0_sha,replay_identity=self.replay_sha)
        owner=check_frame_owner(row,self.owners)
        status_path=checked_path(row["native_status"])
        expected=(owner["native_root"]/"run"/f"frame_{frame_id:06d}"/"STATUS.json"
                  if owner["generation"]==0 else
                  owner["native_root"]/"run"/f"generation_{owner['generation']:03d}"/f"frame_{frame_id:06d}"/"STATUS.json")
        if status_path!=expected:raise ValueError("Native evidence belongs to another generation/frame directory")
        st=read_json(status_path)
        for key,expected in [("global_frame_id",frame_id),("pts",f["pts"]),("session_id",owner["session_id"]),("generation",owner["generation"]),("RGB_sha256",f["sha256"]),("time_base",self.region["time_base"]),("pixel_space_id",self.region["pixel_space_id"])]:
            if st.get(key)!=expected:raise ValueError("Native evidence/frame binding mismatch: "+key)
        targets=[t for t in row["fused"]["tracks"] if t["status"]=="sam_supported"]
        support=None;current=None
        if st.get("raw") is not None:
            current=self.verifier.raw(st["raw"],status_path.parent)
            native_ids=np.asarray(current["outputs"]["out_obj_ids"]);native_masks=np.asarray(current["outputs"]["out_binary_masks"])
            if (current["frame_index"]!=frame_id or len(set(native_ids.tolist()))!=len(native_ids)
                    or not set(native_ids.tolist())<=set(MAPPING.values()) or native_masks.dtype!=np.bool_
                    or native_masks.shape!=(len(native_ids),*rgb.shape[1:])):
                raise ValueError("Current native visible support identity/shape differs")
            visible={}
            for tid,oid in MAPPING.items():
                if oid not in native_ids:continue
                mask=native_masks[native_ids.tolist().index(oid)];ys,xs=np.nonzero(mask)
                visible[tid]=None if not len(xs) else [float(xs.min()),float(ys.min()),float(xs.max()+1),float(ys.max()+1)]
            row["_roi_visible_boxes"]=visible
        support_ledger_sha=owner["ledger_sha"]
        if self.register_project_bodies:
            status=native_presence(current,frame_id,tuple(rgb.shape[1:]))
            if any(status[t["track_id"]]!="target_present" for t in targets):
                raise ValueError("Claimed SAM support not present in authenticated current raw")
            row["native_status"]=status
            row["pixel_space_id"]=self.region["pixel_space_id"]
            row["native_owners"]={tid:{"session_id":owner["session_id"],"native_obj_id":oid,
                "generation":owner["generation"],"ledger_sha":owner["ledger_sha"],
                "project_seed":copy.deepcopy(self.project_seeds[tid])} for tid,oid in MAPPING.items()}
            row["project_body_scope"]=copy.deepcopy(self.body_scope)
            row["project_body_registrations"]={tid:copy.deepcopy(event) for tid,event in self.body_events.items() if status[tid]=="target_present"}
            support_ledger_sha=_canonical(row["native_owners"])
        if targets:
            if current is None:raise ValueError("SAM-supported ROI lacks current native response")
            raw=current
            ids=np.asarray(raw["outputs"]["out_obj_ids"]);masks=np.asarray(raw["outputs"]["out_binary_masks"])
            if (raw["frame_index"]!=frame_id or len(set(ids.tolist()))!=len(ids)
                    or not set(ids.tolist())<=set(MAPPING.values())
                    or masks.dtype!=np.bool_ or masks.shape!=(len(ids),*rgb.shape[1:])):
                raise ValueError("Native mask identity/geometry differs")
            if owner["generation"] and st.get("native_obj_map")!=MAPPING:
                raise ValueError("Native generation instance mapping differs")
            expected=[];actual=[];rid=rgb_digest(rgb)
            for t in targets:
                oid=MAPPING[t["track_id"]]
                if oid not in ids:raise ValueError("Claimed SAM support missing in raw output")
                mask=masks[ids.tolist().index(oid)]
                if not mask.any():raise ValueError("Claimed SAM support is empty")
                receipt=VideoSAMSupportReceipt(frame_id,f["pts"],tuple(st["time_base"]),rid,
                    st["pixel_space_id"],t["track_id"],oid,owner["session_id"],owner["generation"],
                    owner["generation"],self.replay_sha,owner["source_sha"],support_ledger_sha,
                    array_digest(mask),tuple(t["bbox_for_roi"]))
                expected.append(receipt);actual.append((receipt,mask))
            support=VideoSAMSupportAdmission(st["pixel_space_id"],tuple(st["time_base"]),
                owner["source_sha"],support_ledger_sha,tuple(expected),tuple(actual))
        context=VideoROIContext.from_replay(row,rgb,K,b0_sha256=self.b0_sha,replay_identity=self.replay_sha,sam_support=support)
        return rgb,context

def _canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def native_presence(raw,frame_id,shape):
    """Authenticate actual current transport, never infer target presence from history."""
    if raw is None:return {tid:"target_visibility_unknown" for tid in MAPPING}
    outputs=raw["outputs"];ids=np.asarray(outputs["out_obj_ids"])
    masks=np.asarray(outputs["out_binary_masks"]);probs=np.asarray(outputs["out_probs"])
    if (type(raw.get("frame_index")) is not int or raw["frame_index"]!=frame_id
        or ids.ndim!=1 or ids.dtype.kind not in "iu" or len(set(ids.tolist()))!=len(ids)
        or not set(ids.tolist())<=set(MAPPING.values()) or probs.shape!=(len(ids),)
        or not np.isfinite(probs).all() or not np.all(((probs>=0)&(probs<=1))|(probs==-1e4))
        or masks.dtype!=np.bool_ or masks.shape!=(len(ids),*shape)):
        raise ValueError("Actual current joint native output identity/shape/probability differs")
    result={}
    for tid,oid in MAPPING.items():
        if oid not in ids:result[tid]="target_absent";continue
        i=ids.tolist().index(oid)
        result[tid]=("target_absent" if probs[i]==-1e4 else "target_present" if masks[i].any() else "target_visibility_unknown")
    return result

def project_body_events(scope,seeds,replay_sha,time_base,pixel_space):
    """Independent project generation0 from authenticated genesis, not native reanchors."""
    events={}
    for tid,seed in seeds.items():
        event={"schema":"action4.project_body_registration.v1","event_type":"project_body_created",
            "sequence_identity":scope["sequence_identity"],"producer_identity":scope["producer_identity"],
            "replay_identity":replay_sha,"track_id":tid,
            "scoped_body_id":_canonical({"sequence_identity":scope["sequence_identity"],"track_id":tid}),
            "project_generation":0,**copy.deepcopy(seed),"time_base":list(time_base),"pixel_space_id":pixel_space,
            "semantics":"provisional_seed_association_not_semantic_ground_truth",
            "semantic_gt":False,"physical_left_right_verified":False}
        event["event_identity"]=_canonical(event);events[tid]=event
    return events
