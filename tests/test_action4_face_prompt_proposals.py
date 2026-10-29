"""Synthetic CPU face-prompt hypotheses; no actual SAM face inference or GT."""
import copy,json
from dataclasses import replace
from pathlib import Path
import unittest
import numpy as np

from tests import test_action4_clip_replay as clip_fixtures
from eval.action4_clip_replay import ClipReplay
from eval.action4_face_inputs import prepare_face_inputs
from eval.action4_face_prompt_proposals import FacePromptThresholds,FacePatchCandidates,propose_face_prompts
from data.action4_sam_roi_inputs import ImageFrame,array_digest
from data.action4_video_sam_snapshot import VideoMaskBinding
from data.action4_surface_snapshot import make_snapshot
from data.action4_roi_contract import RoiSpec
from model.action4_face_geometry import FaceGeometryContract

ROOT=clip_fixtures.ROOT


class FacePromptTests(unittest.TestCase):
    def setUp(self):
        owner=clip_fixtures.ClipReplayTests(methodName="runTest");self.addCleanup(owner.doCleanups);self.owner=owner
        self.folder,_,args,_=owner.fixture(forward=True)
        self.replay=ClipReplay(**args);self.K=np.array([[80.,0,20.],[0,80.,20.],[0,0,1.]])
        self.rgb,self.context=self.replay.load(783,self.K)
        pixels=self.rgb.numpy().transpose(1,2,0).copy()
        self.frame=ImageFrame("clip:783",pixels,self.replay.region.pixel_identity,
            self.replay.doc["frames"][3]["sha256"],"native_original",array_digest(pixels))
        self.contract=FaceGeometryContract.from_bytes((ROOT/"configs/model/action4_six_face_anchor_contract_v001.json").read_bytes(),
            (ROOT/"data/DJI_Action4_construct/r3_v001/asset_manifest.json").read_bytes())
        self.models={"A0":{"path":"outputs/action4_clip_replay_preparation_v001/CPU_NOT_MODEL_A0","sha256":"a"*64},
                     "B0":{"path":"outputs/action4_clip_replay_preparation_v001/CPU_NOT_MODEL_B0","sha256":"b"*64}}
        pred=self.context.select_for_a0(self.context.consume(self.rgb,self.K,b0_sha256="b"*64))[0]
        self.roi=RoiSpec.from_full_box(pred["bbox_for_a0"],1.5)
        xyz=np.full((1,3,112,112),.5,np.float32)
        xyz[0,2,:,:56]=0.  # z_min candidate
        xyz[0,0,:,56:]=0.  # x_min candidate
        self.output={"surface":{"xyz":xyz,"masks":np.zeros((1,2,112,112),np.float32),
                                "regions":np.zeros((1,65,112,112),np.float32)},
                     "rotation":np.eye(3,dtype=np.float32)[None],"translation":np.array([[.2,.2,1.]],np.float32)}
        self.pred=pred
        self.face=self.make_face(self.output)
        self.mask=self.replay._mask("track_00",3)
        native=self.replay.owners["track_00"]
        self.binding=VideoMaskBinding(783,313200,(1,11988),self.frame.pixel_space_id,"track_00",0,
            native["native_obj_id"],native["generation"],native["session_id"],self.replay.replay_identity,
            array_digest(self.mask),self.models["A0"]["sha256"])
        self.thresholds=FacePromptThresholds(.05,.05,8,2.,2,1,1)
        original=json.loads((ROOT/"outputs/action4_six_face_anchor_preparation_v001/patch_candidates_v002/PATCH_CANDIDATES.json").read_text())
        bounds=np.array(self.contract.canonical8_m)*1000.
        lo,hi=bounds.min(0),bounds.max(0)
        self.patchdoc={k:copy.deepcopy(v) for k,v in original.items() if k not in ("patches","triangle_mapping","summary")}
        self.patchdoc["fixture_only_not_actual_patch_extraction"]=True
        self.patchdoc["patches"]=[]
        for axis,face in ((2,4),(0,0)):
            b=np.array([lo,hi]);b[:,axis]=lo[axis]
            normal=np.eye(3)[axis]
            self.patchdoc["patches"].append({"physical_patch_id":f"synthetic_{face}",
                "logical_face_candidates":[face],"geometric_plane_candidate":True,
                "semantic_gt":False,"exterior_verified":False,"bounds_mm":b.tolist(),
                "plane_equation_n_dot_x_plus_d_equals_0":{"normal_object":normal.tolist(),"d_mm":float(-lo[axis])},
                "unknown_or_exclusion_reasons":["synthetic_fixture","hole_visible_internal_surface_not_ruled_out"]})
        self.patches=self.patch_collection(self.patchdoc)

    def make_face(self,output):
        snapshot=make_snapshot(frame=self.frame,prediction=self.pred,output=output,index=0,
            A112=self.roi.affine(112),A448=self.roi.affine(),K_native=self.K,
            K112=self.roi.camera(self.K,112).astype(np.float32),canonical=np.asarray(self.contract.canonical8_m,np.float32),
            models=self.models,video_context=self.context)
        return prepare_face_inputs(snapshot,frame=self.frame,expected_models=self.models,
            expected_prediction_id=self.pred["prediction_id"],roi_spec=self.roi,contract=self.contract,
            min_depth_m=1e-6,grazing_cos_threshold=.01)

    def patch_collection(self,doc):
        path=self.folder/f"synthetic_patch_{len(list(self.folder.glob('synthetic_patch_*')))}.json"
        ref=self.owner.write(path,doc)
        return FacePatchCandidates.from_ref(ref,self.contract)

    def run_proposals(self,**changes):
        args=dict(frame=self.frame,video_context=self.context,body_binding=self.binding,body_mask=self.mask,
                  patch_candidates=self.patches,contract=self.contract,expected_models=self.models,thresholds=self.thresholds)
        face=changes.pop("face_inputs",self.face);args.update(changes)
        return propose_face_prompts(face,**args)

    def test_actual_XYZ_points_bind_body_face_roi_and_are_not_GT(self):
        result=self.run_proposals()
        proposals=[x for x in result["hypotheses"] if x["status"]=="prompt_hypothesis"]
        self.assertEqual({x["face_id"] for x in proposals},{0,4})
        positive_sets=[]
        for p in proposals:
            self.assertEqual(p["labels"],[1,1,0,0])
            positive=set()
            for point,label,source in zip(p["points_native"],p["labels"],p["point_sources"]):
                x,y=point
                if label or source=="adjacent_face_candidate_inside_current_body":self.assertTrue(self.mask[y,x])
                else:self.assertFalse(self.mask[y,x])
                if label:positive.add(tuple(point))
            positive_sets.append(positive)
            self.assertFalse(p["native_face_mask_produced"])
            points=np.asarray(p["points_native"],float)
            distances=np.linalg.norm(points[:,None]-points[None,:],axis=-1)
            self.assertGreaterEqual(distances[np.triu_indices(len(points),1)].min(),2.)
        self.assertFalse(positive_sets[0]&positive_sets[1])
        self.assertFalse(result["semantic_gt"]);self.assertFalse(result["GT_used"])
        self.assertFalse(result["corner_probabilities_available"])
        self.assertEqual([f["face_id"] for f in result["face_diagnostics"]],list(range(6)))
        self.assertTrue(all(f["observed_face_certified"] is False for f in result["face_diagnostics"]))
        self.assertEqual(result["dependencies"]["body"]["scoped_body_id"],self.face.metadata["body_identity"]["scoped_body_id"])

    def test_mixed_depth_and_multicandidate_pixels_are_withheld_not_nearest_assigned(self):
        for mode in ("mixed_depth","multiple_faces"):
            doc=copy.deepcopy(self.patchdoc)
            if mode=="mixed_depth":
                extra=copy.deepcopy(doc["patches"][0]);extra["physical_patch_id"]="nearby_internal_layer"
                extra["plane_equation_n_dot_x_plus_d_equals_0"]["d_mm"]-=.02
                doc["patches"].append(extra)
            else:doc["patches"][0]["logical_face_candidates"]=[0,4]
            result=self.run_proposals(patch_candidates=self.patch_collection(doc))
            self.assertGreater(result["support_summary"]["ambiguous_patch_or_face_cells"],0)
            self.assertFalse(any(p["face_id"]==4 and p["status"]=="prompt_hypothesis" for p in result["hypotheses"]))

    def test_unknown_and_insufficient_points_no_fabricated_prompts(self):
        output=copy.deepcopy(self.output);output["surface"]["xyz"][:]=9.
        result=self.run_proposals(face_inputs=self.make_face(output))
        self.assertEqual(result["support_summary"]["body_supported_cells"],result["support_summary"]["unknown_cells"])
        self.assertTrue(all(p["status"]=="no_proposal" for p in result["hypotheses"]))
        result=self.run_proposals(thresholds=FacePromptThresholds(.05,.05,8,100.,2,1,1))
        self.assertTrue(all(p["status"]=="no_proposal" for p in result["hypotheses"]))
        self.assertTrue(all("points_native" not in p for p in result["hypotheses"]))

    def test_wrong_native_frame_instance_mask_models_and_ROI_rejected(self):
        for binding in (replace(self.binding,frame_id=784),replace(self.binding,track_id="track_01"),
                        replace(self.binding,native_obj_id=9),replace(self.binding,session_id="other")):
            with self.assertRaises(ValueError):self.run_proposals(body_binding=binding)
        wrong=np.roll(self.mask,4,axis=0)
        with self.assertRaises(ValueError):self.run_proposals(body_mask=wrong)
        with self.assertRaises(ValueError):
            self.run_proposals(body_mask=wrong,body_binding=replace(self.binding,mask_sha256=array_digest(wrong)))
        models=copy.deepcopy(self.models);models["A0"]["sha256"]="c"*64
        with self.assertRaises(ValueError):self.run_proposals(expected_models=models)
        metadata=copy.deepcopy(self.face.metadata);metadata["roi_spec"]["expansion"]=2.25
        with self.assertRaises(ValueError):self.run_proposals(face_inputs=replace(self.face,metadata=metadata))
        geometry=copy.deepcopy(self.face.geometry_prior);geometry["K_native"][0][0]+=1
        with self.assertRaises(ValueError):self.run_proposals(face_inputs=replace(self.face,geometry_prior=geometry))

    def test_unverified_body_or_patch_promotion_rejected(self):
        metadata=copy.deepcopy(self.face.metadata);metadata["body_identity"]["persistent_identity_ready"]=False
        with self.assertRaises(ValueError):self.run_proposals(face_inputs=replace(self.face,metadata=metadata))
        doc=copy.deepcopy(self.patchdoc);doc["patches"][0]["semantic_gt"]=True
        with self.assertRaises(ValueError):self.patch_collection(doc)
        with self.assertRaises(ValueError):
            self.run_proposals(patch_candidates=replace(self.patches,patches_json=self.patches.patches_json))
        with self.assertRaises(ValueError):FacePromptThresholds(-1.,.05,8,2.,2,1,1)



    def test_logical_consensus_resolves_same_face_depth_for_prompt_only(self):
        doc=copy.deepcopy(self.patchdoc)
        extra=copy.deepcopy(doc["patches"][0]);extra["physical_patch_id"]="distinct_depth_same_logical_face"
        extra["plane_equation_n_dot_x_plus_d_equals_0"]["d_mm"]-=.02
        doc["patches"].append(extra)
        patches=self.patch_collection(doc)
        old=self.run_proposals(patch_candidates=patches)
        policy={"schema":"action4.face_prompt_support_policy.v1","mode":"logical_face_consensus"}
        result=self.run_proposals(patch_candidates=patches,support_policy=policy)
        self.assertFalse(any(p["face_id"]==4 and p["status"]=="prompt_hypothesis" for p in old["hypotheses"]))
        face=next(p for p in result["hypotheses"] if p["face_id"]==4)
        self.assertEqual(face["status"],"prompt_hypothesis")
        self.assertEqual(set(face["physical_patch_ids"]),{"synthetic_4","distinct_depth_same_logical_face"})
        self.assertTrue(face["depth_ambiguous"]);self.assertTrue(face["depth_unresolved"])
        self.assertTrue(face["logical_face_only"])
        for key in ("geometry_fit_eligible","XYZ_binding_available","homography_eligible","PnP_eligible","physical_plane_registered"):
            self.assertFalse(face[key])
        self.assertGreater(result["support_summary"]["same_face_multi_patch_consensus_cells"],0)
        self.assertEqual(old["support_summary"]["unknown_cells"],result["support_summary"]["unknown_cells"])
        self.assertEqual(json.loads(patches.patches_json),doc["patches"])
        self.assertEqual(result["dependencies"]["support_policy"],policy)

    def test_logical_consensus_still_refuses_distinct_face_identity(self):
        doc=copy.deepcopy(self.patchdoc)
        extra=copy.deepcopy(doc["patches"][0]);extra["physical_patch_id"]="different_logical_identity"
        extra["logical_face_candidates"]=[0]
        doc["patches"].append(extra)
        policy={"schema":"action4.face_prompt_support_policy.v1","mode":"logical_face_consensus"}
        result=self.run_proposals(patch_candidates=self.patch_collection(doc),support_policy=policy)
        self.assertGreater(result["support_summary"]["distinct_face_ambiguous_cells"],0)
        self.assertFalse(any(p["face_id"]==4 and p["status"]=="prompt_hypothesis" for p in result["hypotheses"]))
        for p in result["hypotheses"]:self.assertFalse(p["geometry_fit_eligible"])

    def test_default_policy_stays_exclusive_and_optin_is_explicit(self):
        default=self.run_proposals()
        explicit=self.run_proposals(support_policy={
            "schema":"action4.face_prompt_support_policy.v1","mode":"exclusive_physical_patch"})
        summary=lambda result:[{k:v for k,v in p.items() if k not in ("parent_dependency_identity","hypothesis_identity")}
                               for p in result["hypotheses"]]
        self.assertEqual(summary(default),summary(explicit))
        self.assertNotIn("logical_face_only",default)
        with self.assertRaises(ValueError):
            self.run_proposals(support_policy={"schema":"other","mode":"logical_face_consensus"})


    def test_pose_visibility_is_optional_prior_not_self_confirming_gate(self):
        output=copy.deepcopy(self.output)
        output["rotation"][0]=np.diag([-1.,1.,-1.])
        face=self.make_face(output)
        old=self.run_proposals(face_inputs=face,support_policy={"schema":"action4.face_prompt_support_policy.v1","mode":"logical_face_consensus"})
        self.assertFalse(any(x["status"]=="prompt_hypothesis" for x in old["hypotheses"]))
        policy={"schema":"action4.face_prompt_support_policy.v2","mode":"logical_face_consensus","allow_pose_prior_disagreement":True}
        result=self.run_proposals(face_inputs=face,support_policy=policy)
        proposals=[x for x in result["hypotheses"] if x["status"]=="prompt_hypothesis"]
        self.assertEqual({x["face_id"] for x in proposals},{0,4})
        self.assertTrue(all(x["pose_prior_disagrees"] and x["independent_image_face_confirmation_required"] for x in proposals))
        self.assertTrue(all(not x["geometry_fit_eligible"] for x in proposals))
        output["surface"]["xyz"][:]=9.
        missing=self.run_proposals(face_inputs=self.make_face(output),support_policy=policy)
        self.assertFalse(any(x["status"]=="prompt_hypothesis" for x in missing["hypotheses"]))
        bad=dict(policy,allow_pose_prior_disagreement=1)
        with self.assertRaises(ValueError):self.run_proposals(support_policy=bad)


    def projected_proposals(self,**changes):
        from scipy.spatial.transform import Rotation
        from eval.action4_face_prompt_proposals import propose_projected_face_region_prompts,ProjectedFacePromptThresholds
        output=copy.deepcopy(self.output)
        output["rotation"][0]=Rotation.from_euler("xy",[-25,35],degrees=True).as_matrix()
        output["translation"][0]=[-.12,-.12,.85]
        output["surface"]["xyz"][:]=9.  # deliberately no physical-plane XYZ support
        self.projected_output=output
        face=changes.pop("face_inputs",self.make_face(output))
        args=dict(frame=self.frame,video_context=self.context,body_binding=self.binding,body_mask=self.mask,
                  patch_candidates=self.patches,contract=self.contract,expected_models=self.models,
                  thresholds=ProjectedFacePromptThresholds(.1,4,1.,1,1))
        args.update(changes)
        return propose_projected_face_region_prompts(face,**args)

    def test_projected_region_four_spread_points_and_true_negative_sources(self):
        from eval.action4_face_prompt_proposals import _polygon_interior_distance
        r=self.projected_proposals()
        proposals=[h for h in r["hypotheses"] if h["status"]=="prompt_hypothesis"]
        self.assertTrue(proposals)
        for h in proposals:
            self.assertEqual(h["labels"],[1,1,1,1,0,0])
            pts=np.asarray(h["points_native"])
            self.assertEqual(len(np.unique(pts,axis=0)),6)
            self.assertTrue(all(self.mask[y,x] for x,y in pts[:5]))
            self.assertFalse(self.mask[pts[-1,1],pts[-1,0]])
            inside,distance=_polygon_interior_distance(pts[:4],h["projected_quad_native_uv"])
            self.assertTrue(inside.all());self.assertTrue((distance>=.1).all())
            pair=np.linalg.norm(pts[:,None]-pts[None],axis=2)
            self.assertGreaterEqual(pair[np.triu_indices(6,1)].min(),1.)
            self.assertEqual(h["physical_patch_ids"],[])
            self.assertFalse(h["physical_surface_verified"]);self.assertFalse(h["geometry_fit_eligible"])
        self.assertFalse(r["dependencies"]["dense_XYZ_used_for_point_selection"])
        self.assertEqual(r["source_strategy"],"projected_cuboid_face_prior")
        self.assertEqual(len(r["face_diagnostics"]),6)
        legacy=self.run_proposals(face_inputs=self.make_face(self.projected_output))
        self.assertFalse(any(h["status"]=="prompt_hypothesis" for h in legacy["hypotheses"]))

    def test_projected_rejects_wrong_native_body_and_domain(self):
        with self.assertRaises(ValueError):
            self.projected_proposals(body_binding=replace(self.binding,track_id="track_01"))
        with self.assertRaises(ValueError):
            self.projected_proposals(body_mask=np.zeros_like(self.mask))
        with self.assertRaises(ValueError):
            self.projected_proposals(body_binding=replace(self.binding,frame_id=782))
        with self.assertRaises(ValueError):
            self.projected_proposals(body_binding=replace(self.binding,pixel_space_id="foreign_pixels"))

    def test_projected_no_overlap_or_insufficient_points_remains_unknown(self):
        # Original fixture geometry projects away from this actual body support.
        r=self.projected_proposals(face_inputs=self.face)
        self.assertFalse(any(h["status"]=="prompt_hypothesis" for h in r["hypotheses"]))
        from eval.action4_face_prompt_proposals import ProjectedFacePromptThresholds
        r=self.projected_proposals(thresholds=ProjectedFacePromptThresholds(.1,4,100.,1,1))
        self.assertFalse(any(h["status"]=="prompt_hypothesis" for h in r["hypotheses"]))
        self.assertFalse(r["native_face_mask_produced"])

    def test_projected_envelope_consumed_by_real_load_jobs(self):
        from data.action4_surface_snapshot import save_snapshot
        from eval.action4_face_native_pilot import load_jobs
        from model.action4_sam_session_contracts import RegionSpec
        proposal=self.projected_proposals()
        snapshot=make_snapshot(frame=self.frame,prediction=self.pred,output=self.projected_output,index=0,
            A112=self.roi.affine(112),A448=self.roi.affine(),K_native=self.K,
            K112=self.roi.camera(self.K,112).astype(np.float32),canonical=np.asarray(self.contract.canonical8_m,np.float32),
            models=self.models,video_context=self.context)
        sp=self.folder/"projected_surface.npz";save_snapshot(sp,snapshot)
        bp=self.folder/"projected_body.npz"
        with bp.open("xb") as stream:np.savez_compressed(stream,mask=self.mask)
        pr=self.owner.write(self.folder/"projected_proposal.json",proposal)
        selected=[h["hypothesis_identity"] for h in proposal["hypotheses"] if h["status"]=="prompt_hypothesis"]
        self.assertTrue(selected)
        cfg={"face_pilot":{"proposal_ref":pr,"surface_snapshot_ref":self.owner.ref(sp),"body_mask_ref":self.owner.ref(bp),
             "expected_models":self.models,"hypothesis_ids":selected,"face_generation":0}}
        first=self.replay.doc["frames"][3]
        region=RegionSpec("projected_test",(783,784),(first["pts"],first["pts"]+400),
            self.replay.region.time_base,"a"*64,self.frame.pixel_space_id,40,40)
        jobs=load_jobs(cfg,{"frames":[first]},region)
        self.assertEqual(len(jobs),len(selected))
        self.assertTrue(all(j["labels"]==[1,1,1,1,0,0] for j in jobs))
        cfg["face_pilot"]["hypothesis_ids"]=[selected[0],selected[0]]
        with self.assertRaises(ValueError):load_jobs(cfg,{"frames":[first]},region)

if __name__=="__main__":
    unittest.main()
