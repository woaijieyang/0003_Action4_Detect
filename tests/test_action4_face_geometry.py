"""CPU geometry tests against the real topology/parent bytes; no model/masks."""
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from data.action4_adapter import DataAdapter
from model.action4_face_geometry import FaceGeometryContract,project_six_faces

ROOT=Path("/data/zy/vln/a-test")


class FaceGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=(ROOT/"configs/model/action4_six_face_anchor_contract_v001.json").read_bytes()
        cls.doc=json.loads(cls.raw)
        cls.parent=(ROOT/cls.doc["parent_asset"]["path"]).read_bytes()
        cls.contract=FaceGeometryContract.from_bytes(cls.raw,cls.parent)
        cls.points=np.asarray(cls.contract.canonical8_m)
        cls.K=np.array([[400.,0.,640.],[0.,400.,360.],[0.,0.,1.]])

    def project(self,R=None,t=None,**changes):
        values=dict(canonical8_m=self.points,contract=self.contract,R=np.eye(3) if R is None else R,
            t_m=np.array([0.,0.,1.]) if t is None else t,K=self.K,units="m",
            pixel_space_id="raw_video_native_integer_centers",image_size_wh=(1280,720),
            min_depth_m=1e-6,grazing_cos_threshold=1e-6)
        values.update(changes)
        return project_six_faces(**values)

    def test_single_face_and_three_faces_without_topk_selection(self):
        one=self.project()
        self.assertEqual([f["face_id"] for f in one["faces"] if f["geometric_front_facing"]],[4])
        three=self.project(t=[.2,.2,1.])
        self.assertEqual([f["face_id"] for f in three["faces"] if f["geometric_front_facing"]],[0,2,4])
        self.assertEqual([c["corner_id"] for c in three["corners"]],list(range(8)))
        self.assertTrue(all(f["diagnostic_status"]=="visible_candidate"
                            for f in three["faces"] if f["geometric_front_facing"]))

    def test_rotation_translation_match_existing_native_projector(self):
        theta=.4
        rotation=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        t=np.array([.1,-.04,1.2])
        out=self.project(R=rotation,t=t)
        transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=t
        uv,z=DataAdapter.project_corners(self.points,self.K,transform,min_depth=1e-6)
        np.testing.assert_array_equal(np.asarray([c["native_uv"] for c in out["corners"]]),uv)
        np.testing.assert_allclose([c["camera_xyz_m"] for c in out["corners"]],self.points@rotation.T+t)
        for face in out["faces"]:
            ids=face["corner_ids"]
            np.testing.assert_allclose(face["center_camera_m"],self.points[ids].mean(0)@rotation.T+t)
            np.testing.assert_allclose(face["outward_normal_camera"],rotation@np.asarray(face["outward_normal_object"]))

    def test_shared_corners_and_twelve_edges_are_consistent(self):
        out=self.project(t=[.2,.2,1.]);corners=out["corners"]
        counts=Counter()
        for face in out["faces"]:
            ids=face["corner_ids"]
            self.assertEqual(face["quad_native_uv"],[corners[i]["native_uv"] for i in ids])
            for i in range(4):counts[tuple(sorted((ids[i],ids[(i+1)%4])))]+=1
            xyz=np.asarray(face["quad_camera_xyz_m"])
            normal=np.cross(xyz[1]-xyz[0],xyz[2]-xyz[1])
            self.assertGreater(float(np.dot(normal,face["outward_normal_camera"])),0)
        self.assertEqual(len(counts),12)
        self.assertEqual(set(counts.values()),{2})
        self.assertEqual(len(out["edges"]),12)
        for edge in out["edges"]:
            self.assertEqual(edge["native_endpoints"],[corners[i]["native_uv"] for i in edge["corner_ids"]])

    def test_outside_points_never_clamped_to_image_or_ROI(self):
        out=self.project(t=[10.,0.,1.])
        self.assertTrue(all(c["native_uv"][0]>1280 for c in out["corners"]))
        self.assertTrue(all(f["diagnostic_status"]=="outside" for f in out["faces"]))
        self.assertTrue(any(f["geometric_front_facing"] for f in out["faces"]))

    def test_grazing_threshold_is_explicit_geometric_only(self):
        half_x=self.points[:,0].max()
        out=self.project(t=[half_x,0.,1.])
        face=out["faces"][0]
        self.assertAlmostEqual(face["view_cosine"],0.)
        self.assertEqual(face["orientation"],"grazing")
        self.assertFalse(face["geometric_front_facing"])
        self.assertFalse(out["thresholds"]["calibrated_visibility_gate"])

    def test_negative_and_near_zero_depth_never_divided(self):
        for t in ([0.,0.,-1.],[0.,0.,0.],[0.,0.,self.points[:,2].max()]):
            with np.errstate(divide="raise",invalid="raise"):
                out=self.project(t=t)
            json.dumps(out,allow_nan=False)
            for corner in out["corners"]:
                if not corner["positive_depth"]:self.assertIsNone(corner["native_uv"])
            for face in out["faces"]:
                if not face["positive_depth"]:
                    self.assertFalse(face["geometric_front_facing"])
                    self.assertEqual(face["diagnostic_status"],"nonpositive_or_near_depth")

    def test_invalid_rotation_nan_shape_units_pixel_space_and_order_rejected(self):
        for kwargs in [
            {"R":np.diag([-1.,1.,1.])},{"R":np.eye(3)*1.01},{"R":np.full((3,3),np.nan)},
            {"t_m":[0.,0.]},{"K":np.full((3,3),np.nan)},{"canonical8_m":self.points[::-1]},
            {"canonical8_m":self.points*1000},{"units":"mm"},{"pixel_space_id":"ROI112"},
            {"grazing_cos_threshold":-1.},{"min_depth_m":0.}]:
            with self.assertRaises((ValueError,TypeError)):self.project(**kwargs)

    def test_source_identity_and_topology_tamper_rejected(self):
        out=self.project()
        self.assertEqual(out["source_contract_sha256"],hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(out["parent_asset"]["sha256"],hashlib.sha256(self.parent).hexdigest())
        for kind in ("face","corner","winding","normal"):
            doc=copy.deepcopy(self.doc)
            if kind=="face":doc["faces"][0]["face_id"]=8
            elif kind=="corner":doc["corner_ids"][0]=False
            elif kind=="winding":doc["faces"][0]["corner_ids"].reverse()
            else:doc["faces"][0]["outward_normal_object"]=[1,0,0]
            with self.assertRaises(ValueError):
                FaceGeometryContract.from_bytes(json.dumps(doc).encode(),self.parent)
        with self.assertRaises(ValueError):
            FaceGeometryContract.from_bytes(self.raw,self.parent+b" ")
        forged=FaceGeometryContract(self.contract.canonical8_m,"a"*64,"b"*64,"invented")
        with self.assertRaises(TypeError):self.project(contract=forged)

    def test_no_observation_or_hidden_mask_claims(self):
        out=self.project()
        self.assertFalse(out["observation_evidence_used"])
        self.assertFalse(out["mask_generated"])
        self.assertFalse(out["face_tracking_accepted"])
        for face in out["faces"]:
            self.assertEqual(face["observation_status"],"not_measured_geometry_only")
            self.assertEqual(face["occlusion_status"],"unknown")
            self.assertNotIn("mask",face)
            self.assertNotIn("observed",face)



    def test_exact_float32_snapshot_encoding_uses_contract_float64_projection(self):
        # make_snapshot stores cpu(canonical) as canonical_m, preserving dtype;
        # SHAPES is the actual snapshot array contract, not an invented adapter.
        from data.action4_surface_snapshot import SHAPES
        self.assertEqual(SHAPES["canonical_m"],(8,3))
        canonical_m=np.asarray(self.points,dtype=np.float32)
        snapshot_array=np.frombuffer(np.ascontiguousarray(canonical_m).tobytes(),
                                     dtype=canonical_m.dtype).reshape(canonical_m.shape)
        expected=self.project()
        actual=self.project(canonical8_m=snapshot_array)
        self.assertEqual(actual["canonical_input_representation"]["dtype"],"float32")
        self.assertEqual(actual["canonical_input_representation"]["encoding"],"exact_contract_float32")
        self.assertEqual(actual["canonical_input_representation"]["projection_geometry"],"authenticated_contract_float64")
        self.assertEqual(actual["corners"],expected["corners"])
        self.assertEqual(actual["faces"],expected["faces"])
        self.assertEqual(actual["edges"],expected["edges"])
        self.assertFalse(actual["camera_calibration_verified"])
        self.assertIn("not proof of rectification",actual["distortion_handling"])

    def test_float16_one_ulp_perturbation_and_wrong_float64_encoding_rejected(self):
        changed=self.points.astype(np.float32)
        changed[0,0]=np.nextafter(changed[0,0],np.float32(np.inf))
        for points in (self.points.astype(np.float16),changed,
                       self.points.astype(np.float32).astype(np.float64)):
            with self.assertRaises(ValueError):self.project(canonical8_m=points)
        changed64=self.points.copy()
        changed64[0,0]=np.nextafter(changed64[0,0],np.inf)
        with self.assertRaises(ValueError):self.project(canonical8_m=changed64)

    def test_degenerate_contract_zero_extent_rejected_even_with_matching_parent_sha(self):
        parent=json.loads(self.parent);doc=copy.deepcopy(self.doc)
        for point in parent["corners_3d_mm"]:point[0]=0.
        for point in doc["canonical8_m"]:point[0]=0.
        encoded=json.dumps(parent).encode()
        doc["parent_asset"]["sha256"]=hashlib.sha256(encoded).hexdigest()
        with self.assertRaisesRegex(ValueError,"positive extent"):
            FaceGeometryContract.from_bytes(json.dumps(doc).encode(),encoded)


if __name__=="__main__":
    unittest.main()
