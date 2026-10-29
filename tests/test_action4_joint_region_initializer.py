import unittest
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from model.action4_region_pose_initializers import refine_clear_seed_regions, _project

class JointRegionTests(unittest.TestCase):
    def fixture(self):
        K=np.array([[800.,0,320],[0,800.,240],[0,0,1]])
        R=Rotation.from_euler("xyz",[.4,-.2,.15]).as_matrix();t=np.array([0.,0.,.4])
        body=np.array([[x,y,z] for x in [-.035,.035] for y in [-.016,.016] for z in [-.022,.022]])
        theta=np.linspace(0,2*np.pi,64,endpoint=False)
        button=np.c_[-.018+.006*np.cos(theta),.003+.006*np.sin(theta),np.full(64,.023)]
        parts=[]
        for name,v in [("body",body),("button",button)]:
            mask=np.zeros((480,640),np.uint8)
            cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(_project(v,R,t,K)).astype(np.int32)),1)
            parts.append(dict(part_id=name,vertices_m=v,mask=mask.astype(bool),full_visibility_seed_review=True))
        return K,R,t,body,parts
    def test_joint_fit_reduces_known_fixture_projection_error(self):
        K,R,t,body,parts=self.fixture();R0=Rotation.from_rotvec([.08,-.05,.04]).as_matrix()@R;t0=t+[.003,-.002,.008]
        out=refine_clear_seed_regions(K,R0,t0,parts,cad_diagonal_m=.09)
        before=np.mean(np.linalg.norm(_project(body,R0,t0,K)-_project(body,R,t,K),axis=1))
        after=np.mean(np.linalg.norm(_project(body,np.array(out["R"]),np.array(out["t_m"]),K)-_project(body,R,t,K),axis=1))
        self.assertLess(after,before*.35);self.assertLess(out["final_cost"],out["initial_cost"])
        self.assertFalse(out["final_prediction_replaced"])
    def test_no_full_visibility_assumption_no_fit(self):
        K,R,t,_,parts=self.fixture();parts[0]["full_visibility_seed_review"]=False
        with self.assertRaises(ValueError):refine_clear_seed_regions(K,R,t,parts,cad_diagonal_m=.09)
    def test_empty_region_rejected(self):
        K,R,t,_,parts=self.fixture();parts[1]["mask"][:]=False
        with self.assertRaises(ValueError):refine_clear_seed_regions(K,R,t,parts,cad_diagonal_m=.09)
    def test_estimated_landmarks_are_consumed_and_never_claim_calibration(self):
        K,R,t,body,parts=self.fixture()
        points=body[[0,2,4,6]]
        group=dict(points_object_m=points,pixels_native=_project(points,R,t,K),group_weight=1.,
            XYZ_source="existing_appearance_texture_mapping_estimate",real_metric_calibration=False)
        r=refine_clear_seed_regions(K,R,t,parts,cad_diagonal_m=.09,landmark_groups=[group])
        moved=dict(group,pixels_native=group["pixels_native"]+[8.,0.])
        other=refine_clear_seed_regions(K,R,t,parts,cad_diagonal_m=.09,landmark_groups=[moved])
        self.assertEqual(r["estimated_landmark_groups"],1)
        self.assertGreater(np.linalg.norm(np.asarray(r["t_m"])-other["t_m"]),1e-5)
        self.assertFalse(other["geometric_accuracy_verified"])
    def test_nonfinite_landmark_and_unknown_metric_binding_rejected(self):
        K,R,t,body,parts=self.fixture();points=body[[0,2,4,6]]
        group=dict(points_object_m=points,pixels_native=_project(points,R,t,K),group_weight=1.,
            XYZ_source="existing_appearance_texture_mapping_estimate",real_metric_calibration=True)
        with self.assertRaises(ValueError):refine_clear_seed_regions(K,R,t,parts,cad_diagonal_m=.09,landmark_groups=[group])
        group["real_metric_calibration"]=False;group["pixels_native"][0,0]=np.nan
        with self.assertRaises(ValueError):refine_clear_seed_regions(K,R,t,parts,cad_diagonal_m=.09,landmark_groups=[group])

    def test_computational_canvas_and_K_translation_leave_objective_unchanged(self):
        K,R,t,body,parts=self.fixture();union=parts[0]["mask"]|parts[1]["mask"]
        yy,xx=np.nonzero(union);l=int(xx.min())-3;top=int(yy.min())-3;r=int(xx.max())+4;b=int(yy.max())+4
        costs=[]
        for pad in [0,128]:
            cropped=[dict(p,mask=np.pad(p["mask"][top:b,l:r],pad)) for p in parts]
            shifted=K.copy();shifted[0,2]+=-l+pad;shifted[1,2]+=-top+pad
            out=refine_clear_seed_regions(shifted,R,t+[.04,0,0],cropped,cad_diagonal_m=.09,max_nfev=1)
            costs.append(out["initial_cost"])
        self.assertAlmostEqual(costs[0],costs[1],places=7)
    def test_exterior_projection_retains_distance_information(self):
        K,R,t,body,parts=self.fixture();union=parts[0]["mask"]|parts[1]["mask"]
        yy,xx=np.nonzero(union);l=int(xx.min())-3;top=int(yy.min())-3;r=int(xx.max())+4;b=int(yy.max())+4
        cropped=[dict(p,mask=p["mask"][top:b,l:r]) for p in parts]
        shifted=K.copy();shifted[0,2]-=l;shifted[1,2]-=top
        near=refine_clear_seed_regions(shifted,R,t+[.12,0,0],cropped,cad_diagonal_m=.09,max_nfev=1)
        far=refine_clear_seed_regions(shifted,R,t+[.16,0,0],cropped,cad_diagonal_m=.09,max_nfev=1)
        self.assertGreater(far["initial_cost"],near["initial_cost"])

    def test_partial_region_reports_only_eligible_observed_residuals(self):
        K,R,t,body,parts=self.fixture()
        parts[0].update(partial_boundary=True,observed_boundary_eligibility=np.ones_like(parts[0]["mask"]),full_visibility_seed_review=False)
        r=refine_clear_seed_regions(K,R,t,parts,cad_diagonal_m=.09,max_nfev=1)
        self.assertEqual(r["parts"]["body"]["observation_direction"],"observed_to_model_only")
        self.assertEqual(r["parts"]["body"]["residual_count"],96)
        self.assertEqual(r["parts"]["button"]["residual_count"],192)
    def test_eligibility_window_does_not_create_a_new_mask_boundary(self):
        from scipy.ndimage import binary_erosion
        K,R,t,body,parts=self.fixture()
        parts[0].update(partial_boundary=True,observed_boundary_eligibility=binary_erosion(parts[0]["mask"],iterations=3),full_visibility_seed_review=False)
        with self.assertRaises(ValueError):refine_clear_seed_regions(K,R,t,parts,cad_diagonal_m=.09,max_nfev=1)

if __name__=="__main__":unittest.main()
