"""Rendered RGB CPU fixtures, no real-video or deployed-pose claims."""
import unittest
import numpy as np
from PIL import Image,ImageDraw
from model.action4_face_edge_pose import EdgePoseConfig,refine_face_edge_pose,observed_jacobian_diagnostics
class EdgeTests(unittest.TestCase):
 def setUp(self):
  self.K=np.array([[300.,0,80],[0,300,80],[0,0,1.]])
  self.R=np.eye(3);self.t=np.array([0.,0.,.3])
  corners=np.array([[-.03,-.02,0],[.03,-.02,0],[.03,.02,0],[-.03,.02,0]])
  self.curves=[np.linspace(corners[i],corners[(i+1)%4],35) for i in range(4)]
  self.body=np.zeros((160,160),bool);self.body[20:140,20:140]=True
  self.lcd=np.zeros_like(self.body)
  self.cfg=EdgePoseConfig(.03,.6,3.,2.,8.,1.,.15,.3,16,150,2.,1e-5,1e7,.15,.05,.05)
  self.rgb=self.render(0)
 def render(self,shift):
  image=Image.new("RGB",(160,160),"black")
  ImageDraw.Draw(image).rectangle((50+shift,60,110+shift,100),fill="white")
  return np.asarray(image).copy()
 def run_pose(self,rgb=None,**kw):
  return refine_face_edge_pose(self.rgb if rgb is None else rgb,self.body,self.lcd,self.K,self.curves,
      self.R,self.t+np.array([.003,.001,0]),cad_diagonal_m=.08,config=self.cfg,**kw)
 def test_known_render_improves_residual_and_recovers_projection(self):
  r=self.run_pose();best=min([x for x in r["solutions"] if x.get("usable")],key=lambda x:x["raw_observation_rms"])
  self.assertLess(best["mean_normal_residual_px"],best["raw_mean_normal_residual_px"])
  self.assertTrue(best["positive_depth"]);self.assertGreaterEqual(best["observation_count"],16)
  pose=r["candidate_pose"];p=np.concatenate(self.curves)
  uv=(p@np.asarray(pose["R"]).T+pose["t_m"])@self.K.T;uv=uv[:,:2]/uv[:,2,None]
  truth=(p+self.t)@self.K.T;truth=truth[:,:2]/truth[:,2,None]
  self.assertLess(float(np.median(np.linalg.norm(uv-truth,axis=1))),2.)
  self.assertFalse(r["accepted"]);self.assertFalse(r["deployed"])
 def test_real_RGB_edge_change_changes_solution(self):
  a=self.run_pose();b=self.run_pose(self.render(4))
  self.assertGreater(np.linalg.norm(np.array(a["candidate_pose"]["t_m"])-b["candidate_pose"]["t_m"]),.0005)
 def test_no_support_and_LCD_exclusion(self):
  self.body[:]=False
  self.assertEqual(self.run_pose()["reasons"],["no_supported_RGB_edges"])
  self.body[:]=True;self.lcd[:]=True
  self.assertEqual(self.run_pose()["reasons"],["no_supported_RGB_edges"])
 def test_occlusion_mask_boundary_not_used_as_full_contour(self):
  # The only RGB step lies exactly at mask support edge and is eroded away.
  rgb=np.zeros_like(self.rgb);rgb[:,80:]=255
  self.body[:]=False;self.body[20:140,80:140]=True
  self.assertEqual(self.run_pose(rgb)["rgb_edge_pixels"],0)
 def test_invalid_pixels_nan_and_pose(self):
  with self.assertRaises(ValueError):self.run_pose(self.rgb.transpose(2,0,1))
  self.K[0,0]=np.nan
  with self.assertRaises(ValueError):self.run_pose()
 def test_observation_jacobian_has_no_prior_or_damping(self):
  d=observed_jacobian_diagnostics(lambda x:np.zeros(20),np.zeros(6),1e-5,1e6)
  self.assertEqual(d["rank"],0);self.assertTrue(d["degenerate"]);self.assertIsNone(d["condition_number"])
 def test_similar_residual_distinct_pose_is_ambiguous(self):
  corners=np.array([[-.01,-.02,0],[.01,-.02,0],[.01,.02,0],[-.01,.02,0]])
  self.curves=[np.linspace(corners[i],corners[(i+1)%4],35) for i in range(4)]
  image=Image.new("RGB",(160,160),"black");draw=ImageDraw.Draw(image)
  draw.rectangle((40,60,60,100),fill="white");draw.rectangle((100,60,120,100),fill="white")
  self.rgb=np.asarray(image).copy();self.t=np.array([-.03,0,.3])
  r=refine_face_edge_pose(self.rgb,self.body,self.lcd,self.K,self.curves,self.R,self.t,cad_diagonal_m=.08,config=self.cfg,previous_trusted_pose=(self.R,np.array([.03,0,.3])))
  self.assertIn("ambiguous_similar_RGB_residual_distinct_pose",r["reasons"])
  self.assertFalse(r["geometric_candidate_eligible"])
 def test_orientation_chooses_farther_real_parallel_RGB_edge(self):
  from scipy.ndimage import sobel
  from scipy.spatial import cKDTree
  from model.action4_face_edge_pose import select_edge_correspondences
  image=Image.new("L",(80,80),0);d=ImageDraw.Draw(image)
  d.line((41,20,41,60),fill=255,width=1)  # nearer wrong vertical edge
  d.line((20,45,60,45),fill=255,width=1)  # farther correct horizontal edge
  gray=np.asarray(image,float)/255.
  gx=sobel(gray,axis=1)/8;gy=sobel(gray,axis=0)/8;mag=np.hypot(gx,gy)
  y,x=np.nonzero(mag>.05);edges=np.c_[x,y];normal=np.c_[gx[y,x],gy[y,x]]/mag[y,x,None]
  args=(cKDTree(edges),normal,np.array([[40.,40.]]),np.array([[1.,0.]]))
  _,old,_,_=select_edge_correspondences(*args,radius=8,neighbors=64,orientation_max_degrees=None)
  _,new,ok,stats=select_edge_correspondences(*args,radius=8,neighbors=64,orientation_max_degrees=15)
  self.assertTrue(ok[0]);self.assertGreater(stats["selected_neighbor_rank"][0],1)
  self.assertLess(edges[old[0],1],43);self.assertGreaterEqual(edges[new[0],1],44)
  _,_,none,_=select_edge_correspondences(*args,radius=2,neighbors=64,orientation_max_degrees=15)
  self.assertFalse(none[0])
 def test_LCD_outer_band_excludes_far_parallel_top_label_RGB_edges(self):
  from dataclasses import replace
  image=Image.new("RGB",(160,160),"black");d=ImageDraw.Draw(image)
  d.line((40,30,120,30),fill="white",width=2)
  d.line((40,60,120,60),fill="white",width=2)
  self.rgb=np.asarray(image).copy();self.lcd[65:96,60:101]=True
  legacy=self.run_pose()
  self.cfg=replace(self.cfg,face_border_max_distance_px=12.)
  bounded=self.run_pose()
  self.assertLess(bounded["rgb_edge_pixels"],legacy["rgb_edge_pixels"])
  self.assertGreater(bounded["rgb_edge_pixels"],0)
  for solution in bounded["solutions"]:
   if solution.get("usable"):
    self.assertGreater(np.asarray(solution["image_edge_matches_xy"])[:,1].min(),50)
 def test_optional_gate_parameters_fail_closed(self):
  from dataclasses import replace
  for params in ({"orientation_max_degrees":90},{"correspondence_neighbors":True},
                 {"correspondence_neighbors":129},{"face_border_max_distance_px":1}):
   with self.assertRaises(ValueError):replace(self.cfg,**params)
 def profile(self,gray,allowed=None):
  from scipy.ndimage import sobel
  from model.action4_edge_profiles import normal_profile_matches
  if allowed is None:allowed=np.ones(gray.shape,bool)
  return normal_profile_matches(sobel(gray,axis=1)/8,sobel(gray,axis=0)/8,allowed,
      np.array([[40.,40.]]),np.array([[1.,0.]]),radius=10,step_px=1.,
      tangent_half_width_px=0.,min_side_support_px=2.,min_gradient=.03,
      min_prominence=.02,max_peak_width_px=4.,ambiguity_ratio=.85,orientation_max_degrees=30.)
 def test_normal_profile_ignores_tangential_wrong_edge_and_responds_to_RGB(self):
  gray=np.zeros((80,80));gray[:,42:]+=.25;gray[45:]+=.5
  xy,n,ok,d=self.profile(gray)
  self.assertTrue(ok[0]);self.assertGreaterEqual(xy[0,1],44)
  changed=np.zeros((80,80));changed[:,42:]+=.25;changed[48:]+=.5
  xy2,_,ok2,_=self.profile(changed)
  self.assertTrue(ok2[0]);self.assertGreater(xy2[0,1],xy[0,1])
  self.assertEqual(xy[0,0],40.)
 def test_normal_profile_dual_peak_flat_and_support_unknown(self):
  bar=np.zeros((80,80));bar[36:45]=1.
  _,_,ok,d=self.profile(bar)
  self.assertFalse(ok[0]);self.assertEqual(d[0]["reason"],"ambiguous_multiple_RGB_peaks")
  self.assertGreaterEqual(d[0]["second_to_first_peak_ratio"],.85)
  _,_,ok,d=self.profile(np.zeros((80,80)))
  self.assertFalse(ok[0]);self.assertEqual(d[0]["reason"],"flat_or_weak_profile")
  _,_,ok,d=self.profile(bar,np.zeros((80,80),bool))
  self.assertFalse(ok[0]);self.assertEqual(d[0]["reason"],"insufficient_profile_support")
  allowed=np.zeros((80,80),bool);allowed[44:46]=True
  _,_,ok,d=self.profile(bar,allowed)
  self.assertFalse(ok[0])
 def test_profile_config_explicit_and_default_knn_identical(self):
  from dataclasses import replace
  a=self.run_pose()
  self.cfg=replace(self.cfg,correspondence_method="knn")
  b=self.run_pose()
  self.assertEqual(a["candidate_pose"],b["candidate_pose"])
  with self.assertRaises(ValueError):replace(self.cfg,correspondence_method="normal_profile")
  self.cfg=replace(self.cfg,correspondence_method="normal_profile",profile_step_px=1.,
      profile_tangent_half_width_px=1.,profile_min_side_support_px=2.,
      profile_min_prominence=.02,profile_max_peak_width_px=4.,profile_ambiguity_ratio=.85)
  r=self.run_pose()
  self.assertIn("normal_profiles",r["solutions"][0]["matching_diagnostics"])
  self.assertFalse(r["deployed"])
  flat=self.run_pose(np.zeros_like(self.rgb))
  self.assertEqual(flat["reasons"],["no_supported_RGB_edges"])
  self.assertTrue(flat["solutions"][0]["matching_diagnostics"]["normal_profiles"])
 def mixture_config(self):
  from dataclasses import replace
  return replace(self.cfg,correspondence_method="normal_profile_mixture",profile_sampling_domain="valid_rgb",
      profile_step_px=1.,profile_tangent_half_width_px=0.,profile_min_side_support_px=2.,
      profile_min_prominence=.02,profile_max_peak_width_px=4.,profile_ambiguity_ratio=.85,
      mixture_ambiguity_nll=.1,max_nfev=900)
 def test_mixture_single_peak_exact_gaussian_and_width_information(self):
  from model.action4_face_edge_pose import mixture_profile_statistics
  peak={"displacement_px":2.,"height":1.,"width_px":1.}
  a=mixture_profile_statistics([2.],[[peak]])[0]
  self.assertAlmostEqual(a["negative_log_likelihood"],.5*np.log(2*np.pi))
  self.assertAlmostEqual(a["observed_displacement_curvature_per_px2"],1.)
  self.assertEqual(a["posterior"],[1.])
  wide=dict(peak,width_px=7.064460135)
  b=mixture_profile_statistics([2.],[[wide]])[0]
  self.assertAlmostEqual(b["observed_displacement_curvature_per_px2"],1/9,places=6)
  self.assertGreater(b["mixture_total_variance_px2"],a["mixture_total_variance_px2"])
  two=[dict(peak,displacement_px=-3.),dict(peak,displacement_px=3.)]
  c=mixture_profile_statistics([0.],[two])[0]
  self.assertLess(c["observed_displacement_curvature_per_px2"],0.)
  np.testing.assert_allclose(c["posterior"],[.5,.5],atol=1e-15,rtol=0)
 def test_mixture_preserves_local_parallel_peaks_with_one_SE3(self):
  outer=np.array([[-.03,-.02,0],[.03,-.02,0],[.03,.02,0],[-.03,.02,0]])
  inner=outer.copy();inner[:,0]*=.8;inner[:,1]*=.7
  self.curves=[np.linspace(c[i],c[(i+1)%4],12) for c in (outer,inner) for i in range(4)]
  im=Image.new("RGB",(160,160),"black");draw=ImageDraw.Draw(im)
  draw.rectangle((50,60,110,100),fill="white");draw.rectangle((56,66,104,94),fill="black")
  self.rgb=np.asarray(im).copy();self.cfg=self.mixture_config()
  r=self.run_pose();a=r["solutions"][0]
  self.assertTrue(a["usable"]);self.assertTrue(any(len(x)>1 for x in a["mixture_peak_sources"]))
  self.assertLess(a["mean_mixture_nll"],a["raw_mean_mixture_nll"])
  self.assertFalse(r["deployed"]);self.assertFalse(r["accepted"])
  pose=r["candidate_pose"];points=np.concatenate(self.curves)
  predicted=(points@np.array(pose["R"]).T+pose["t_m"])@self.K.T
  truth=(points+self.t)@self.K.T
  error=np.linalg.norm(predicted[:,:2]/predicted[:,2,None]-truth[:,:2]/truth[:,2,None],axis=1)
  self.assertLess(float(np.median(error)),2.)
  self.assertIn("geometry_only",a["jacobian"])
  for line in a["mixture_line_diagnostics"]:
   self.assertAlmostEqual(sum(line["posterior"]),1.)
  for peaks in a["mixture_peak_sources"]:
   self.assertTrue(all(not p["reasons"] for p in peaks))
   self.assertTrue(all(p["width_px"]<=self.cfg.profile_max_peak_width_px for p in peaks))
 def test_mixture_global_repeated_pattern_remains_ambiguous(self):
  corners=np.array([[-.01,-.02,0],[.01,-.02,0],[.01,.02,0],[-.01,.02,0]])
  self.curves=[np.linspace(corners[i],corners[(i+1)%4],12) for i in range(4)]
  im=Image.new("RGB",(160,160),"black");draw=ImageDraw.Draw(im)
  draw.rectangle((40,60,60,100),fill="white");draw.rectangle((100,60,120,100),fill="white")
  self.rgb=np.asarray(im).copy();self.t=np.array([-.03,0,.3]);self.cfg=self.mixture_config()
  r=refine_face_edge_pose(self.rgb,self.body,self.lcd,self.K,self.curves,self.R,self.t,
      cad_diagonal_m=.08,config=self.cfg,previous_trusted_pose=(self.R,np.array([.03,0,.3])))
  self.assertIn("ambiguous_similar_mixture_likelihood_distinct_pose",r["reasons"])
  self.assertFalse(r["geometric_candidate_eligible"])
 def test_mixture_does_not_relax_peak_ownership_or_width(self):
  from dataclasses import replace
  self.cfg=replace(self.mixture_config(),profile_max_peak_width_px=.1)
  r=self.run_pose()
  self.assertIsNone(r["candidate_pose"])
  self.assertFalse(r["geometric_candidate_eligible"])
  for profile in r["solutions"][0]["matching_diagnostics"]["normal_profiles"]:
   self.assertEqual(profile["mixture_legal_peak_count"],0)
  self.cfg=self.mixture_config();self.body[:]=False
  self.assertEqual(self.run_pose()["reasons"],["no_supported_RGB_edges"])
 def test_object_origin_chart_pure_rotation_at_two_depths(self):
  from scipy.spatial.transform import Rotation
  from model.action4_face_edge_pose import pose_from_increment,pose_increment_record
  x=np.array([0.,.2,0.,0.,0.,0.]);delta=Rotation.from_rotvec(x[:3]).as_matrix()
  for depth in (.3,1.):
   t=np.array([0.,0.,depth])
   R,newt=pose_from_increment(x,np.eye(3),t,.08,"object_origin_delta")
   np.testing.assert_array_equal(newt,t);np.testing.assert_allclose(R,delta)
   report=pose_increment_record(x,np.eye(3),t,.08,"object_origin_delta")
   self.assertEqual(report["net_object_origin_translation_CADdiag_units"],[0.,0.,0.])
   self.assertGreater(np.linalg.norm(report["equivalent_left_translation_CADdiag_units"]),.3)
   np.testing.assert_allclose(delta@t+report["equivalent_left_translation_m"],newt,atol=1e-15)
 def test_legacy_chart_exact_original_formula_and_explicit_defaults(self):
  from scipy.spatial.transform import Rotation
  from model.action4_face_edge_pose import pose_from_increment
  from dataclasses import replace
  x=np.array([.03,-.05,.02,.1,-.2,.05]);t=np.array([.1,.2,.8])
  delta=Rotation.from_rotvec(x[:3]).as_matrix()
  R,newt=pose_from_increment(x,np.eye(3),t,.08)
  np.testing.assert_array_equal(R,delta)
  np.testing.assert_array_equal(newt,delta@t+x[3:]*.08)
  a=self.run_pose();self.cfg=replace(self.cfg,translation_chart="legacy_left");b=self.run_pose()
  self.assertEqual(a["candidate_pose"],b["candidate_pose"])
  with self.assertRaises(ValueError):replace(self.cfg,translation_chart="unregistered")
 def test_object_origin_RGB_fixture_and_Jacobian_chart(self):
  from dataclasses import replace
  self.cfg=replace(self.cfg,translation_chart="object_origin_delta")
  r=self.run_pose();s=r["solutions"][0]
  self.assertLess(s["mean_normal_residual_px"],s["raw_mean_normal_residual_px"])
  self.assertEqual(s["jacobian"]["translation_chart"],"object_origin_delta")
  self.assertTrue(s["jacobian"]["condition_comparable_only_with_same_chart"])
  increment=s["pose_increment"]
  np.testing.assert_allclose(np.asarray(increment["net_object_origin_translation_m"])/.08,
                              s["local_translation_CADdiag_units"],atol=1e-14)
  self.assertEqual(increment["translation_bound_semantics"],"net_object_origin_delta")
  zero=observed_jacobian_diagnostics(lambda x:np.zeros(12),np.zeros(6),1e-5,1e6,
      translation_chart="object_origin_delta")
  self.assertEqual(zero["rank"],0);self.assertTrue(zero["degenerate"])
 def test_object_origin_mixture_uses_same_explicit_chart(self):
  from dataclasses import replace
  self.cfg=replace(self.mixture_config(),translation_chart="object_origin_delta")
  r=self.run_pose();s=r["solutions"][0]
  self.assertEqual(s["pose_increment"]["translation_chart"],"object_origin_delta")
  self.assertEqual(s["jacobian"]["translation_chart"],"object_origin_delta")
  self.assertEqual(s["jacobian"]["geometry_only"]["translation_chart"],"object_origin_delta")
  self.assertFalse(r["prior_residual_used"]);self.assertFalse(r["deployed"])
 def test_previous_is_initializer_not_prior_and_inputs_unchanged(self):
  rgb=self.rgb.copy();curves=[c.copy() for c in self.curves]
  r=self.run_pose(previous_trusted_pose=(self.R,self.t))
  self.assertEqual(len(r["solutions"]),2);self.assertFalse(r["prior_residual_used"])
  np.testing.assert_array_equal(rgb,self.rgb)
  for a,b in zip(curves,self.curves):np.testing.assert_array_equal(a,b)
 def test_separate_RGB_sampling_keeps_candidate_ownership_and_no_mask_edges(self):
  from scipy.ndimage import sobel
  from model.action4_edge_profiles import normal_profile_matches
  gray=np.zeros((80,80));gray[45:]=1
  allowed=np.zeros_like(gray,bool);allowed[44:46]=True
  kw=dict(radius=10,step_px=1.,tangent_half_width_px=0.,min_side_support_px=2.,
    min_gradient=.03,min_prominence=.02,max_peak_width_px=4.,ambiguity_ratio=.85,orientation_max_degrees=30.)
  def run(g,owner=allowed):
   return normal_profile_matches(sobel(g,axis=1)/8,sobel(g,axis=0)/8,owner,
     np.array([[40.,40.]]),np.array([[1.,0.]]),sampling_valid=np.ones_like(owner),**kw)
  xy,normal,ok,d=run(gray)
  self.assertTrue(ok[0]);self.assertTrue(allowed[int(xy[0,1]),int(xy[0,0])])
  self.assertEqual(d[0]["sampling_domain"],"valid_RGB_separate_from_candidate_ownership")
  _,_,ok,_=run(np.zeros_like(gray));self.assertFalse(ok[0])
  wrong=np.zeros_like(gray);wrong[48:]=1
  _,_,ok,d=run(wrong);self.assertFalse(ok[0])
  self.assertTrue(any("peak_outside_candidate_ownership" in p["reasons"] for p in d[0]["peaks"]))
 def test_separate_domain_config_never_silently_changes_knn(self):
  from dataclasses import replace
  with self.assertRaises(ValueError):replace(self.cfg,profile_sampling_domain="valid_rgb")
if __name__=="__main__":unittest.main()
