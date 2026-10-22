"""Bounded diagnostic analytic arm seeds; original +/-25 Euler interface untouched."""
from __future__ import annotations
import math
import numpy as np
from data import action4_full_rig_fit as fk

def nearest_rotation(raw,tolerance=1e-5):
 raw=np.asarray(raw,float)
 if raw.shape!=(3,3) or not np.isfinite(raw).all():raise ValueError("Finite 3x3 rotation input required")
 if not math.isfinite(tolerance) or tolerance<=0:raise ValueError("Finite positive rotation tolerance required")
 u,_,vh=np.linalg.svd(raw);R=u@vh
 if np.linalg.det(R)<0:u[:,-1]*=-1;R=u@vh
 if np.max(abs(R-raw))>tolerance:raise ValueError("Rotation correction exceeds numerical tolerance")
 return R
def align_rotation(a,b):
 a=np.array(a,float);b=np.array(b,float)
 if a.shape!=(3,) or b.shape!=(3,) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError("Finite 3-vectors required")
 an,bn=float(np.linalg.norm(a)),float(np.linalg.norm(b))
 if not math.isfinite(an) or not math.isfinite(bn) or an<=0 or bn<=0:raise ValueError("Nonzero finite direction norms required")
 a/=an;b/=bn
 cross=np.cross(a,b);dot=float(np.clip(a@b,-1,1));n=np.linalg.norm(cross)
 # Near antiparallel, 1+dot loses precision or rounds to zero. Keep the
 # signed small offset using normalized sin/cos instead of snapping to pi.
 if dot<-1+1e-6 and n>0:
  axis=cross/n;axis-=a*(axis@a);axis/=np.linalg.norm(axis)
  K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
  length=math.hypot(float(n),dot);sine=float(n)/length;cosine=dot/length
  return np.eye(3)+sine*K+(1-cosine)*(K@K)
 if n<1e-12:
  if dot>0:return np.eye(3)
  basis=np.eye(3)[np.argmin(abs(a))];axis=np.cross(a,basis);axis/=np.linalg.norm(axis)
  return 2*np.outer(axis,axis)-np.eye(3)
 K=np.array([[0,-cross[2],cross[1]],[cross[2],0,-cross[0]],[-cross[1],cross[0],0]])
 return np.eye(3)+K+K@K/(1+dot)
def sphere_circle_near(c1,c2,r1,r2,prior,tol=1e-10):
 c1,c2,h=np.asarray(c1,float),np.asarray(c2,float),np.asarray(prior,float)
 if any(v.shape!=(3,) or not np.isfinite(v).all() for v in (c1,c2,h)):raise ValueError("Finite sphere centres/prior 3-vectors required")
 r1,r2,tol=float(r1),float(r2),float(tol)
 if not np.isfinite([r1,r2,tol]).all() or min(r1,r2,tol)<=0:raise ValueError("Finite positive radii and tolerance required")
 delta=c2-c1;d=float(np.linalg.norm(delta))
 if not math.isfinite(d):raise ValueError("Non-finite sphere centre distance")
 base={"center_distance_m":d,"radii_m":[r1,r2]}
 if d<tol:
  gap=max(abs(r1-r2)-d,0.)
  if gap>0:return {**base,"status":"no_intersection","point":None,"gap_m":gap,"reason":"concentric_or_near_concentric_unequal_radii"}
  return {**base,"status":"coincident_centres_underdetermined","point":None,"gap_m":0.}
 gap=max(d-r1-r2,abs(r1-r2)-d,0.)
 if gap>tol:return {**base,"status":"no_intersection","point":None,"gap_m":gap}
 axis=delta/d;a=(r1*r1-r2*r2+d*d)/(2*d);center=c1+a*axis;rho2=r1*r1-a*a
 if rho2<-(tol*max(r1,r2,1.)):return {**base,"status":"no_intersection_numerical","point":None,"gap_m":gap}
 radius=math.sqrt(max(0,rho2));toward=h-center;toward-=axis*(axis@toward);length=np.linalg.norm(toward)
 ambiguous=bool(length<tol and radius>tol)
 if length<tol:
  basis=np.eye(3)[np.argmin(abs(axis))];toward=np.cross(axis,basis);toward/=np.linalg.norm(toward)
 else:toward/=length
 point=center+radius*toward
 error=max(abs(np.linalg.norm(point-c1)-r1),abs(np.linalg.norm(point-c2)-r2))
 if not np.isfinite(point).all() or not math.isfinite(error):raise ValueError("Non-finite sphere intersection result")
 return {**base,"status":"tangent" if radius<=tol else "circle","point":point.tolist(),"gap_m":gap,
 "circle_center_m":center.tolist(),"circle_radius_m":radius,"circle_axis":axis.tolist(),
 "prior_projection_ambiguous":ambiguous,"sphere_closure_max_error_m":float(error),
 "point_selection":"Nearest point to engineeringhead prior; not an independently measured observer."}
def geometry(snapshot,candidate):
 out={}
 for side in ("left","right"):
  upper=side+"_upper_arm";fore=side+"_forearm";Mu=np.array(snapshot["bones"][upper]["rest_matrix"]);Mf=np.array(snapshot["bones"][fore]["rest_matrix"])
  F=np.array(snapshot["brace_rest_body"][side])@fk.translation(.003)
  Pcap=np.array(candidate["target_B"][side])@np.linalg.inv(F)@Mf
  out[side]={"shoulder_body":Mu[:3,3],"upper_length":snapshot["bones"][upper]["length_m"],
   "fore_length":snapshot["bones"][fore]["length_m"],"P_fore_capture":Pcap,"G":np.linalg.inv(Mf)@F}
 return out
def observer_for_rotation(R,geom,config):
 centres={s:g["shoulder_body"]-R.T@g["P_fore_capture"][:3,3] for s,g in geom.items()}
 circle=sphere_circle_near(centres["left"],centres["right"],geom["left"]["upper_length"],geom["right"]["upper_length"],
 config["observer_center_prior_m"],config["analytic_circle_tolerance_m"])
 if circle["point"] is None:return None,circle
 O=np.array(circle["point"]);E=np.eye(4);E[:3,:3]=R;E[:3,3]=-R@O
 return E,circle
def deterministic_initial_rotation(geom,config):
 R0=np.array(config["R0_capture_from_body"],float)
 shoulders=geom["right"]["shoulder_body"]-geom["left"]["shoulder_body"]
 elbows=geom["right"]["P_fore_capture"][:3,3]-geom["left"]["P_fore_capture"][:3,3]
 return align_rotation(R0@shoulders,elbows)@R0
def matrix_fk(snapshot,local_Q):
 allowed={s+"_"+p for s in ("left","right") for p in ("upper_arm","forearm","hand")}
 if set(local_Q)-allowed:raise ValueError("Only scoped arm/hand matrices; no root/finger/scale overrides")
 for Q in local_Q.values():
  fk.proper(Q,tolerance=1e-8)
  if np.max(abs(np.array(Q)[:3,3]))>1e-12:raise ValueError("Native matrix adapter forbids joint translation")
 P={};pending=dict(snapshot["bones"])
 while pending:
  progress=False
  for name,b in list(pending.items()):
   parent=b["parent"]
   if parent is not None and parent not in P:continue
   M=np.array(b["rest_matrix"]);Q=np.array(local_Q.get(name,np.eye(4)))
   P[name]=M@Q if parent is None else P[parent]@np.linalg.inv(np.array(snapshot["bones"][parent]["rest_matrix"]))@M@Q
   del pending[name];progress=True
  if not progress:raise ValueError("Invalid hierarchy")
 return P
def construct_seed(snapshot,candidate,E,config,wrist_pose=None):
 from scipy.spatial.transform import Rotation
 geom=geometry(snapshot,candidate);Einv=np.linalg.inv(E);Qs={};targets={};checks={}
 for side,g in geom.items():
  Pu=np.eye(4);Mu=np.array(snapshot["bones"][side+"_upper_arm"]["rest_matrix"]);Mf=np.array(snapshot["bones"][side+"_forearm"]["rest_matrix"])
  Pf=Einv@g["P_fore_capture"];elbow=Pf[:3,3];vector=elbow-g["shoulder_body"]
  closure=float(np.linalg.norm(vector)-g["upper_length"])
  if abs(closure)>config["length_closure_tolerance_m"]:
   return {"status":"rejected_length_closure","side":side,"closure_error_m":closure,"local_Q":None}
  Rupper=align_rotation(Mu[:3,1],vector)@Mu[:3,:3]
  Pu[:3,:3]=nearest_rotation(Rupper,config["nearest_SO3_correction_max"]);Pu[:3,3]=g["shoulder_body"]
  rawupper=np.linalg.inv(Mu)@Pu
  rawfore=np.linalg.inv(Mf)@Mu@np.linalg.inv(Pu)@Pf
  detail={"upper_length_closure_error_m":closure,"unprojected_target_elbow_body":elbow.tolist(),
  "raw_local_translation_max_m":max(float(abs(rawupper[:3,3]).max()),float(abs(rawfore[:3,3]).max()))}
  if detail["raw_local_translation_max_m"]>config["local_Q_translation_tolerance_m"]:
   return {"status":"rejected_nonrotational_local_Q","side":side,"details":detail,"local_Q":None}
  for part,raw in [("upper_arm",rawupper),("forearm",rawfore)]:
   orth=float(abs(raw[:3,:3].T@raw[:3,:3]-np.eye(3)).max())
   if orth>config["raw_Q_rotation_orthogonality_tolerance"]:raise ValueError("Non-numerical raw Q rotation defect")
   Q=np.eye(4);Q[:3,:3]=nearest_rotation(raw[:3,:3],config["nearest_SO3_correction_max"]);Qs[side+"_"+part]=Q
   detail[part+"_local_XYZ_deg"]=Rotation.from_matrix(Q[:3,:3]).as_euler("xyz",degrees=True).tolist()
   detail[part+"_SO3_correction_max"]=float(abs(Q[:3,:3]-raw[:3,:3]).max())
  angles=(wrist_pose or {}).get(side,[0,0,0]);fk.validate_pose({side+"_hand":angles})
  if abs(angles[1])>1e-12:raise ValueError("No wrist localY twist in this diagnostic")
  hand=np.eye(4);hand[:3,:3]=fk.euler_xyz(angles);Qs[side+"_hand"]=hand;checks[side]=detail;targets[side]=Pf
 P=matrix_fk(snapshot,Qs)
 for side,g in geom.items():
  points=[[0,0,0],[0,g["fore_length"],0]]
  error=float(abs(fk.apply(P[side+"_forearm"],points)-fk.apply(targets[side],points)).max())
  checks[side]["pure_rotation_FK_elbow_wrist_max_error_m"]=error
  if error>config["pure_rotation_FK_target_elbow_wrist_tolerance_m"]:
   return {"status":"rejected_FK_target_difference","side":side,"details":checks[side],"local_Q":None}
 outside=[name for name,Q in Qs.items() if max(abs(Rotation.from_matrix(Q[:3,:3]).as_euler("xyz",degrees=True)))>25+1e-8]
 return {"status":"math_closed_diagnostic_only","local_Q":{n:Q.tolist() for n,Q in Qs.items()},
 "posed_bones_body":{n:p.tolist() for n,p in P.items()},"checks":checks,"outside_original25_degree_interface":outside,
 "scope":"Exact sphereclosure; only <=micrometre source floating error tolerated, no physical elbow snapping or scale. NativepureQ not production-admissible."}
