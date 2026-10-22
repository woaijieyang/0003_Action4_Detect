"""Diagnostic-only shared-camera FK forward. No fitting, rendering or asset writing."""
from __future__ import annotations
import argparse,json,hashlib,math,sys
from pathlib import Path
import numpy as np
ROOT=Path("/data/zy/vln/a-test")
ASSET="data/action4_scene_assets/finger_rig_init_v001/build_v007/finger_rig_initial.blend"
ASSET_SHA="9e04944c022cfb7e14f688d1a60541f368a16f7318af85e1a38e0cc034eaee12"
G1="data/action4_scene_assets/global_rig_init_v001/validated_build_v002/rig_contract.json"
FINGER="data/action4_scene_assets/finger_rig_init_v001/build_v007/finger_contract.json"
PALM="outputs/action4_realism_reassessment_v001/finger_rig_init_v001/palm_frame_semantic_evidence.json"
LEFT="outputs/shared_left_attachment_fit_v001/results.json"
RIGHT="outputs/action4_realism_reassessment_v001/shared_attachment_shape_coupled_v001/results.json"
SEMANTICS="outputs/action4_realism_reassessment_v001/hand_mount_visible_evidence_v001/observations.json"
MANIFEST="data/DJI_Action4_construct/r3_v001/asset_manifest.json"
FRAME="data/action4_real/cache_v001/frames/frame_00000000.png"
CONTROL_NAMES=tuple(f"{s}_{p}" for s in ("left","right") for p in ("upper_arm","forearm","hand"))
POSES={"neutral":{},"small_bilateral":{"left_upper_arm":[7,-9,11],"left_forearm":[12,8,-6],"right_upper_arm":[-8,10,-5],"right_forearm":[5,-7,9]},
 "hand_only":{"left_hand":[10,-4,6],"right_hand":[-9,5,-8]}}
def bounded(p):
 p=(ROOT/p).resolve()
 if not p.is_relative_to(ROOT):raise ValueError("Outside project")
 return p
def sha(p):return hashlib.sha256(bounded(p).read_bytes()).hexdigest()
def read(p):return json.loads(bounded(p).read_text())
def write(p,value):
 p=bounded(p)
 if not p.is_relative_to(ROOT/"outputs"):raise ValueError("Diagnostic output must be under outputs")
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("x") as f:json.dump(value,f,indent=2,allow_nan=False)
def translation(y):
 T=np.eye(4);T[1,3]=y;return T
def proper(T,tolerance=3e-6):
 T=np.asarray(T,float)
 if T.shape!=(4,4) or not np.isfinite(T).all() or np.max(abs(T[3]-[0,0,0,1]))>1e-8:raise ValueError("Invalid SE3")
 R=T[:3,:3]
 if np.max(abs(R.T@R-np.eye(3)))>tolerance or abs(np.linalg.det(R)-1)>tolerance:raise ValueError("Scale/reflection not allowed")
 return T
def euler_xyz(degrees):
 x,y,z=np.radians(degrees);cx,sx=np.cos(x),np.sin(x);cy,sy=np.cos(y),np.sin(y);cz,sz=np.cos(z),np.sin(z)
 return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
def validate_pose(pose):
 if not isinstance(pose,dict) or set(pose)-set(CONTROL_NAMES):raise ValueError("Only existing six arm/hand joint rotations; no root/scale/mount/finger override")
 for v in pose.values():
  if not isinstance(v,list) or len(v)!=3 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or abs(x)>25 for x in v):
   raise ValueError("Existing engineering +/-25 degree domain only")
def fk_bones(snapshot,pose):
 validate_pose(pose);result={};pending=dict(snapshot["bones"])
 while pending:
  progress=False
  for name,b in list(pending.items()):
   parent=b["parent"]
   if parent is not None and parent not in result:continue
   M=np.array(b["rest_matrix"]);Q=np.eye(4);Q[:3,:3]=euler_xyz(pose.get(name,[0,0,0]))
   result[name]=M@Q if parent is None else result[parent]@np.linalg.inv(np.array(snapshot["bones"][parent]["rest_matrix"]))@M@Q
   del pending[name];progress=True
  if not progress:raise ValueError("Cyclic/missing bone parent")
 return result
def apply(T,p):return np.asarray(p)@np.asarray(T)[:3,:3].T+np.asarray(T)[:3,3]
def project(points,f):
 points=np.asarray(points,float);valid=points[:,2]>.005;xy=[]
 for p,ok in zip(points,valid):xy.append((p[:2]/p[2]*f+[1623.5,1231.5]).tolist() if ok else None)
 return {"xy":xy,"positive_depth":valid.tolist(),"inside_native":[bool(ok and 0<=v[0]<3248 and 0<=v[1]<2464) for v,ok in zip(xy,valid)]}
def source_paths():
 return [ASSET,G1,FINGER,PALM,LEFT,RIGHT,SEMANTICS,MANIFEST,FRAME,"src/data/action4_global_rig.py","src/data/action4_finger_rig.py",
 "src/data/action4_full_rig_fit.py","scripts/action4_full_rig_forward.py"]
def check_source_identities(sources):
 for p,h in sources.items():
  if sha(p)!=h:raise ValueError("Source changed: "+p)
def snapshot_digest(snapshot):
 value={k:v for k,v in snapshot.items() if k!="invariants_sha256"}
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def validate_snapshot(snapshot):
 if snapshot.get("schema")!="action4.full_rig_forward_snapshot.v1" or snapshot.get("status")!="diagnostic_only" or snapshot.get("dataset_generation_allowed") is not False:
  raise ValueError("Diagnostic snapshot contract required")
 if snapshot.get("invariants_sha256")!=snapshot_digest(snapshot):raise ValueError("Snapshot payload changed")
 if len(snapshot["bones"])!=35 or set(snapshot["sources"])!=set(source_paths()) or snapshot["sources"].get(ASSET)!=ASSET_SHA:
  raise ValueError("Snapshot source/bone coverage changed")
 for name,b in snapshot["bones"].items():
  proper(b["rest_matrix"])
  if not math.isfinite(b["length_m"]) or b["length_m"]<=0:raise ValueError("Invalid bone length")
 check_source_identities(snapshot["sources"])

def draw_overview(snapshot,result,path):
 from PIL import Image,ImageDraw,ImageFont
 im=Image.new("RGB",(1800,1000),"white");dr=ImageDraw.Draw(im);font=ImageFont.load_default(size=23);small=ImageFont.load_default(size=18)
 points=[np.array(b["head_tail_capture"]) for b in result["bones"].values()]
 for data in result["sides"].values():
  points.extend([np.array(data["canonical8_capture"]),apply(data["target_C"],snapshot["camera_corners_m"])])
 allpoints=np.concatenate(points+[np.zeros((1,3))]);corners=np.array(snapshot["camera_corners_m"])
 edges=[(i,j) for i,a in enumerate(corners) for j,b in enumerate(corners) if j>i and np.sum(abs(a-b)>1e-8)==1]
 for panel,axes in enumerate(((0,1),(0,2))):
  rect=(panel*900+65,145,panel*900+850,880);lo=allpoints[:,axes].min(0);hi=allpoints[:,axes].max(0)
  center=(lo+hi)/2;extent=max((hi-lo).max()*1.15,.2);lo=center-extent/2;hi=center+extent/2
  def xy(p):
   q=np.asarray(p)[list(axes)];u=(q-lo)/(hi-lo)
   return (rect[0]+float(u[0])*(rect[2]-rect[0]),rect[3]-float(u[1])*(rect[3]-rect[1]))
  dr.rectangle(rect,outline="#aaaaaa",width=2)
  for f in np.linspace(0,1,5):
   x=rect[0]+f*(rect[2]-rect[0]);y=rect[3]-f*(rect[3]-rect[1])
   dr.line((x,rect[1],x,rect[3]),fill="#eeeeee");dr.line((rect[0],y,rect[2],y),fill="#eeeeee")
   dr.text((x-20,rect[3]+8),f"{lo[0]+f*(hi[0]-lo[0]):.2f}",fill="black",font=small)
   dr.text((rect[0]-60,y-10),f"{lo[1]+f*(hi[1]-lo[1]):.2f}",fill="black",font=small)
  for name,b in result["bones"].items():
   pts=b["head_tail_capture"];color="#279c27" if name.startswith("left") else "#db6e00" if name.startswith("right") else "#8555aa"
   dr.line([xy(p) for p in pts],fill=color,width=3)
   if name in CONTROL_NAMES:dr.text(xy(pts[0]),name,font=small,fill=color)
  for side,data in result["sides"].items():
   color="#009eaa" if side=="left" else "#d92bbd";pred=np.array(data["canonical8_capture"]);target=apply(data["target_C"],snapshot["camera_corners_m"])
   for a,b in edges:
    dr.line([xy(target[a]),xy(target[b])],fill="#b3a000",width=2)
    dr.line([xy(pred[a]),xy(pred[b])],fill=color,width=4)
   dr.text(xy(pred.mean(0)),side+" predicted",font=small,fill=color)
  origin=xy([0,0,0]);dr.ellipse((origin[0]-5,origin[1]-5,origin[0]+5,origin[1]+5),fill="black")
  dr.text((origin[0]+8,origin[1]),"capture origin",font=small,fill="black")
  dr.text((panel*900+65,100),"Capture "+("XY" if panel==0 else "XZ")+" orthographic; metres, NOT native projection",font=font,fill="black")
  dr.text((panel*900+390,930),"X (m)",font=font,fill="black")
  dr.text((panel*900+65,900),("Y" if panel==0 else "Z depth")+" (m); all35 bones incl offscreen/behind points",font=small,fill="black")
 dr.text((20,15),result["candidate_id"]+" / complete shared-camera FK overview / neutral initialization, NOT fitted",font=font,fill="black")
 dr.text((20,50),"Green/orange skeleton; cyan/magenta predicted cameras; mustard previous body candidates; no anatomy scaling.",font=small,fill="black")
 im.save(bounded(path))

def export_snapshot(output):
 import bpy
 from data.action4_global_rig import apply_pose
 if sha(ASSET)!=ASSET_SHA:raise ValueError("Finger asset identity changed")
 sources={p:sha(p) for p in source_paths()}
 bpy.ops.wm.open_mainfile(filepath=str(bounded(ASSET)))
 rig=bpy.data.objects["global_metric_armature"];rigworld=np.array(rig.matrix_world)
 if np.max(abs(rigworld-np.eye(4)))>1e-8:raise ValueError("Unexpected body root world transform")
 bones={}
 for b in rig.data.bones:
  bones[b.name]={"parent":None if b.parent is None else b.parent.name,"rest_matrix":[list(r) for r in b.matrix_local],
   "length_m":float(b.length),"inherit_scale":b.inherit_scale,"connected":bool(b.use_connect)}
 if len(bones)!=35:raise ValueError("Expected original7 plus28 finger bones")
 for b in rig.pose.bones:
  if np.max(abs(np.array(b.scale)-1))>1e-8 or np.max(abs(np.array(b.location)))>1e-8:raise ValueError("Source bone scale/translation")
 palm=read(PALM);braces={};parts={};original_A={}
 for side in ("left","right"):
  brace=bpy.data.objects[side+"_brace_frame"];T=np.array(brace.matrix_world);braces[side]=T.tolist()
  core=bpy.data.objects[side+"_Action4_core"];original_A[side]=(np.linalg.inv(T)@np.array(core.matrix_world)).tolist()
  parts[side]=[]
  for name in [side+"_brace_initial_geometry",side+"_mount_initial_support"]:
   obj=bpy.data.objects.get(name)
   if obj is None:raise ValueError("Expected existing source part missing")
   local=np.linalg.inv(T)@np.array(obj.matrix_world)
   parts[side].append({"name":name,"brace_from_object":local.tolist(),"object_local_bbox":[list(p) for p in obj.bound_box],
     "interface_semantics":"Actual object origin/bounding geometry only, NOT a verified mechanical mating port."})
 snap={"schema":"action4.full_rig_forward_snapshot.v1","status":"diagnostic_only","sources":sources,"bones":bones,"brace_rest_body":braces,
 "source_camera_A":original_A,"existing_parts":parts,"camera_corners_m":(np.array(read(MANIFEST)["corners_3d_mm"])*.001).tolist(),
 "camera_corner_order":read(MANIFEST)["corner_order"],"palm":{},"missing_assembly":["white_wrist_module","mount_ball","mount_link","quick_release","cable_connector","attached_cable"],
 "original_band_width_m":.028,"diagnostic_band_width_m":.034,"reference_center_shift_m":.003,"dataset_generation_allowed":False}
 for side in ("left","right"):
  M=np.array(bones[side+"_hand"]["rest_matrix"]);p=palm["sides"][side]
  snap["palm"][side]={"normal_hand_local":p["palm_normal_in_existing_G1_hand_bone"],
   "center_hand_local":apply(np.linalg.inv(M),[p["plane_center_m"]])[0].tolist(),"source_proxy_not_GT":True}
 cfg=read("configs/data/action4_global_rig_init_v001.json");checks=[]
 for name,pose in POSES.items():
  apply_pose(rig,pose,cfg);P=fk_bones(snap,pose);bone_err=max(float(abs(P[n]-np.array(rig.pose.bones[n].matrix)).max()) for n in bones)
  brace_err={};camera_err={};actual_braces={}
  for side in ("left","right"):
   fore=side+"_forearm";F=P[fore]@np.linalg.inv(np.array(bones[fore]["rest_matrix"]))@np.array(braces[side])
   actual=np.array(bpy.data.objects[side+"_brace_frame"].matrix_world);actual_braces[side]=actual.tolist()
   brace_err[side]=float(abs(F-actual).max())
   camera_err[side]=float(abs(F@np.array(original_A[side])-np.array(bpy.data.objects[side+"_Action4_core"].matrix_world)).max())
  if bone_err>3e-6 or max(brace_err.values())>3e-6 or max(camera_err.values())>3e-6:raise ValueError("Actual Blender FK mismatch")
  checks.append({"name":name,"pose":pose,"max_35bone_matrix_error":bone_err,"brace_errors":brace_err,"source_camera_errors":camera_err,
   "actual_Blender_bone_matrices":{n:[list(r) for r in rig.pose.bones[n].matrix] for n in bones},"actual_Blender_braces":actual_braces})
 apply_pose(rig,{},cfg);check_source_identities(sources);snap["actual_Blender_tests"]=checks
 snap["invariants_sha256"]=snapshot_digest(snap)
 write(output,snap);return {"snapshot":str(output),"sources_unchanged":True,"bone_count":35,"tests":[{k:v for k,v in x.items() if not k.startswith("actual_")} for x in checks]}
def candidates():
 if sha(LEFT)!="feb6c791d9263134680291a1af4132de73bcce7ead0a1c5013ed6648158fc93d":raise ValueError("Left identity")
 if sha(RIGHT)!="3038d35cf74782bfe7d0525c2a3330b61d154ad716050decd3f32e7f65c30bb1":raise ValueError("Right identity")
 if sha(SEMANTICS)!="80b8c8e3a291e2ada68def29b8c66b6932343f8cb957c3e5eff2931a17b923e8":raise ValueError("Semantic evidence identity")
 left,right=read(LEFT),read(RIGHT);result=[]
 for k in (.6,.8125,1.1):
  r=next(x for x in right["cases"] if x["K_ratio"]==k and x["width_mm"]==34)
  for branch in (0,1):
   l=next(x for x in left["starts"] if x["K_ratio"]==k and x["branch"]==branch)
   A={"left":proper(l["A_left_newreference_from_camera"]),"right":proper(r["A_reference_transformed_not_fitted"])}
   B={"left":proper(l["B_capture_from_left_newreference"]),"right":proper(r["B_reference_transformed"]["0"])}
   result.append({"id":"K"+str(k).replace(".","p")+"_L"+str(branch)+"_R0","K_ratio":k,"A":A,"target_B":B,
    "left_source_id":l["id"],"right_source_id":r["id"],"status":"diagnostic_candidates_not_production"})
 return result
def fk_forward(snapshot,candidate,pose,E=None):
 P=fk_bones(snapshot,pose);F={};f=candidate["K_ratio"]*3248
 for side in ("left","right"):
  fore=side+"_forearm";F[side]=P[fore]@np.linalg.inv(np.array(snapshot["bones"][fore]["rest_matrix"]))@np.array(snapshot["brace_rest_body"][side])@translation(.003)
 E=proper(candidate["target_B"]["left"]@np.linalg.inv(F["left"])) if E is None else proper(E)
 result={"status":"diagnostic_only","candidate_id":candidate["id"],"K_ratio":candidate["K_ratio"],"pose":pose,"common_E_capture_from_body":E.tolist(),
 "observer_rule":"E = target B_L inv(F_L); same E used for both sides; not an estimated real observer.","bones":{},"sides":{},"dataset_generation_allowed":False}
 for name,b in snapshot["bones"].items():
  points=apply(E@P[name],[[0,0,0],[0,b["length_m"],0]])
  result["bones"][name]={"matrix_body":P[name].tolist(),"head_tail_capture":points.tolist(),"projection":project(points,f),"parent":b["parent"],"length_m":b["length_m"]}
 for side in ("left","right"):
  A=proper(candidate["A"][side]);B=E@F[side];C=B@A;targetC=candidate["target_B"][side]@A
  core=apply(C,snapshot["camera_corners_m"]);target=apply(targetC,snapshot["camera_corners_m"])
  R=C[:3,:3]@targetC[:3,:3].T;angle=math.degrees(math.acos(float(np.clip((np.trace(R)-1)/2,-1,1))))
  projection=project(core,f);targetproj=project(target,f)
  valid=np.array(projection["positive_depth"])&np.array(targetproj["positive_depth"])
  rms=None if not valid.all() else float(np.sqrt(np.mean(np.sum((np.array(projection["xy"])-np.array(targetproj["xy"]))**2,axis=1))))
  hand=P[side+"_hand"];pc=apply(E@hand,[snapshot["palm"][side]["center_hand_local"]])[0]
  normal=(E@hand)[:3,:3]@np.array(snapshot["palm"][side]["normal_hand_local"]);normal/=np.linalg.norm(normal)
  cosine=float(normal@pc/max(np.linalg.norm(pc),1e-12))
  thumbnames=[n for n in P if n.startswith(side+"_thumb_")];thumb=sorted(thumbnames,key=lambda n:int(n.rsplit("_",1)[1]))[-1]
  tip=apply(E@P[thumb],[[0,snapshot["bones"][thumb]["length_m"],0]])[0]
  pp=project([pc,tip,pc+normal*.04],f)
  dx=None if pp["xy"][0] is None or pp["xy"][1] is None else pp["xy"][1][0]-pp["xy"][0][0]
  expected=-1 if side=="left" else 1
  parts=[]
  for p in snapshot["existing_parts"][side]:
   local=translation(-.003)@np.array(p["brace_from_object"]);T=B@local
   pts=apply(T,p["object_local_bbox"])
   parts.append({"name":p["name"],"matrix_capture":T.tolist(),"bbox_capture":pts.tolist(),"projection":project(pts,f),
    "interface_semantics":p["interface_semantics"],"assembly_reach_verified":False})
  result["sides"][side]={"F_body_brace":F[side].tolist(),"B_capture_brace":B.tolist(),"C_capture_camera":C.tolist(),"A_fixed":A.tolist(),
   "canonical8_capture":core.tolist(),"canonical8_projection":projection,"target_C":targetC.tolist(),
   "difference_to_body_candidate":{"translation_m":float(np.linalg.norm(C[:3,3]-targetC[:3,3])),"rotation_deg":angle,
    "canonical8_RMS_px":rms,"target_is_uncertain_candidate_not_GT":True},
   "palm":{"center_capture":pc.tolist(),"normal_capture":normal.tolist(),"thumb_tip_proxy_capture":tip.tolist(),"projection":pp,
    "normal_dot_view":cosine,"source_proxy_facing":"palm_toward_capture" if cosine<0 else "palm_away_from_capture",
    "f0_observed_surface":"palm","thumb_proxy_dx_px":dx,"f0_observed_thumb_side":"left_of_palm" if side=="left" else "right_of_palm",
    "thumb_side_diagnostic":None if dx is None else ("same_side" if dx*expected>0 else "opposite_side"),
    "scope":"Hand-pose-dependent source proxies, not measured normal or point correspondence; neutral mismatch does not prove candidate impossible."},
   "existing_parts":parts,"missing_assembly":snapshot["missing_assembly"],"assembly_status":"UNVERIFIED: source support/28mm band retained as existing geometry; analytic34mm reference only."}
 return result
def _clip(a,b,width=3248,height=2464):
 if a is None or b is None:return None
 a=np.asarray(a,float);b=np.asarray(b,float);delta=b-a;t0,t1=0.,1.
 for p,q in [(-delta[0],a[0]),(delta[0],width-1-a[0]),(-delta[1],a[1]),(delta[1],height-1-a[1])]:
  if abs(p)<1e-12:
   if q<0:return None
  elif p<0:t0=max(t0,q/p)
  else:t1=min(t1,q/p)
  if t0>t1:return None
 return a+t0*delta,a+t1*delta
def draw(snapshot,result,path):
 from PIL import Image,ImageDraw,ImageFont
 raw=Image.open(bounded(FRAME)).convert("RGB");header=210
 im=Image.new("RGB",(3248,2464+header),"white");im.paste(raw,(0,header));dr=ImageDraw.Draw(im);font=ImageFont.load_default(size=30)
 def line(a,b,color,width=4):
  seg=_clip(a,b)
  if seg:dr.line([(float(p[0]),float(p[1]+header)) for p in seg],fill=color,width=width)
 for name,b in result["bones"].items():
  color="#44dd44" if name.startswith("left") else "#ff9900" if name.startswith("right") else "#aa66ff"
  line(*b["projection"]["xy"],color,4 if "thumb" not in name else 6)
 corners=np.array(snapshot["camera_corners_m"])
 for side,data in result["sides"].items():
  color="#00ffff" if side=="left" else "#ff44cc";uv=data["canonical8_projection"]["xy"]
  target=project(apply(data["target_C"],snapshot["camera_corners_m"]),result["K_ratio"]*3248)["xy"]
  for i,a in enumerate(corners):
   for j,b in enumerate(corners):
    if j>i and np.sum(abs(a-b)>1e-8)==1:
     line(uv[i],uv[j],color,5);line(target[i],target[j],"#ffff00",2)
  pp=data["palm"]["projection"]["xy"];line(pp[0],pp[2],"#ffffff",7);line(pp[0],pp[1],color,4)
  for part in data["existing_parts"]:
   pts=part["projection"]["xy"];bbox=np.array(part["bbox_capture"])
   # Source object bbox corner ordering follows Blender; draw its canonical box edges.
   for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:line(pts[a],pts[b],"#999999",2)
 right=result["sides"]["right"]["difference_to_body_candidate"]
 lines=[result["candidate_id"]+" / "+("neutral" if not result["pose"] else "small pose")+" / DIAGNOSTIC INITIALIZATION, NOT video-fit",
 "One common E anchored to left brace; complete35bone data in JSON; behind/offscreen projections not invented",
 "cyan/magenta predicted virtual8; yellow previousbody candidates; green/orange skeleton; white palm-normal proxy",
 f"Right candidate discrepancy: {right['translation_m']:.3f}m, {right['rotation_deg']:.1f}deg; no assembly/contact/skin acceptance",
 "f0 observed both palms and outer thumbs; compare HAND pose, never use brace normal. Source accessories remain incomplete."]
 for i,t in enumerate(lines):dr.text((12,10+i*39),t,font=font,fill="black")
 im.save(bounded(path))
def forward_suite(snapshot_path,output):
 snapshot=read(snapshot_path);validate_snapshot(snapshot)
 out=bounded(output)
 if out.exists():raise FileExistsError("Use fresh forward directory")
 out.mkdir(parents=True);cs=candidates();records=[];tests=[]
 for candidate in cs:
  values={name:fk_forward(snapshot,candidate,pose) for name,pose in POSES.items()}
  for name,value in values.items():
   write(out/(candidate["id"]+"_"+name+".json"),value)
   if name=="neutral" or candidate["id"]=="K1p1_L0_R0":draw(snapshot,value,out/(candidate["id"]+"_"+name+".png"))
   if name=="neutral":draw_overview(snapshot,value,out/(candidate["id"]+"_overview.png"))
  neutral,hand=values["neutral"],values["hand_only"]
  hand_err=max(float(abs(np.array(neutral["sides"][s][k])-hand["sides"][s][k]).max()) for s in ("left","right") for k in ("B_capture_brace","C_capture_camera"))
  if hand_err>1e-12:raise ValueError("Hand-only changed camera/brace")
  comparison={}
  for actual in snapshot["actual_Blender_tests"]:
   name=actual["name"];F={}
   for side in ("left","right"):F[side]=np.array(actual["actual_Blender_braces"][side])@translation(.003)
   E=candidate["target_B"]["left"]@np.linalg.inv(F["left"])
   comparison[name]=max(float(abs(E@F[s]@candidate["A"][s]-np.array(values[name]["sides"][s]["C_capture_camera"])).max()) for s in ("left","right"))
  if max(comparison.values())>6e-6:raise ValueError("Candidate camera vs actual Blender brace mismatch")
  tests.append({"candidate":candidate["id"],"hand_only_camera_brace_max_error":hand_err,"candidate_C_vs_actual_Blender_brace_composition":comparison})
  records.append({"id":candidate["id"],"neutral_differences":{s:neutral["sides"][s]["difference_to_body_candidate"] for s in ("left","right")},
   "neutral_palm":{s:neutral["sides"][s]["palm"] for s in ("left","right")},
   "neutral_visible_corners":{s:sum(neutral["sides"][s]["canonical8_projection"]["inside_native"]) for s in ("left","right")}})
 check_source_identities(snapshot["sources"])
 report={"status":"diagnostic_only_not_fitted","snapshot_path":str(bounded(snapshot_path).relative_to(ROOT)),"snapshot_sha256":sha(snapshot_path),
 "candidates":records,"tests":tests,"source_hashes":snapshot["sources"],"source_assets_unchanged":True,"optimizer_calls":0,
 "pose_domain_degrees":25,"production_A_or_K_selected":False,"dataset_generation_allowed":False,
 "missing_assembly":snapshot["missing_assembly"],"assembly_reach":"NOT_VERIFIED: no fabricated mating ports/reach."}
 write(out/"report.json",report);return report
def main():
 argv=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else None
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=["export","forward"]);p.add_argument("--output",required=True);p.add_argument("--snapshot")
 a=p.parse_args(argv)
 if a.mode=="forward" and not a.snapshot:p.error("--snapshot required")
 result=export_snapshot(a.output) if a.mode=="export" else forward_suite(a.snapshot,a.output)
 print(json.dumps({"status":result.get("status","snapshot_exported"),"output":a.output}))
if __name__=="__main__":main()
