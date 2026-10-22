"""G2b actual component4 body-only projection diagnostics. Never calibration or GT."""
from __future__ import annotations
import argparse,json,hashlib,math,time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion,binary_dilation
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from PIL import Image,ImageDraw,ImageFont
ROOT=Path("/data/zy/vln/a-test")
PACKET="outputs/action4_realism_reassessment_v001/rig_relation/visible_body_evidence_v002/visible_body_observations_final.json"
MESH="data/DJI_Action4_construct/r3_v001/mesh_v001.npz"
MANIFEST="data/DJI_Action4_construct/r3_v001/asset_manifest.json"
RATIOS=(.6,.8125,1.1)
ORDER=("upper","right","lower","left")


def path(p):
 p=(ROOT/p).resolve()
 if not p.is_relative_to(ROOT):raise ValueError("Outside project")
 return p


def sha(p):return hashlib.sha256(path(p).read_bytes()).hexdigest()


def write(p,v):
 p=path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("x") as f:json.dump(v,f,indent=2,allow_nan=False)


FORWARD_SOURCES=(PACKET,MESH,MANIFEST,"src/data/action4_body_projection_fit.py","scripts/action4_body_projection_fit.py")
REVIEW_IMAGES=tuple(f"{case}_{variant}_CAD.png" for case in ("f0_L","f0_R") for variant in ("outer_support","rear_facing"))
FORWARD_ARTIFACTS=tuple(f"{case}_{variant}_{kind}.png" for case in ("f0_L","f0_R") for variant in ("outer_support","rear_facing") for kind in ("RAW","CAD","VIRTUAL8"))+("feature_definitions.json","forward_report.json")


def forward_sources():
 return {name:sha(name) for name in FORWARD_SOURCES}


def forward_artifact(directory,name):
 if name not in FORWARD_ARTIFACTS:raise ValueError("Unknown forward artifact")
 root=path(directory);target=path(root/name)
 if not root.is_relative_to(ROOT/"outputs") or not target.is_relative_to(root):
  raise ValueError("Forward artifacts must remain inside their outputs directory")
 return target


def write_forward_manifest(directory,source_identity):
 if source_identity!=forward_sources():raise ValueError("Source/input changed during forward generation")
 root=path(directory)
 manifest={"schema_version":"action4.body_forward_manifest.v1",
           "status":"generated_pending_actual_review","sources":source_identity,
           "artifacts":{name:sha(forward_artifact(root,name)) for name in FORWARD_ARTIFACTS},
           "required_review_images":list(REVIEW_IMAGES)}
 write(root/"forward_manifest.json",manifest)
 return manifest


def validate_forward_review(directory):
 root=path(directory)
 if not root.is_relative_to(ROOT/"outputs"):raise ValueError("Forward review must be under outputs")
 manifest_path=path(root/"forward_manifest.json");review_path=path(root/"review.json")
 if not manifest_path.is_relative_to(root) or not review_path.is_relative_to(root):raise ValueError("Forward manifest/review escapes its directory")
 manifest=json.loads(manifest_path.read_text());review=json.loads(review_path.read_text())
 if manifest.get("schema_version")!="action4.body_forward_manifest.v1" or manifest.get("status")!="generated_pending_actual_review":
  raise ValueError("Versioned generated forward manifest required")
 if manifest.get("sources")!=forward_sources():raise ValueError("Stale forward source/input identity")
 if set(manifest.get("artifacts",{}))!=set(FORWARD_ARTIFACTS) or manifest.get("required_review_images")!=list(REVIEW_IMAGES):
  raise ValueError("Forward artifact coverage differs")
 for name,digest in manifest["artifacts"].items():
  if sha(forward_artifact(root,name))!=digest:raise ValueError("Stale forward artifact: "+name)
 if (review.get("schema_version")!="action4.body_forward_review.v1" or review.get("actually_viewed") is not True
     or review.get("decision")!="proceed_bounded_body_only_diagnostic_not_alignment_acceptance"
     or not isinstance(review.get("reviewer"),str) or not review["reviewer"].strip()
     or not isinstance(review.get("observations"),str) or not review["observations"].strip()):
  raise ValueError("Explicit actual forward review required")
 reference={"path":str(manifest_path.relative_to(ROOT)),"sha256":sha(manifest_path)}
 if review.get("forward_manifest")!=reference:raise ValueError("Review does not bind this exact forward manifest")
 images={name:manifest["artifacts"][name] for name in REVIEW_IMAGES}
 if review.get("reviewed_artifacts")!=images:raise ValueError("Review does not bind all four exact CAD images")
 return {"forward_manifest":reference,"review":{"path":str(review_path.relative_to(ROOT)),"sha256":sha(review_path)},
         "reviewed_artifacts":images,"scope":"forward initialization reviewed only; not fitted alignment or production approval"}


def pose(q):
 T=np.eye(4);T[:3,:3]=Rotation.from_rotvec(q[:3]).as_matrix();T[:3,3]=q[3:];return T


def project(v,T,f):
 p=np.asarray(v)@T[:3,:3].T+T[:3,3]
 return p[:,:2]/np.maximum(p[:,2,None],.005)*f+np.array([1623.5,1231.5])


def nearest(p,segments):
 a,b=segments[:,0],segments[:,1];d=b-a
 t=np.clip(((p[:,None,:]-a)*d).sum(-1)/np.maximum((d*d).sum(-1),1e-12),0,1)
 delta=p[:,None,:]-(a+t[...,None]*d);idx=(delta*delta).sum(-1).argmin(1)
 return delta[np.arange(len(p)),idx],idx


def features(v,faces,components,variant):
 face=faces[components==4]
 if variant=="rear_facing":
  n=np.cross(v[face[:,1]]-v[face[:,0]],v[face[:,2]]-v[face[:,0]])
  n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-12)
  face=face[n[:,1]>.98]
 ids=np.unique(face);points=v[ids][:,[0,2]]
 all_edges={tuple(sorted((int(a),int(b)))) for tri in faces[components==4] for a,b in zip(tri,np.roll(tri,-1))}
 if variant=="rear_facing":
  counts=Counter(tuple(sorted((int(a),int(b)))) for tri in face for a,b in zip(tri,np.roll(tri,-1)))
  adjacency={}
  for (a,b),count in counts.items():
   if count==1:adjacency.setdefault(a,[]).append(b);adjacency.setdefault(b,[]).append(a)
  if any(len(n)!=2 for n in adjacency.values()):raise ValueError("Rear surface boundary is not closed degree2 topology")
  visited=set();loops=[]
  for first in adjacency:
   if first in visited:continue
   current=first;previous=None;cycle=[]
   while current not in visited:
    visited.add(current);cycle.append(current);nxt=[n for n in adjacency[current] if n!=previous]
    previous,current=current,nxt[0]
   if current!=first:raise ValueError("Broken rear boundary loop")
   loops.append(cycle)
  def area(cycle):
   p=v[cycle][:,[0,2]];return abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))/2
  loop=np.array(max(loops,key=area))
 else:loop=ids[ConvexHull(points).vertices]
 if any(tuple(sorted((int(a),int(b)))) not in all_edges for a,b in zip(loop,np.roll(loop,-1))):
  raise ValueError("Support contour contains non-topological chords; do not call them actual CAD edges")
 center=(points.min(0)+points.max(0))/2;half=(points.max(0)-points.min(0))/2
 assigned={k:[] for k in ORDER}
 for a,b in zip(loop,np.roll(loop,-1)):
  mid=(v[a,[0,2]]+v[b,[0,2]])/2;norm=(mid-center)/half
  key=("left" if norm[0]>0 else "right") if abs(norm[0])>abs(norm[1]) else ("upper" if norm[1]>0 else "lower")
  assigned[key].append([int(a),int(b)])
 if any(not x for x in assigned.values()):raise ValueError("Missing ordered component4 region")
 return {"variant":variant,"component":4,"actual_mesh_vertex_edge_pairs":assigned,
         "warning":"Rear-view support contour of actual component4 vertices; bevel boundary hypothesis, not fixed observed3D endpoints or LCD corners",
         "rear_normal_threshold":.98 if variant=="rear_facing" else None,"topology_edges_verified":True,
         "curve_reference_center_body_m":v[loop].mean(0).tolist(),"boundary_vertex_count":len(loop),
         "derivation":"largest-area actual boundaryloop of rear-facing submesh" if variant=="rear_facing" else "rear-direction support loop verified against actual mesh edge set"}


def prepare():
 packet=json.loads(path(PACKET).read_text())
 if sha(PACKET)!="fe94ac19a77b3a3b13410cf01afcad6e1de8ebe0bd0754d4bb7b49d8746e1076":raise ValueError("Packet identity changed")
 with np.load(path(MESH),allow_pickle=False) as a:v=a["vertices_mm"]*.001;faces=a["faces"];components=a["face_components"]
 cases=[c for c in packet["cases"] if c["frame_id"]==0]
 if {c["track"] for c in cases}!={"L","R"}:raise ValueError("Only f0 bilateral evidence expected")
 for c in cases:
  if sha(c["source"])!=c["source_sha256"] or c["polygon_completeness"]!="near_complete_visible_outline":raise ValueError("Source/evidence changed")
  g=c["rear_outer_face_ordered_edge_graph"]
  if g["image_order"]!=list(ORDER) or len(g["segments"])!=4:raise ValueError("Ordered graph required")
  if any(x["fixed_model_vertices"] is not None or x["segment_endpoints_are_corners"] for x in g["segments"]):raise ValueError("Do not invent endpoint correspondences")
 variants={name:features(v,faces,components,name) for name in ("outer_support","rear_facing")}
 extreme_ids=np.sort(ConvexHull(v).vertices)
 return {"vertices":v,"faces":faces,"components":components,"cases":cases,"features":variants,"support_vertex_ids":extreme_ids,
         "canonical_corners":np.array(json.loads(path(MANIFEST).read_text())["corners_3d_mm"])*.001}


def polygon_signed_distance(points,polygon):
 segments=np.stack([polygon,np.roll(polygon,-1,axis=0)],axis=1)
 delta,_=nearest(points,segments);distance=np.linalg.norm(delta,axis=1)
 a=segments[:,0];b=segments[:,1];px=points[:,0,None];py=points[:,1,None]
 denom=b[:,1]-a[:,1];safe=np.where(np.abs(denom)<1e-12,1e-12,denom)
 crossing=((a[:,1]>py)!=(b[:,1]>py))&(px<((b[:,0]-a[:,0])*(py-a[:,1])/safe+a[:,0]))
 inside=(crossing.sum(1)%2)==1
 return np.where(inside,-distance,distance)


def uncertainty(case):
 polygon=np.array(case["visible_body_region_polygon_px"],np.float32);center=polygon.mean(0);sectors=[]
 for row in case["uncertain_boundary_segments"]:
  p=np.array(row["points_px"])-center;angles=np.unwrap(np.arctan2(p[:,1],p[:,0]))
  margin=math.atan2(2*case["manual_uncertainty_radius_px"],max(np.linalg.norm(p,axis=1).min(),1))
  sectors.append([float(angles.min()-margin),float(angles.max()+margin)])
 return polygon,center,sectors


def unknown_angle(points,center,sectors):
 theta=np.arctan2(points[:,1]-center[1],points[:,0]-center[0]);mask=np.zeros(len(points),bool)
 for lo,hi in sectors:
  for shift in (-2*np.pi,0,2*np.pi):mask|=(theta+shift>=lo)&(theta+shift<=hi)
 return mask


def initial(case,f,feature):
 variant=feature["variant"]
 graph=case["rear_outer_face_ordered_edge_graph"]["segments"]
 upper=np.array(graph[0]["points_px"]);lower=np.array(graph[2]["points_px"])
 direction=upper[1]-upper[0];roll=math.atan2(direction[1],direction[0])
 base=np.array([[-1,0,0],[0,0,-1],[0,-1,0]],float)
 tilt=15 if variant=="outer_support" else 30
 R=Rotation.from_euler("z",roll).as_matrix()@Rotation.from_euler("x",tilt,degrees=True).as_matrix()@base
 center=np.concatenate([np.array(g["points_px"]) for g in graph]).mean(0)
 width=np.mean([np.linalg.norm(upper[1]-upper[0]),np.linalg.norm(lower[1]-lower[0])])
 # Segments are partial, so measured span is initialization only.
 depth=f*.066/max(width*1.15,80)
 t=np.r_[(center-np.array([1623.5,1231.5]))*depth/f,depth]-R@np.array(feature["curve_reference_center_body_m"])
 return np.r_[Rotation.from_matrix(R).as_rotvec(),t]


def residual(q,model,case,f,feature):
 T=pose(q);values=[];meta=[];sigma=case["manual_uncertainty_radius_px"]
 for key,ob in zip(ORDER,case["rear_outer_face_ordered_edge_graph"]["segments"]):
  pairs=np.array(feature["actual_mesh_vertex_edge_pairs"][key]);segments=project(model["vertices"][pairs].reshape(-1,3),T,f).reshape(-1,2,2)
  points=np.array(ob["points_px"],float);delta,idx=nearest(points,segments);values.extend((delta/sigma).ravel())
  # Direction uses the closest local tangent; endpoint identities remain latent.
  middle=points.mean(0)[None,:];_,mi=nearest(middle,segments);pred=segments[mi[0],1]-segments[mi[0],0]
  pred/=max(np.linalg.norm(pred),1e-9);obs=points[1]-points[0];length=np.linalg.norm(obs);obs/=length
  if pred@obs<0:pred=-pred
  directional=(pred-obs)*length/(2*sigma);values.extend(directional)
  meta.append({"observation_id":ob["id"],"assigned_component4_region":key,"matched_actual_mesh_vertex_pairs":[pairs[int(i)].tolist() for i in idx],
               "endpoint_distance_px":np.linalg.norm(delta,axis=1).tolist(),"direction_equivalent_error_px":float(np.linalg.norm(directional)*sigma)})
 polygon,center,sectors=uncertainty(case)
 projected=project(model["vertices"][model["support_vertex_ids"]],T,f)
 distances=polygon_signed_distance(projected,polygon)
 outside=np.maximum(0,distances-sigma);outside[unknown_angle(projected,center,sectors)]=0
 # Sparse actual CAD surface samples; final gate uses fulltriangle union, not this support approximation.
 values.extend(outside/sigma/math.sqrt(max(len(outside)/24,1)))
 minz=(model["vertices"]@T[:3,:3].T+T[:3,3])[:,2].min()
 face_center=T[:3,3]+T[:3,1]*.0162;face_cos=float(T[:3,1]@face_center/max(np.linalg.norm(face_center),1e-9))
 values.extend([30*max(0,.04-minz)/.01,30*max(0,face_cos)])
 return np.array(values),meta,{"min_depth_m":float(minz),"rear_face_cos":face_cos}


def raster(model,T,f,case):
 # Native-coordinate crop, quarter-size raster. Every original CAD triangle is union-filled.
 polygon=np.array(case["visible_body_region_polygon_px"],float)
 projected=project(model["vertices"],T,f)
 lo=np.floor(np.minimum(polygon.min(0),projected.min(0))-50);hi=np.ceil(np.maximum(polygon.max(0),projected.max(0))+50)
 if np.any(hi-lo>4000):return None
 factor=4.;shape=np.ceil((hi-lo)/factor).astype(int)
 mask_image=Image.new("1",(int(shape[0]),int(shape[1])),0);draw_mask=ImageDraw.Draw(mask_image)
 uv=np.round((projected-lo)/factor).astype(np.int32)
 for face in model["faces"]:draw_mask.polygon([tuple(p) for p in uv[face]],fill=1)
 mask=np.array(mask_image,dtype=np.uint8)
 manual_image=Image.new("1",mask_image.size,0);poly=np.round((polygon-lo)/factor).astype(np.int32)
 ImageDraw.Draw(manual_image).polygon([tuple(p) for p in poly],fill=1);manual=np.array(manual_image,dtype=np.uint8)
 radius=max(1,int(math.ceil(case["manual_uncertainty_radius_px"]/factor)))
 yyk,xxk=np.mgrid[-radius:radius+1,-radius:radius+1];kernel=xxk*xxk+yyk*yyk<=radius*radius
 positive=binary_erosion(manual,structure=kernel);negative=~binary_dilation(manual,structure=kernel)
 yy,xx=np.mgrid[:mask.shape[0],:mask.shape[1]];pixels=np.column_stack([xx.ravel(),yy.ravel()])*factor+lo
 _,center,sectors=uncertainty(case);unknown=unknown_angle(pixels,center,sectors).reshape(mask.shape)&(manual==0)
 negative&=~unknown
 fp=(mask>0)&negative;fn=(mask==0)&positive
 return {"mask":mask,"manual":manual,"positive":positive,"negative":negative,"unknown":unknown,"origin":lo,"factor":factor,
         "metrics":{"certain_visible_false_negative_fraction":float(fn.sum()/max(positive.sum(),1)),
                    "known_background_excess_relative_visible_area":float(fp.sum()/max(manual.sum(),1)),
                    "projected_CAD_area_native_px2":int(mask.sum()*factor**2),"manual_region_area_native_px2":int(manual.sum()*factor**2),
                    "raster_scale_native_pixels_per_sample":factor,"unknown_outward_sectors_radians":sectors,
                    "scope":"Whole actual triangle-union projection; uncertain boundary outward sectors excluded from background penalty. Not amodalGT."}}


def line_pairs(points):
 return np.stack([points,np.roll(points,-1,axis=0)],axis=1)


def draw(model,case,f,q,feature,directory,label):
 directory=path(directory);directory.mkdir(parents=True,exist_ok=True)
 T=pose(q);raw=Image.open(path(case["source"])).convert("RGB");box=case["crop_xyxy_exclusive"];header=125
 font=ImageFont.load_default(size=16)
 def canvas():
  crop=raw.crop(box);im=Image.new("RGB",(max(crop.width,850),crop.height+header),"white");im.paste(crop,(0,header));return im,ImageDraw.Draw(im)
 def xy(p):return (float(p[0]-box[0]),float(p[1]-box[1]+header))
 im,d=canvas();d.text((8,8),f"{label}: RAW / {case['id']} / native crop / NO GT",font=font,fill="black");im.save(directory/(label+"_RAW.png"))
 im,d=canvas();colors={"upper":"#00ffff","right":"#ff55dd","lower":"#44dd44","left":"#ff9900"}
 for key,ob in zip(ORDER,case["rear_outer_face_ordered_edge_graph"]["segments"]):
  pairs=np.array(feature["actual_mesh_vertex_edge_pairs"][key]);segs=project(model["vertices"][pairs].reshape(-1,3),T,f).reshape(-1,2,2)
  for seg in segs:d.line([xy(p) for p in seg],fill=colors[key],width=4)
  d.line([xy(p) for p in ob["points_px"]],fill="#ffff00",width=3)
  p=np.mean(ob["points_px"],axis=0);d.text(xy(p),key,font=font,fill=colors[key],stroke_width=1,stroke_fill="black")
 r=raster(model,T,f,case)
 if r:
  boundary=(r["mask"]>0)&~binary_erosion(r["mask"]>0)
  yy,xx=np.nonzero(boundary)
  for p in np.column_stack([xx,yy])*r["factor"]+r["origin"]:
   x,y=xy(p);d.rectangle((x,y,x+2,y+2),fill="#22aaff")
 title=[f"{label}: ACTUAL CAD/component4 feature hypothesis, NOT calibrated / {case['id']}",
        f"variant={feature['variant']} / f/W={f/3248:.4f}; edge pairs are not observed3D endpoints",
        "yellow=manual edges; colors=injective component4 regions; blue=full CAD triangle silhouette",
        "screen/sticker excluded; uncertain external regions not treated as known background"]
 for i,t in enumerate(title):d.text((8,8+25*i),t,font=font,fill="black")
 im.save(directory/(label+"_CAD.png"))
 im,d=canvas();corners=project(model["canonical_corners"],T,f)
 for i,a in enumerate(model["canonical_corners"]):
  for j,b in enumerate(model["canonical_corners"]):
   if j>i and np.sum(np.abs(a-b)>1e-8)==1:d.line([xy(corners[i]),xy(corners[j])],fill="#dd2222",width=3)
 for i,p in enumerate(corners):d.text(xy(p),str(i),font=font,fill="#dd2222")
 d.text((8,8),f"{label}: TASK8 VIRTUAL CUBOID ONLY / NOT physical corners or trainingGT",font=font,fill="black")
 im.save(directory/(label+"_VIRTUAL8.png"))
 return None if r is None else r["metrics"]


def forward(output):
 out=path(output)
 if out.exists():raise FileExistsError("Fresh output required")
 out.mkdir(parents=True);sources=forward_sources();m=prepare()
 write(out/"feature_definitions.json",m["features"])
 cases=[]
 for case in m["cases"]:
  for name,feature in m["features"].items():
   q=initial(case,3248*.8125,feature)
   metrics=draw(m,case,3248*.8125,q,feature,out,case["id"]+"_"+name)
   cases.append({"case":case["id"],"variant":name,"q_initial":q.tolist(),"whole_CAD_metrics":metrics})
 # Known-pose forward crosscheck vs independent homogeneous K[R|t], no optimizer.
 q=np.r_[.2,-.4,.1,.05,-.02,.5];T=pose(q);uv=project(m["vertices"][::137],T,2639.)
 K=np.array([[2639,0,1623.5],[0,2639,1231.5],[0,0,1]],float)
 homogeneous=np.column_stack([m["vertices"][::137],np.ones(len(m["vertices"][::137]))])@(K@T[:3]).T
 independent=homogeneous[:,:2]/homogeneous[:,2,None]
 error=float(np.max(abs(uv-independent)));assert error<1e-7
 write(out/"forward_report.json",{"status":"FORWARD_ONLY_PENDING_ACTUAL_IMAGE_REVIEW","known_pose_homogeneous_max_abs_error_px":error,
       "cases":cases,"dataset_generation_allowed":False,"assumptions":"Image ordered graph maps upper+Z/right-X/lower-Z/left+X in rear view, injective. Two actualcomponent4 contour hypotheses, noLCD/cuboid shortcut."})
 write_forward_manifest(out,sources)


def run(output,forward_dir):
 out=path(output)
 if out.exists():raise FileExistsError("Fresh fit output required")
 review_binding=validate_forward_review(forward_dir)
 out.mkdir(parents=True);m=prepare()
 sources={p:sha(p) for p in [PACKET,MESH,MANIFEST,"src/data/action4_body_projection_fit.py"]}
 write(out/"config.json",{"scope":"G2b body-only diagnostic; no A/brace/human/world optimization","frame_ids":[0],"K_ratios":RATIOS,
 "max_starts":12,"max_nfev_each":150,"principal_point":[1623.5,1231.5],"square_pixels":True,"zero_D_assumed":True,
 "source_hashes":sources,"feature_hypotheses":list(m["features"]),"output_image_separation":["RAW","CAD","VIRTUAL8"],
 "support_diagnostic_gate":"Both masked false-negative and background-excess fractions<=0.05. Engineering diagnostic, not production tolerance.",
 "pose_parameter_count_each":6,"dataset_generation_allowed":False,"forward_review_binding":review_binding})
 results=[]
 for ratio in RATIOS:
  f=3248*ratio
  for case in m["cases"]:
   for variant,feature in m["features"].items():
    label=f"{case['id']}_K{str(ratio).replace('.','p')}_{variant}"
    q=initial(case,f,feature);q0=q.copy();started=time.perf_counter();calls=[0]
    def cost(x):calls[0]+=1;return residual(x,m,case,f,feature)[0]
    lo=np.r_[[-np.pi]*3,[-1,-1,.08]];hi=np.r_[[np.pi]*3,[1,1,2.]]
    q=np.clip(q,lo+1e-8,hi-1e-8)
    try:
     fit=least_squares(cost,q,bounds=(lo,hi),max_nfev=150,loss="soft_l1",f_scale=1.,x_scale="jac")
     initial_r,initial_meta,_=residual(q0,m,case,f,feature);final_r,meta,valid=residual(fit.x,m,case,f,feature)
     support=draw(m,case,f,fit.x,feature,out/"overlays",label)
     okay=support is not None and valid["min_depth_m"]>.04 and valid["rear_face_cos"]<0 and all(support[k]<=.05 for k in ["certain_visible_false_negative_fraction","known_background_excess_relative_visible_area"])
     result={"case":case["id"],"fixed_f_px":f,"fixed_f_over_width":ratio,"variant":variant,"status":"diagnostic_candidate_pending_visual_review" if okay else "rejected_support_or_pose",
       "optimizer_success":bool(fit.success),"termination":fit.message,"nfev":fit.nfev,"actual_residual_calls":calls[0],"seconds":time.perf_counter()-started,
       "initial_q":q0.tolist(),"final_q":fit.x.tolist(),"T_capture_body":pose(fit.x).tolist(),"initial_edges":initial_meta,"final_edges":meta,
       "initial_normalized_residual_rms":float(np.sqrt(np.mean(initial_r**2))),"final_normalized_residual_rms":float(np.sqrt(np.mean(final_r**2))),
       "pose_validity":valid,"whole_actual_CAD_mask_metrics":support,"no_global_K_or_A_estimated":True}
    except Exception as exc:
     import traceback
     result={"case":case["id"],"fixed_f_over_width":ratio,"variant":variant,"status":"failure_retained","error":str(exc),"traceback":traceback.format_exc()}
    write(out/(label+".json"),result);results.append(result);print("BODY_START_COMPLETE",label,result["status"],flush=True)
 for p,h in sources.items():
  if sha(p)!=h:raise ValueError("Frozen bodyfit inputs changed")
 write(out/"summary.json",{"status":"BODY_ONLY_DIAGNOSTIC_NOT_CALIBRATION","dataset_generation_allowed":False,"starts":results,"K_calibrated":False,"A_optimized":False})
 return {"starts":len(results),"status":"BODY_ONLY_DIAGNOSTIC_NOT_CALIBRATION"}


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=["forward","fit"]);p.add_argument("--output",required=True);p.add_argument("--forward-dir")
 a=p.parse_args()
 if a.mode=="forward":forward(a.output)
 else:print(json.dumps(run(a.output,a.forward_dir)))
if __name__=="__main__":main()
