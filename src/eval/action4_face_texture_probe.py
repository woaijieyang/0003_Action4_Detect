"""Bounded OpenCV texture correspondence probe, NOT an M3T integration or pose."""
import json,hashlib
from pathlib import Path
import cv2
import numpy as np
from data.action4_paths import PROJECT_ROOT as ROOT
from eval.action4_roi_sweep import read_binding,binding
from eval.action4_reanchor_roi_provider import ReanchorROIProvider
from eval.action4_sam_full_review import ReceiptVerifier,checked_path

PARAMS=dict(reference_frame=808,frame_ids=list(range(808,823)),tracks=["track_00","track_01"],
 orb_features=1000,ratio=.75,max_hamming=64,ransac_px=3.,erosion_px=4,seed=20261027)

def match_features(a,b):
 if a is None or b is None or len(a)<2 or len(b)<2:return []
 matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
 def filtered(x,y):
  return {m.queryIdx:m for pair in matcher.knnMatch(x,y,k=2) if len(pair)==2 for m,n in [pair]
          if m.distance<PARAMS["ratio"]*n.distance and m.distance<=PARAMS["max_hamming"]}
 forward=filtered(a,b);reverse=filtered(b,a)
 return [m for i,m in forward.items() if m.trainIdx in reverse and reverse[m.trainIdx].trainIdx==i]

def features(rgb,mask,box):
 if mask is None:return None
 support=mask.copy();yy,xx=np.indices(mask.shape);l,t,r,b=box
 support &= (xx>=l)&(xx<r)&(yy>=t)&(yy<b)
 support=cv2.erode(support.astype(np.uint8),np.ones((9,9),np.uint8))
 y,x=np.nonzero(support)
 if not len(x):return None
 l=max(0,int(x.min())-32);t=max(0,int(y.min())-32);r=min(mask.shape[1],int(x.max())+33);b=min(mask.shape[0],int(y.max())+33)
 gray=cv2.cvtColor(rgb[t:b,l:r],cv2.COLOR_BGR2GRAY)
 kp,d=cv2.ORB_create(nfeatures=PARAMS["orb_features"]).detectAndCompute(gray,support[t:b,l:r]*255)
 xy=np.array([k.pt for k in kp],np.float32).reshape(-1,2)+[l,t]
 if len(xy):
  ij=np.rint(xy).astype(int);keep=support[ij[:,1],ij[:,0]].astype(bool);xy=xy[keep];d=d[keep]
 return dict(xy=xy,descriptors=d,support_pixels=int(support.sum()),crop=[l,t,r,b])

def run():
 cv2.setNumThreads(1);cv2.setRNGSeed(PARAMS["seed"])
 out=checked_path("outputs/action4_face_texture_probe_v001")
 # Fixed domain/thresholds are written before consuming observations.
 (out/"PREREGISTERED.json").write_text(json.dumps(dict(PARAMS,GT_used=False,pose_update=False,M3T_integrated=False),indent=2))
 lock=read_binding(binding(ROOT/"outputs/action4_fixed_best_video_v001/INPUT_LOCK.json"))
 provider=ReanchorROIProvider(**lock["video"]["track_replay"])
 jobs=json.loads((ROOT/"outputs/action4_front_region_partial_v002/JOBS.json").read_text())
 lookup={(j["frame_id"],j["track_id"]):j for j in jobs}
 reference={};rows=[];verifier=ReceiptVerifier()
 for fid in PARAMS["frame_ids"]:
  masks,receipt=provider.current_native_masks(fid)
  image_path=verifier.verify(f"data/action4_real/cache_v001/frames/frame_{fid:08d}.png",receipt["RGB_sha256"])
  rgb=cv2.imread(str(image_path))
  for track in PARAMS["tracks"]:
   job=lookup[fid,track];mask=masks.get(track)
   row=dict(frame_id=fid,track_id=track,pts=receipt["pts"],RGB_sha256=receipt["RGB_sha256"],
       mask_sha256=receipt.get("mask_sha256",{}).get(track),status="unavailable",missing_native_part=job["status"]!="prepared",
       pose_updated=False,XYZ_binding_available=False,formal_decision_allowed=False)
   f=features(rgb,mask,job["A0_raw"]["bbox_b0_raw"])
   if f is None:row["reason"]="no_current_instance_support";rows.append(row);continue
   row.update(keypoints=len(f["xy"]),support_pixels=f["support_pixels"])
   if fid==808:reference[track]=f;row.update(status="reference_only_not_temporal_validation",matches=0);rows.append(row);continue
   if track not in reference:row["reason"]="no_reference";rows.append(row);continue
   ref=reference[track];matches=match_features(ref["descriptors"],f["descriptors"])
   a=np.array([ref["xy"][m.queryIdx] for m in matches],np.float32).reshape(-1,2);b=np.array([f["xy"][m.trainIdx] for m in matches],np.float32).reshape(-1,2)
   H,inlier=(None,None) if len(matches)<4 else cv2.findHomography(a,b,cv2.RANSAC,PARAMS["ransac_px"],maxIters=2000,confidence=.995)
   keep=np.zeros(len(matches),bool) if inlier is None else inlier.ravel().astype(bool)
   row.update(status="image_correspondence_candidate" if keep.sum()>=4 else "insufficient_geometric_matches",matches=len(matches),inliers=int(keep.sum()),
       reference_uv=a.tolist(),current_uv=b.tolist(),inlier_mask=keep.tolist(),homography=None if H is None else H.tolist(),
       homography_is_not_rigid_pose=True)
   if keep.sum()>=4:
    cov=np.cov(a[keep].T);row["reference_spread_eigenvalues_px2"]=np.linalg.eigvalsh(cov).tolist()
    row["reference_inlier_hull_area_px2"]=float(cv2.contourArea(cv2.convexHull(a[keep])))
    if lookup[808,track]["status"]=="prepared":
     j=lookup[808,track]
     with np.load(verifier.verify(j["input"]["path"],j["input"]["sha256"])) as z:part=z["part_mask"]
     xy=np.rint(a[keep]-j["crop_native"][:2]).astype(int)
     inside=(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<part.shape[1])&(xy[:,1]<part.shape[0])
     flags=np.zeros(len(xy),bool);flags[inside]=part[xy[inside,1],xy[inside,0]]
     row["inliers_outside_lens_region"]=int((~flags).sum())
   rows.append(row)
 (out/"RESULTS.json").write_text(json.dumps(rows,indent=2))
 report=dict(status="TEXTURE_CORRESPONDENCE_PROBE_NOT_POSE_ACCEPTANCE",records=len(rows),expected=30,
   M3T_integrated=False,XYZ_binding_available=False,pose_updated=False,parameters=PARAMS,
   per_track={t:dict(reference_keypoints=next(r["keypoints"] for r in rows if r["track_id"]==t and r["frame_id"]==808),
     candidate_pairs=sum(r["status"]=="image_correspondence_candidate" for r in rows if r["track_id"]==t),
     missing_native_part_records=sum(r["missing_native_part"] for r in rows if r["track_id"]==t),
     inliers=[r.get("inliers",0) for r in rows if r["track_id"]==t and r["frame_id"]!=808]) for t in PARAMS["tracks"]},
   limits=["SAM/B0 instance gating is predicted support, not semantic correctness or occlusion proof.","Homography consensus is a diagnostic, not 3D pose or independent GT.","No trusted keyframe CAD XYZ binding; circle ambiguity is not yet resolved metrically."])
 (out/"COMPLETE.json").write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=="__main__":run()
