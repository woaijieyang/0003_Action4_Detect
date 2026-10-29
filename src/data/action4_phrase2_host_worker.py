"""CPU publication workers: decode/measure actual masks, validate CAD sidecars, commit slots."""
import json,time,os,traceback,shutil,hashlib
from pathlib import Path
import numpy as np
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_phrase2_quality import validate,publish
from data import action4_phrase2_queue as queue

def surface_check(record):
    mesh=np.load(R/"data/DJI_Action4_construct/r3_v001/mesh_v001.npz")
    vertices=mesh["vertices_mm"]/1000;faces=mesh["faces"];rows=[]
    for obj in record["instances"]:
        info=obj.get("surface_supervision")
        if not info:raise ValueError("Required phrase2 surface supervision missing")
        p=contained_path(R,info["path"],exists=True)
        with np.load(p) as d:
            xyz=d["xyz_object_m"].astype(float);uv=d["native_pixels"];K=d["K"];T=d["object_to_capture"];index=d["face_indices"]
            if len(xyz)==0 or not np.isfinite(xyz).all() or not np.allclose(K,record["camera"]["K"],atol=1e-8,rtol=0) or not np.allclose(T,obj["canonical8_geometry"]["object_to_capture_cv_m"],atol=1e-8,rtol=0):raise ValueError("Surface identity/finite check failed")
            tri=vertices[faces[index]];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0];normal=np.cross(e1,e2);plane=np.abs(np.einsum("ij,ij->i",xyz-tri[:,0],normal))/np.linalg.norm(normal,axis=1)
            camera=xyz@T[:3,:3].T+T[:3,3];h=camera@K.T;err=np.linalg.norm(h[:,:2]/h[:,2:]-uv,axis=1)
            if (camera[:,2]<=0).any() or plane.max()>1e-6 or err.max()>.005:raise ValueError("Actual CAD surface mismatch")
            rows.append({"instance_id":obj["instance_id"],"points":len(xyz),"max_reprojection_px":float(err.max()),"max_plane_error_m":float(plane.max()),"bytes":p.stat().st_size})
    return rows

def remove_staging(payload):
    p=contained_path(R,payload["staging_dir"],exists=True);slot=payload["slot"];expected=R/"data/phrase2"/slot["split"]/".staging"/slot["sample_id"]
    if not p.is_relative_to(expected) or p.name!=f"a{payload['proposal']['phrase2']['attempt']:05d}" or p.is_symlink():raise ValueError("Not owned candidate directory")
    for root,dirs,files in os.walk(p,followlinks=False):
        if any((Path(root)/name).is_symlink() for name in dirs+files):raise ValueError("Candidate contains a symlink")
    shutil.rmtree(p)

def run(worker_id):
    os.environ["OMP_NUM_THREADS"]="2";os.environ["OPENBLAS_NUM_THREADS"]="1"
    cfg=json.loads((R/"configs/data/action4_phrase2_v001.json").read_text());out=R/cfg["output_root"];processed=0
    directory=out/"quality_workers"/worker_id;directory.mkdir(parents=True,exist_ok=False)
    while not (out/"STOP.json").exists():
        claim=queue.claim(quality=True)
        if claim is None:
            counts=queue.counts()
            if counts.get("accepted",0)==5000:break
            time.sleep(1);continue
        slot=json.loads(claim["spec"])
        try:
            pointer=json.loads(claim["record"])["candidate"];payload=json.loads(contained_path(R,pointer,exists=True).read_text())
            if payload["slot"]!=slot or not payload["collision"]["accepted"] or payload["actual_projection_error_px"]>.005:raise ValueError("Candidate slot/geometry binding mismatch")
            record=payload["record"];ok,reason,stats=validate(record,payload["proposal"],slot,cfg)
            if ok:
                geometry=surface_check(record)
                if not (out/"ROOT_PREVIEW_ACCEPTED.json").exists():
                    queue.transition(slot["slot_id"],"preview_checked",record={"candidate":pointer,"stats":stats,"surface_validation":geometry})
                    (R/payload["runtime_dir"]/"PREVIEW_CHECKED.json").write_text(json.dumps({"slot_id":slot["slot_id"],"actual":stats,"geometry":geometry,"formal_published":False},indent=2))
                else:
                    accepted=publish(record,payload["proposal"],slot,stats)
                    accepted["surface_validation"]=geometry;accepted["source_candidate_record"]=pointer
                    (R/payload["runtime_dir"]/"PUBLISHED_RECORD.json").write_text(json.dumps(accepted))
                    queue.transition(slot["slot_id"],"accepted",record=accepted)
                    (R/payload["runtime_dir"]/"ACCEPTED.json").write_text(json.dumps({"slot_id":slot["slot_id"],"actual":stats,"geometry":geometry,"formal_image":accepted["image"]},indent=2))
                    try:remove_staging(payload)
                    except OSError as error:(R/payload["runtime_dir"]/"CLEANUP_PENDING.json").write_text(json.dumps({"error":str(error),"formal_record_valid":True}))
            else:
                db=queue.connect();db.execute("CREATE TABLE IF NOT EXISTS failures(reason TEXT PRIMARY KEY,count INTEGER)")
                db.execute("INSERT INTO failures VALUES(?,1) ON CONFLICT(reason) DO UPDATE SET count=count+1",(reason,));n=db.execute("SELECT count FROM failures WHERE reason=?",(reason,)).fetchone()[0];db.close()
                (R/payload["runtime_dir"]/"REJECTED.json").write_text(json.dumps({"reason":reason,"measurements":stats,"slot":slot,"attempt":payload["proposal"]["phrase2"]["attempt"]},indent=2))
                if n>5:remove_staging(payload)
                queue.transition(slot["slot_id"],"pending",error=reason)
            processed+=1;(directory/"STATUS.json").write_text(json.dumps({"processed":processed,"pid":os.getpid()}))
        except BaseException:
            failure={"worker":worker_id,"pid":os.getpid(),"slot_id":slot["slot_id"],"traceback":traceback.format_exc()}
            (directory/"FAILURE.json").write_text(json.dumps(failure,indent=2));queue.transition(slot["slot_id"],"failed",error=failure["traceback"]);(out/"STOP.json").write_text(json.dumps(failure,indent=2));raise
    (directory/"COMPLETE.json").write_text(json.dumps({"processed":processed}))
if __name__=="__main__":
    import sys
    run(sys.argv[1])
