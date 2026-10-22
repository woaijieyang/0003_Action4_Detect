"""Actual synthetic correspondences verify iterative refinement and fallback."""
import json,numpy as np,torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_gdr_pose_data import geometry
from model.action4_pose_refinement import refine_network_pose
manifest=json.loads((R/"outputs/action4_gdr_surface_candidate_v001/CACHE_COMPLETE.json").read_text())
part=torch.load(R/manifest["splits"]["train"]["parts"][0]["path"],map_location="cpu",weights_only=True)
g=geometry(part["metadata"][:2],"train");rows=[]
for i in range(2):
    xyz=part["targets"]["xyz"][i].numpy();mask=part["targets"]["valid"][i].numpy().astype(float)
    K=g["K"][i].numpy();cad=g["canonical"][i].numpy();rot=g["rotation"][i].numpy();t=g["translation"][i].numpy()
    perturbed=t+np.array([.01,-.01,.02])
    result=refine_network_pose(xyz,mask,K,cad,rot,perturbed)
    assert result["valid"] and result["refined"]
    error=float(np.max(np.abs(np.array(result["corners_hm"])-g["corners"][i].numpy())))
    assert error<.001
    fallback=refine_network_pose(xyz,np.zeros_like(mask),K,cad,rot,t)
    assert fallback["valid"] and not fallback["refined"] and "insufficient" in fallback["reason"]
    rows.append({"object":part["metadata"][i]["sample_id"],"max_corner_error_px":error})
print(json.dumps({"passed":True,"oracle_correspondence_test":True,"rows":rows,"scope":"geometry solver test, not predicted accuracy"}))
