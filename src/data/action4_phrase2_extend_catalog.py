"""Reuse all normal-spread calibration and append only wide-spread forearm candidates."""
from pathlib import Path
import sys,json,hashlib,numpy as np
R=Path("/data/zy/vln/a-test");sys.path.insert(0,str(R/"src"))
def run():
    from data.action4_phrase2_runtime import Phrase2Runtime
    from data.action4_phrase2_geometry import LocalArmVisibility,rotation_for_arm,pose_for_box
    from data.action4_phrase2_sampling import occlusion_bin
    root=R/"outputs/phrase2_generation_v001";parent=root/"OCCLUSION_CATALOG_v005.json";catalog=json.loads(parent.read_text());runtime=Phrase2Runtime("wall");cad=np.array(runtime.metadata["canonical_definition"]["corners_3d_m"]);size=np.array([3248,2464])
    for side in ["left","right"]:
        tree=LocalArmVisibility(runtime,side);A=np.array(runtime.anchor["sides"][side]["A_brace_from_CAD"]);normal=-.12 if side=="left" else .12;wide=-.65 if side=="left" else .65
        for anchor in catalog["rows"][side]["anchors"]:
            for x in anchor["samples"]:x["direction_x"]=normal
            for dz in [-.4,-.8,-1.3]:
                for roll in np.arange(-180,181,2):
                    RB=rotation_for_arm(runtime,side,float(roll),[wide,1,dz]);B,C=pose_for_box(RB,A,cad,runtime.K,size,anchor["center_uv"],.11);m=tree.estimate(B,C,20,return_sources=True)
                    anchor["samples"].append({"roll_deg":float(roll),"direction_x":wide,"direction_z":dz,**m})
            anchor["eligible_counts"]=[sum(x["occlusion_estimate"] is not None and occlusion_bin(x["occlusion_estimate"])==i and (i==0 or x["primary_class"]=="own_wrist_forearm") for x in anchor["samples"]) for i in range(3)]
            print(side,anchor["center_uv"],anchor["eligible_counts"],flush=True)
    catalog["parent_catalog_ref"]={"path":str(parent.relative_to(R)),"sha256":hashlib.sha256(parent.read_bytes()).hexdigest()};catalog["normal_spread_results_reused"]=True;catalog["new_scope"]="only additional ±.65 forearm horizontal direction, all fixed mounts and dimensions unchanged"
    (root/"OCCLUSION_CATALOG_v006.json").write_text(json.dumps(catalog,indent=2))
if __name__=="__main__":run()
