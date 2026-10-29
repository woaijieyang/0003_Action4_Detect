"""One coarse arm-roll/source scan over a small set of camera sightlines."""
from pathlib import Path
import sys,json,numpy as np
R=Path("/data/zy/vln/a-test");sys.path.insert(0,str(R/"src"))
def run():
    from data.action4_phrase2_runtime import Phrase2Runtime
    from data.action4_phrase2_geometry import LocalArmVisibility,rotation_for_arm,pose_for_box
    from data.action4_phrase2_sampling import occlusion_bin
    runtime=Phrase2Runtime("wall");size=np.array([runtime.scene.render.resolution_x,runtime.scene.render.resolution_y]);cad=np.array(runtime.metadata["canonical_definition"]["corners_3d_m"]);rows={}
    for side,us in [("left",[.20,.40,.60]),("right",[.45,.65,.90])]:
        tree=LocalArmVisibility(runtime,side);A=np.array(runtime.anchor["sides"][side]["A_brace_from_CAD"]);anchors=[]
        for u in us:
            for v in [.40,.60,.85]:
                values=[]
                for direction_z in [-.4,-.8,-1.3]:
                    for roll in np.arange(-180,181,2):
                        direction=[-.12 if side=="left" else .12,1,direction_z]
                        RB=rotation_for_arm(runtime,side,float(roll),direction);B,C=pose_for_box(RB,A,cad,runtime.K,size,[u,v],.11)
                        m=tree.estimate(B,C,20,return_sources=True);values.append({"roll_deg":float(roll),"direction_z":direction_z,**m})
                counts=[sum(occlusion_bin(x["occlusion_estimate"])==i and (i==0 or x["primary_class"]=="own_wrist_forearm") for x in values if x["occlusion_estimate"] is not None) for i in range(3)]
                anchors.append({"center_uv":[u,v],"samples":values,"eligible_counts":counts})
                print(side,u,v,counts,flush=True)
        rows[side]={"anchors":anchors,"arm_axis_B":runtime.arm_axis_B[side].tolist()}
    out=R/"outputs/phrase2_generation_v001/OCCLUSION_CATALOG_v005.json";out.write_text(json.dumps({"schema":"action4.phrase2_arm_occlusion_catalog.v2","method":"local body vs same-side actor; nine sightlines per side,2degree coarse scan,20grid; actual scene and masks final","rows":rows,"medium_heavy_primary_source":"own_wrist_forearm","training_accepted":False},indent=2))
if __name__=="__main__":run()
