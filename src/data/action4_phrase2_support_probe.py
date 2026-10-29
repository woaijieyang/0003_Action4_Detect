from pathlib import Path
import sys,json,numpy as np
R=Path("/data/zy/vln/a-test");sys.path.insert(0,str(R/"src"))
from data.action4_phrase2_runtime import Phrase2Runtime
from data.action4_phrase2_geometry import rotation_for_arm,pose_for_box
from data.action4_modular_assembly import world_geometry
r=Phrase2Runtime("table_full_garment");o=r.scene.objects["modular_environment_v002_table"]
report={"table_matrix_capture":(np.linalg.inv(r.W)@np.array(o.matrix_world)).tolist(),"table_dims":list(o.dimensions),"table_vertices_local":[list(v.co) for v in o.data.vertices],"examples":[]}
for side,u in [("left",.4),("right",.6)]:
 A=np.array(r.anchor["sides"][side]["A_brace_from_CAD"]);cad=np.array(r.metadata["canonical_definition"]["corners_3d_m"])
 for width in [.08,.11,.14,.18]:
  RB=rotation_for_arm(r,side,0,[-.12 if side=="left" else .12,1,.1]);B,C=pose_for_box(RB,A,cad,r.K,np.array([3248,2464]),[u,.6],width)
  inv,low,high=r.support_solid;inside=[]
  for name in r.metadata["actor_groups"][side]["objects"]:
   obj=r.scene.objects[name]
   if obj.type not in ["MESH","CURVE"] or obj.hide_render:continue
   v,_=world_geometry(obj);base=np.linalg.inv(r.W@np.array(r.metadata["actor_groups"][side]["base_brace_capture_m"]));localB=v@base[:3,:3].T+base[:3,3];v=localB@B[:3,:3].T+B[:3,3];tl=v@inv[:3,:3].T+inv[:3,3];hit=((tl>low+1e-6)&(tl<high-1e-6)).all(1)
   if hit.any():inside.append({"name":name,"vertices_inside":int(hit.sum()),"first_capture":v[hit][0].tolist(),"first_table_local":tl[hit][0].tolist()})
  report["examples"].append({"side":side,"width":width,"body_z":float(C[2,3]),"inside":inside})
(R/"outputs/phrase2_generation_v001/SUPPORT_PROBE.json").write_text(json.dumps(report,indent=2));print(json.dumps(report))
