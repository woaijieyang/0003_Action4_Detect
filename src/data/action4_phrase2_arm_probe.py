from pathlib import Path
import sys,json,numpy as np
R=Path("/data/zy/vln/a-test");sys.path.insert(0,str(R/"src"))
from data.action4_phrase2_runtime import Phrase2Runtime
from data.action4_arm_endcap_candidate import boundary_components
r=Phrase2Runtime("wall");rows=[]
for side in ["left","right"]:
 B=np.array(r.metadata["actor_groups"][side]["base_brace_capture_m"]);inv=np.linalg.inv(r.W@B);axis=r.arm_axis_B[side];summary={}
 for role in ["forearm_natural_hand","forearm_continuation"]:
  obj=r.scene.objects[side+"_"+role];m=inv@np.array(obj.matrix_world);v=np.array([p.co[:] for p in obj.data.vertices])@m[:3,:3].T+m[:3,3];projection=v@axis;radius=np.linalg.norm(v-projection[:,None]*axis,axis=1)
  groups,_=boundary_components(obj.data);summary[role]={"axis_range_m":[float(projection.min()),float(projection.max())],"max_radius_m":float(radius.max()),"boundary_centroids_B":[v[sorted(g)].mean(0).tolist() for g in groups],"material_names":[x.name for x in obj.data.materials],"UV_layers":[x.name for x in obj.data.uv_layers],"vertex_count":len(v)}
 rows.append({"side":side,"terminal_distance_from_brace_m":float(np.linalg.norm(r.arm_terminal_B[side])),"roles":summary})
(R/"outputs/phrase2_generation_v001/ARM_PROPORTION_PROBE.json").write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
