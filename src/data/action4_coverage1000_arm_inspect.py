"""Read-only current mesh topology and seam measurements for an isolated repair."""
from pathlib import Path
import sys,json
import numpy as np
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
from data.action4_phrase2_runtime import Phrase2Runtime
from data.action4_arm_endcap_candidate import boundary_components
from data.action4_modular_assembly import world_geometry
r=Phrase2Runtime('table_full_garment');rows={}
for side in ['left','right']:
    inv=np.linalg.inv(r.W@np.array(r.metadata['actor_groups'][side]['base_brace_capture_m']))
    obj=r.scene.objects[side+'_forearm_continuation'];hand=r.scene.objects[side+'_forearm_natural_hand']
    v,f=world_geometry(obj);v=v@inv[:3,:3].T+inv[:3,3]
    hv,hf=world_geometry(hand);hv=hv@inv[:3,:3].T+inv[:3,3]
    groups,_=boundary_components(obj.data);hg,_=boundary_components(hand.data)
    rings=[]
    for i in range(len(v)//44):
        p=v[i*44:(i+1)*44];center=p.mean(0)
        rings.append({'index':i,'center_B':center.tolist(),'mean_radius_m':float(np.linalg.norm(p-center,axis=1).mean()),'source_faces':int(np.isin(f,np.arange(i*44,(i+1)*44)).any(1).sum())})
    rows[side]={'runtime_arm_axis_B':r.arm_axis_B[side].tolist(),'junction_B':r.arm_junction_points_B[side].mean(0).tolist(),
        'continuation_boundaries':[sorted(g) for g in groups],'hand_boundary_ids':sorted(hg[0]),'rings':rings,
        'continuation_objects_transform':np.array(obj.matrix_world).tolist(),'table_plane':None if r.table_plane is None else [np.asarray(a).tolist() for a in r.table_plane]}
    p=R/'outputs/coverage1000_v001/arm_repair_v001';p.mkdir(exist_ok=True)
    np.savez_compressed(p/(side+'_source.npz'),vertices_B=v,faces=f,hand_vertices_B=hv,hand_faces=hf,hand_boundary_ids=sorted(hg[0]))
out=R/'outputs/coverage1000_v001/arm_repair_v001/ORIGINAL_ARM_TOPOLOGY.json';out.open('x').write(json.dumps(rows,indent=2));print(json.dumps(rows),flush=True)
