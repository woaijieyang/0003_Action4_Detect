"""Once-only internal-contact audit for the newly changed upper-arm region."""
from pathlib import Path
import sys,json,collections
import numpy as np
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
from data.action4_coverage1000_arm_runtime import CoverageArmRuntime
from data.action4_phrase2_collision import Phrase2CollisionChecker,fast_pair
from data.action4_template_collision import Geometry

def capped(geometry):
    counts=collections.Counter();direct={}
    for face in geometry.faces:
        for a,b in zip(face,np.roll(face,-1)):
            edge=tuple(sorted((int(a),int(b))));counts[edge]+=1;direct[edge]=(int(a),int(b))
    edges=[direct[e] for e,n in counts.items() if n==1];ids=sorted({v for e in edges for v in e})
    if len(ids)!=44 or len(edges)!=44:raise ValueError('Exact elbow rim required')
    V=geometry.vertices.tolist();center=len(V);V.append(geometry.vertices[ids].mean(0).tolist());F=geometry.faces.tolist()+[[b,a,center] for a,b in edges]
    result=Geometry.create(V,F)
    if not result.closed:raise ValueError('Closed continuation proxy failed')
    return result

rows=[];failed=[]
for scene in ['table_full_garment','cloth_lift10_full_garment','wall']:
    runtime=CoverageArmRuntime(scene,True,'v002');checker=Phrase2CollisionChecker(runtime)
    for side in ['left','right']:
        tube=capped(checker.cache[side+'_forearm_continuation'])
        for name,other in checker.cache.items():
            if checker.groups[name]!=side or name in [side+'_forearm_continuation',side+'_forearm_natural_hand']:continue
            shape=checker.closed_components.get(name,other)
            matrix=runtime.world[name];axes=[*np.eye(3),*matrix[:3,:3].T]
            result=fast_pair(tube,shape,axes);row={'scene':scene,'side':side,'objects':[side+'_forearm_continuation',name],**result};rows.append(row)
            if not result['status'].startswith('CLEAR'):failed.append(row)
report={'status':'PASS_NEW_UPPERARM_INTERNAL_MECHANICAL_CLEARANCE' if not failed else 'FAILED_NEW_UPPERARM_INTERNAL_CLEARANCE',
    'checks':len(rows),'rows':rows,'failed':failed,'actual_geometry_changed':False,
    'method':'Exact new segment with collision-only elbow cap vs all own mechanical parts; exact hand seam separately certified; no baseline exemption for the changed region.'}
(R/'outputs/coverage1000_v001/arm_repair_v001/INTERNAL_MECHANICAL_AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':report['status'],'checks':len(rows),'failed':failed}),flush=True)
