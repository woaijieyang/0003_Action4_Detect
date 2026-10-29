"""Fixed cuboid self-visibility plus measured scene rays, without changing GT points."""
import numpy as np

SEMANTICS = 'canonical8_front_facing_incident_face_AND_actual_scene_ray_clear_AND_inframe_v1'

def canonical_visibility(corners_object_m, object_to_capture, scene_ray_clear, inframe):
    corners=np.asarray(corners_object_m,dtype=float)
    T=np.asarray(object_to_capture,dtype=float)
    ray=np.asarray(scene_ray_clear,dtype=bool);inside=np.asarray(inframe,dtype=bool)
    if corners.shape!=(8,3) or T.shape!=(4,4) or ray.shape!=(8,) or inside.shape!=(8,):
        raise ValueError('Exactly eight fixed-ID corners, rigid4x4 and eight ray/inframe flags required')
    if not np.isfinite(corners).all() or not np.isfinite(T).all():raise ValueError('Nonfinite geometry')
    if not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=2e-5,rtol=0):raise ValueError('Rigid transform required')
    origin=-T[:3,:3].T@T[:3,3]
    low=corners.min(0);high=corners.max(0)
    if ((origin>=low-1e-9)&(origin<=high+1e-9)).all():raise ValueError('Observing camera inside target cuboid')
    self_visible=(((origin<low-1e-9)[None,:]&np.isclose(corners,low,atol=1e-9,rtol=0)) |
                  ((origin>high+1e-9)[None,:]&np.isclose(corners,high,atol=1e-9,rtol=0))).any(1)
    return {'self_visible':self_visible.tolist(),'ray_clear':ray.tolist(),'inframe':inside.tolist(),
            'visible':(self_visible&ray&inside).tolist(),'semantics':SEMANTICS}
