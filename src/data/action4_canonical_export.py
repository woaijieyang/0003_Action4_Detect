"""Pure fixed-canonical8 diagnostic export and trusted-context reprojection checks.

No scene, model, filesystem, or renderer operations. Virtual cuboid vertices
are not physical mesh corners; projection validity is not surface visibility.
"""
import copy
import numpy as np

ORDER=('xmin_ymin_zmin','xmax_ymin_zmin','xmax_ymax_zmin','xmin_ymax_zmin',
       'xmin_ymin_zmax','xmax_ymin_zmax','xmax_ymax_zmax','xmin_ymax_zmax')
BLENDER_CAMERA_TO_CV=np.diag([1.,-1.,-1.,1.])


def _rigid(value,name):
    a=np.asarray(value,float)
    if a.shape!=(4,4) or not np.isfinite(a).all() or not np.allclose(a[3],[0,0,0,1],rtol=0,atol=1e-9):
        raise ValueError(name+' must be a finite homogeneous matrix')
    if not np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),rtol=0,atol=1e-5) or abs(np.linalg.det(a[:3,:3])-1)>1e-5:
        raise ValueError(name+' must be proper rigid, without scale or reflection')
    return a


def _canonical(definition):
    if definition.get('corner_order')!=list(ORDER):raise ValueError('Original canonical8 order required')
    ref=definition.get('source_manifest',{})
    if set(ref)!={'path','sha256'} or not isinstance(ref['path'],str) or not ref['path'] or len(ref['sha256'])!=64 or any(c not in '0123456789abcdef' for c in ref['sha256']):
        raise ValueError('Bound canonical manifest identity required')
    mm=np.asarray(definition.get('corners_3d_mm'),float);m=np.asarray(definition.get('corners_3d_m'),float)
    if mm.shape!=(8,3) or m.shape!=(8,3) or not np.isfinite(mm).all() or not np.isfinite(m).all():raise ValueError('Finite ordered8 coordinates required')
    if not np.array_equal(m,mm*.001):raise ValueError('Canonical mm-to-m conversion differs')
    lo,hi=mm.min(0),mm.max(0)
    expected=np.array([[lo[0],lo[1],lo[2]],[hi[0],lo[1],lo[2]],[hi[0],hi[1],lo[2]],[lo[0],hi[1],lo[2]],
                       [lo[0],lo[1],hi[2]],[hi[0],lo[1],hi[2]],[hi[0],hi[1],hi[2]],[lo[0],hi[1],hi[2]]])
    if not (hi>lo).all() or not np.array_equal(mm,expected):raise ValueError('Canonical vertices reordered or non-cuboid')
    return mm,m


def build_export(*,canonical,instance_id,object_to_world_m,camera_to_world_blender_m,K,
                 image_size_wh,near_plane_m,ray_clear_to_virtual_point=None,
                 legacy_corners_px=None,legacy_corner_visible=None):
    """Build all8 geometry even outside/behind; invalid pixel projections are null.

    Matrices are actual evaluated scene poses supplied by the caller. K applies
    to original native pixel centers. Pixel extent is inclusive [-.5,W-.5],
    distinct from heatmap supervised center-domain [0,63].
    """
    mm,obj=_canonical(canonical)
    if type(instance_id) is not int or instance_id not in (1,2):raise ValueError('Physical Action4 instance_id must be1/2')
    ow=_rigid(object_to_world_m,'object_to_world');cw=_rigid(camera_to_world_blender_m,'camera_to_world')
    k=np.asarray(K,float)
    if k.shape!=(3,3) or not np.isfinite(k).all() or not np.allclose(k[2],[0,0,1],atol=1e-12,rtol=0) or min(k[0,0],k[1,1])<=0:
        raise ValueError('Finite pinhole K required')
    if len(image_size_wh)!=2 or any(type(x) is not int or x<=0 for x in image_size_wh):raise ValueError('Positive integer native dimensions')
    if isinstance(near_plane_m,bool) or not np.isfinite(near_plane_m) or near_plane_m<=0:raise ValueError('Positive near plane required')
    width,height=image_size_wh;wc=BLENDER_CAMERA_TO_CV@np.linalg.inv(cw);oc=wc@ow
    world=(np.c_[obj,np.ones(8)]@ow.T)[:,:3];capture=(np.c_[world,np.ones(8)]@wc.T)[:,:3];depth=capture[:,2]
    positive=depth>0;projectable=depth>near_plane_m;pixels=[];inside=[]
    for point,ok in zip(capture,projectable):
        if not ok:pixels.append(None);inside.append(False);continue
        q=k@point;uv=q[:2]/q[2]
        if not np.isfinite(uv).all():raise ValueError('Nonfinite projected point')
        pixels.append(uv.tolist());inside.append(bool(-.5<=uv[0]<=width-.5 and -.5<=uv[1]<=height-.5))
    if ray_clear_to_virtual_point is None:ray_clear_to_virtual_point=[None]*8
    if len(ray_clear_to_virtual_point)!=8 or any(x is not None and type(x) is not bool for x in ray_clear_to_virtual_point):raise ValueError('Eight boolean/null ray-clear states')
    ray_clear=[x if ok else None for x,ok in zip(ray_clear_to_virtual_point,projectable)]
    old_points=None;old_visible=None;error=None
    if legacy_corners_px is not None:
        old=np.asarray(legacy_corners_px,float)
        if old.shape!=(8,2) or not np.isfinite(old).all() or not positive.all():raise ValueError('Legacy finite8 pixels require all positive depths')
        q=capture@k.T;reprojected=q[:,:2]/q[:,2,None];error=float(np.max(abs(old-reprojected)))
        if error>.003:raise ValueError('Legacy ordered8 pixels fail trusted-context reprojection')
        old_points=old.tolist()
    if legacy_corner_visible is not None:
        if len(legacy_corner_visible)!=8 or any(type(v) is not bool for v in legacy_corner_visible):raise ValueError('Legacy eight visibility booleans required')
        if old_points is None:raise ValueError('Legacy visibility requires legacy pixels')
        old_visible=list(legacy_corner_visible)
        for q,ray,vis in zip(old_points,ray_clear_to_virtual_point,old_visible):
            inframe=(-.5<=q[0]<=width-.5 and -.5<=q[1]<=height-.5)
            if ray is not None and vis!=bool(ray and inframe):raise ValueError('Legacy virtual-ray visibility contradicts its pixel extent')
    return {'schema':'action4.canonical8_geometry_export.v1','instance_id':instance_id,'object_id':1,
        'canonical_source_manifest':copy.deepcopy(canonical['source_manifest']),'corner_ids':list(range(8)),
        'corner_order':list(ORDER),'points_object_mm':mm.tolist(),'points_object_m':obj.tolist(),
        'object_to_world_m':ow.tolist(),'camera_to_world_blender_m':cw.tolist(),'world_to_capture_cv_m':wc.tolist(),
        'object_to_capture_cv_m':oc.tolist(),'points_world_m':world.tolist(),'points_capture_m':capture.tolist(),
        'K_native':k.tolist(),'image_size_wh':list(image_size_wh),'near_plane_m':float(near_plane_m),
        'points_native_px':pixels,'depth_capture_m':depth.tolist(),'point_positive_depth':positive.tolist(),
        'point_near_plane_projectable':projectable.tolist(),'point_inframe':inside,
        'ray_clear_to_virtual_point':ray_clear,'legacy_corners_px':old_points,'legacy_corner_visible':old_visible,
        'legacy_reprojection_max_error_px':error,
        'coordinate_convention':'CAD/world/capture lengths in meters; capture OpenCV +X right,+Y down,+Z forward; Blender camera converted by diag(1,-1,-1)',
        'projection_convention':'native integer pixel centers; inclusive image extent[-.5,W-.5]x[-.5,H-.5]; near-projectable iff z>near; invalid pixels null; never clamp',
        'visibility_semantics':'Legacy corner_visible = ray clear to virtual cuboid point AND normalized-image inframe. Neither legacy nor ray_clear denotes visible physical mesh corners. Projectability and heatmap supervision validity are separate.',
        'training_accepted':False,'claim':'Diagnostic export of fixed virtual cuboid; no calibration or training-data acceptance'}


def verify_export(value,**trusted_context):
    """Rebuild from external canonical/actual pose/K and actual row legacy pixels.

    Caller must supply trusted source/runtime context, never take it from the
    value being checked. This catches ordered-point, unit, pose and K changes.
    """
    expected=build_export(**trusted_context)
    tolerances={'world_to_capture_cv_m':1e-10,'object_to_capture_cv_m':1e-10,
                'points_world_m':1e-10,'points_capture_m':1e-10,'depth_capture_m':1e-10,
                'points_native_px':1e-6,'legacy_reprojection_max_error_px':1e-6}
    def numeric_equal(actual,reference,tolerance):
        if isinstance(reference,list):
            return isinstance(actual,list) and len(actual)==len(reference) and all(numeric_equal(a,b,tolerance) for a,b in zip(actual,reference))
        if reference is None:return actual is None
        return (not isinstance(actual,bool) and isinstance(actual,(int,float)) and np.isfinite(actual)
                and abs(actual-reference)<=tolerance)
    def exact_equal(actual,reference):
        if type(actual) is not type(reference):return False
        if isinstance(reference,list):return len(actual)==len(reference) and all(exact_equal(a,b) for a,b in zip(actual,reference))
        if isinstance(reference,dict):return set(actual)==set(reference) and all(exact_equal(actual[k],v) for k,v in reference.items())
        return actual==reference
    if not isinstance(value,dict) or set(value)!=set(expected):raise ValueError('Canonical8 export schema mismatch')
    for key,reference in expected.items():
        good=numeric_equal(value[key],reference,tolerances[key]) if key in tolerances else exact_equal(value[key],reference)
        if not good:raise ValueError('Canonical8 export differs from trusted source/pose/camera/ordered-row pixels: '+key)
    return True
