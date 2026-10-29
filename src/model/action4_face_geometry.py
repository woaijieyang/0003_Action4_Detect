"""Fixed virtual-cuboid FaceID geometry, never observed surface/mask evidence.

Pinhole coordinates follow DataAdapter: OpenCV +Z camera, integer pixel centres.
No import-time I/O, model construction, device allocation or pipeline mutation.
"""
from dataclasses import dataclass, field
import hashlib
import json

import numpy as np
from data.action4_adapter import DataAdapter

_VERIFIED = object()
_NAMES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
_QUADS = ((0,4,7,3),(1,2,6,5),(0,1,5,4),(3,7,6,2),(0,3,2,1),(4,5,6,7))
_NORMALS = ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1))
_ORDER = ("xmin_ymin_zmin","xmax_ymin_zmin","xmax_ymax_zmin","xmin_ymax_zmin",
          "xmin_ymin_zmax","xmax_ymin_zmax","xmax_ymax_zmax","xmin_ymax_zmax")


def _read(data):
    if not isinstance(data, bytes):
        raise TypeError("Actual source document bytes required")
    def reject(value):
        raise ValueError("Nonfinite JSON number: " + value)
    return json.loads(data, parse_constant=reject)


@dataclass(frozen=True)
class FaceGeometryContract:
    """Immutable typed contract authenticated against parent asset source bytes."""
    canonical8_m: tuple
    source_sha256: str
    parent_asset_sha256: str
    parent_asset_path: str
    schema: str = "action4.canonical_six_face_contract.v1"
    _verification: object = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def from_bytes(cls, contract_bytes, parent_asset_bytes):
        doc, parent = _read(contract_bytes), _read(parent_asset_bytes)
        parent_sha = hashlib.sha256(parent_asset_bytes).hexdigest()
        if (doc.get("schema") != "action4.canonical_six_face_contract.v1"
                or doc.get("corner_ids") != list(range(8))
                or any(type(i) is not int for i in doc["corner_ids"])
                or doc.get("parent_asset", {}).get("sha256") != parent_sha
                or not isinstance(doc["parent_asset"].get("path"), str)):
            raise ValueError("Face contract schema/corner IDs/parent source mismatch")
        expected = [{"face_id":i,"object_axis_name":name,"corner_ids":list(quad),
                     "outward_normal_object":list(normal)}
                    for i,(name,quad,normal) in enumerate(zip(_NAMES,_QUADS,_NORMALS))]
        if (doc.get("faces") != expected or any(
                type(f["face_id"]) is not int or any(type(i) is not int for i in f["corner_ids"])
                for f in doc["faces"])):
            raise ValueError("Fixed FaceID, winding, normals and corner topology required")
        if parent.get("units") != "mm" or tuple(parent.get("corner_order", ())) != _ORDER:
            raise ValueError("Parent asset units/canonical point order differ")
        canonical = np.asarray(doc.get("canonical8_m"), dtype=float)
        parent_corners = np.asarray(parent.get("corners_3d_mm"), dtype=float)
        if (canonical.shape != (8,3) or parent_corners.shape != (8,3)
                or not np.isfinite(canonical).all() or not np.isfinite(parent_corners).all()
                or not np.array_equal(canonical, parent_corners / 1000.)):
            raise ValueError("Contract canonical8 must equal verified parent millimetres converted to metres")
        if not (np.ptp(canonical, axis=0) > 0).all():
            raise ValueError("Canonical cuboid must have strictly positive extent on all three axes")
        fixed = DataAdapter.fixed_corners(canonical.min(0), canonical.max(0))
        if not np.array_equal(canonical, fixed):
            raise ValueError("Canonical8 is not the fixed ordered nondegenerate cuboid")
        result = cls(tuple(tuple(float(x) for x in p) for p in canonical),
                     hashlib.sha256(contract_bytes).hexdigest(), parent_sha, doc["parent_asset"]["path"])
        object.__setattr__(result, "_verification", _VERIFIED)
        return result


def _finite_array(value, shape, label):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("Finite " + label + " with shape " + str(shape) + " required")
    return result


def _quad_intersects_image(quad, width, height):
    """Convex polygon/rectangle SAT; leaves returned native quad untouched."""
    rectangle = np.array([[-.5,-.5],[width-.5,-.5],[width-.5,height-.5],[-.5,height-.5]])
    edges = np.roll(quad,-1,axis=0)-quad
    axes = [np.array([1.,0.]),np.array([0.,1.])]
    axes.extend(np.array([-e[1],e[0]]) for e in edges if np.dot(e,e)>0)
    for axis in axes:
        a,b = quad@axis,rectangle@axis
        if a.max()<b.min() or b.max()<a.min():
            return False
    return True


def project_six_faces(canonical8_m, contract, R, t_m, K, *, units,
                      pixel_space_id, image_size_wh, min_depth_m,
                      grazing_cos_threshold):
    """Project one SE3 into fixed eight corners, six quads and twelve edges.

    Thresholds are explicit numerical diagnostics, not calibrated visibility
    gates. View cosine uses outward camera normal toward the camera centre.
    Occlusion/observed evidence remains unknown even for front-facing faces.
    """
    if not isinstance(contract, FaceGeometryContract) or contract._verification is not _VERIFIED:
        raise TypeError("Source-authenticated FaceGeometryContract required")
    if units != "m":
        raise ValueError("Canonical and translation units must explicitly be metres")
    if pixel_space_id not in ("raw_video_native_integer_centers","native_integer_centers"):
        raise ValueError("Explicit native integer-pixel-centre space required")
    if (not isinstance(image_size_wh,(tuple,list)) or len(image_size_wh)!=2
            or any(type(n) is not int or n<=0 for n in image_size_wh)):
        raise ValueError("Positive integer native image width/height required")
    if (isinstance(min_depth_m,bool) or not np.isfinite(min_depth_m) or min_depth_m<=0
            or isinstance(grazing_cos_threshold,bool) or not np.isfinite(grazing_cos_threshold)
            or not 0<=grazing_cos_threshold<1):
        raise ValueError("Explicit finite positive depth and [0,1) grazing thresholds required")
    supplied = np.asarray(canonical8_m)
    if (supplied.shape != (8,3) or supplied.dtype not in (np.dtype("float32"),np.dtype("float64"))
            or not np.isfinite(supplied).all()):
        raise ValueError("Finite float32/float64 canonical8 with shape (8,3) required")
    points = np.asarray(contract.canonical8_m,dtype=np.float64)
    encoded = points.astype(supplied.dtype)
    if not np.array_equal(supplied,encoded):
        raise ValueError("Canonical8 is not the exact contract encoding for its input dtype/order/units")
    input_representation = {"dtype":str(supplied.dtype),
        "encoding":"exact_contract_float32" if supplied.dtype == np.float32 else "exact_contract_float64",
        "projection_geometry":"authenticated_contract_float64",
        "approximate_matching":False}

    rotation = _finite_array(R,(3,3),"R")
    translation = _finite_array(t_m,(3,),"t_m")
    intrinsics = _finite_array(K,(3,3),"K")
    if (not np.allclose(rotation.T@rotation,np.eye(3),rtol=0,atol=1e-6)
            or not np.isclose(np.linalg.det(rotation),1.,rtol=0,atol=1e-6)):
        raise ValueError("R must be a proper SO(3) rotation")
    if (not np.array_equal(intrinsics[2],[0.,0.,1.])
            or intrinsics[0,0]<=0 or intrinsics[1,1]<=0 or intrinsics[1,0]!=0):
        raise ValueError("Pinhole-form K with positive focal lengths required")
    camera = points@rotation.T+translation
    if not np.isfinite(camera).all():
        raise ValueError("SE3 transformation overflow")
    positive = camera[:,2]>min_depth_m
    projected = [None]*8
    if positive.all():
        transform = np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=translation
        uv,_ = DataAdapter.project_corners(points,intrinsics,transform,min_depth=min_depth_m)
        if not np.isfinite(uv).all():
            raise ValueError("Native projection overflow")
        projected = uv.tolist()
    else:
        # Never divide an invalid depth or mask an already nonfinite projection.
        for i in np.flatnonzero(positive):
            homogeneous = intrinsics@camera[i]
            value = homogeneous[:2]/camera[i,2]
            if not np.isfinite(value).all():
                raise ValueError("Native projection overflow")
            projected[int(i)] = value.tolist()
    width,height = image_size_wh
    corners=[]
    for i in range(8):
        uv=projected[i]
        corners.append({"corner_id":i,"camera_xyz_m":camera[i].tolist(),"depth_m":float(camera[i,2]),
            "positive_depth":bool(positive[i]),"native_uv":uv,
            "inside_image":None if uv is None else bool(-.5<=uv[0]<=width-.5 and -.5<=uv[1]<=height-.5)})
    faces=[]
    for face_id,(name,ids,normal) in enumerate(zip(_NAMES,_QUADS,_NORMALS)):
        center_obj=points[list(ids)].mean(0)
        center_cam=center_obj@rotation.T+translation
        normal_cam=rotation@np.asarray(normal,float)
        distance=float(np.linalg.norm(center_cam))
        # Unit normal/cosine roundoff handling changes no point or quad geometry.
        cosine=None if distance<=min_depth_m else float(np.clip(
            np.dot(normal_cam/np.linalg.norm(normal_cam),-center_cam/distance),-1.,1.))
        all_positive=bool(positive[list(ids)].all())
        front=bool(all_positive and cosine is not None and cosine>grazing_cos_threshold)
        orientation=("undefined_view_direction" if cosine is None else
                     "front_facing_candidate" if cosine>grazing_cos_threshold else
                     "backface" if cosine < -grazing_cos_threshold else "grazing")
        quad=[projected[i] for i in ids]
        intersects=None if not all_positive else _quad_intersects_image(np.asarray(quad),width,height)
        status=("nonpositive_or_near_depth" if not all_positive else
                "outside" if not intersects else
                "visible_candidate" if front else orientation)
        faces.append({"face_id":face_id,"object_axis_name":name,"corner_ids":list(ids),
            "quad_native_uv":quad,"quad_camera_xyz_m":camera[list(ids)].tolist(),
            "outward_normal_object":list(normal),"outward_normal_camera":normal_cam.tolist(),
            "center_object_m":center_obj.tolist(),"center_camera_m":center_cam.tolist(),
            "view_cosine":cosine,"positive_depth":all_positive,
            "geometric_front_facing":front,"orientation":orientation,
            "intersects_image":intersects,"diagnostic_status":status,
            "observation_status":"not_measured_geometry_only","occlusion_status":"unknown"})
    pairs=sorted({tuple(sorted((ids[i],ids[(i+1)%4]))) for ids in _QUADS for i in range(4)})
    edges=[{"corner_ids":list(pair),"native_endpoints":[projected[i] for i in pair]} for pair in pairs]
    return {"schema":"action4.fixed_six_face_geometry.v1","units":"m",
        "pixel_space_id":pixel_space_id,"pixel_convention":"integer_centres; image footprint [-0.5,W-0.5] x [-0.5,H-0.5]",
        "image_size_wh":list(image_size_wh),"source_contract_sha256":contract.source_sha256,
        "parent_asset":{"path":contract.parent_asset_path,"sha256":contract.parent_asset_sha256},
        "canonical8_m":[list(p) for p in contract.canonical8_m],
        "canonical_input_representation":input_representation,
        "R":rotation.tolist(),"t_m":translation.tolist(),"K_native":intrinsics.tolist(),
        "thresholds":{"min_depth_m":float(min_depth_m),"grazing_cos_threshold":float(grazing_cos_threshold),
                      "calibrated_visibility_gate":False},
        "corners":corners,"faces":faces,"edges":edges,
        "geometry_semantics":"virtual bounding cuboid faces, not observed physical planar patches",
        "camera_calibration_verified":False,"distortion_handling":"none; follows parent pinhole assumption, not proof of rectification or zero distortion",
        "observation_evidence_used":False,"mask_generated":False,"face_tracking_accepted":False}
