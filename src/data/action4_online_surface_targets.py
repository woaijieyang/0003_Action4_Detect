"""Reproject accepted sparse CAD correspondences into the actual jittered RGB crop.

No synthetic images or labels are rewritten. Native visible surface points stay
in object coordinates; only their 2D sampling locations change with the ROI.
Masks are sampled from their original visible/amodal images, never interpolated
from an old ROI mask. Returned labels are training/evaluation targets only.
"""
import json
import numpy as np
import torch
from PIL import Image
from data import DataAdapter
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_pose_rgb_batch import prepare_pose_rgb

def _sample_binary_mask_image(image,native_pixels,origin=(0,0),expected_wh=None):
    pixels=np.rint(np.asarray(native_pixels)-np.asarray(origin)).astype(np.int64)
    if expected_wh is not None and image.size!=tuple(expected_wh):raise ValueError("Mask dimensions changed")
    inside=(pixels[:,0]>=0)&(pixels[:,1]>=0)&(pixels[:,0]<image.width)&(pixels[:,1]<image.height)
    values=np.zeros(len(pixels),dtype=np.float32)
    if inside.any():
        selected=pixels[inside];low=selected.min(0);high=selected.max(0)+1
        crop=np.asarray(image.crop((*low,*high)))
        if crop.ndim!=2:raise ValueError("Binary mask must have one channel")
        values[inside]=(crop[selected[:,1]-low[1],selected[:,0]-low[0]]>0)
    return values

def sample_binary_mask(path,native_pixels,origin=(0,0),expected_wh=None):
    with Image.open(contained_path(R,path,exists=True)) as image:
        return _sample_binary_mask_image(image,native_pixels,origin,expected_wh)

def prepare_online_surface_batch(items,split,*,device="cpu",seeds=None,boxes=None,jitter_config=None,surface_source=None,crop_specs=None,roi_expansion=2.25,roi_source=None):
    if not items:raise ValueError("Nonempty explicitly selected surface batch required")
    if split not in ("train","eval","test") or any(x["record"]["split"]!=split for x in items):
        raise ValueError("Explicit matching data split required")
    source=surface_source or {"admission":"outputs/action4_surface_labels_v002/ROOT_VISUAL_REVIEW.json",
        "regions":"data/action4_surface_regions_v001/centers.json",
        "label_roots":{s:f"data/{s}/surface_v002_candidate" for s in ("train","eval","test")}}
    review=json.loads(contained_path(R,source["admission"],exists=True).read_text())
    if not review["candidate_training_allowed"]:raise ValueError("Surface supervision not admitted")
    label_root=contained_path(R,source["label_roots"][split],exists=True)
    centers=np.array(json.loads(contained_path(R,source["regions"],exists=True).read_text())["xyz_object_m"])
    if centers.shape!=(64,3):raise ValueError("Frozen64 region identity required")
    rgb,geometry,metadata=prepare_pose_rgb(items,device=device,seeds=seeds,boxes=boxes,jitter_config=jitter_config,crop_specs=crop_specs,roi_expansion=roi_expansion,roi_source=roi_source)
    yy,xx=np.meshgrid(np.arange(112),np.arange(112),indexing="ij");grid=np.c_[xx.ravel(),yy.ravel()]
    sparse=[];masks=[]
    for item,meta in zip(items,metadata):
        record=item["record"];obj=record["instances"][item["instance"]]
        A=np.asarray(meta["affine_hm"]);T=np.array(obj["canonical8_geometry"]["object_to_capture_cv_m"])
        path=contained_path(R,label_root/f"{record['sample_id']}_i{obj['instance_id']}.npz",exists=True)
        with np.load(path) as data:
            points=np.array(data["xyz_object_m"],dtype=float);native=np.array(data["native_pixels"],dtype=float)
            if not np.allclose(data["K"],record["camera"]["K"],atol=1e-8,rtol=0) or not np.allclose(data["object_to_capture"],T,atol=1e-8,rtol=0):
                raise ValueError("Surface object/camera identity mismatch")
        uv=DataAdapter.transform_points(native,A)
        keep=np.isfinite(uv).all(1)&(uv>=0).all(1)&(uv<=111).all(1)
        # Legitimate empty crop: retain explicit zero support; invalid files still fail above.
        selected=points[keep];cad=np.asarray(meta["canonical_m"]);low=cad.min(0);extent=cad.max(0)-low
        normalized=(selected-low)/extent
        if not np.isfinite(normalized).all():raise ValueError("Invalid CAD coordinates")
        region=np.argmin(np.sum((selected[:,None]-centers[None])**2,axis=-1),axis=1)+1
        sparse.append({"xy_hm":torch.tensor(uv[keep],dtype=torch.float32,device=device),
                       "xyz":torch.tensor(normalized,dtype=torch.float32,device=device),
                       "regions":torch.tensor(region,dtype=torch.long,device=device)})
        pixels=DataAdapter.transform_points(grid,np.linalg.inv(A))
        down448,_=DataAdapter.letterbox_affine((112,112),(448,448))
        A448=down448@A
        yy448,xx448=np.meshgrid(np.arange(448),np.arange(448),indexing='ij')
        p448=DataAdapter.transform_points(np.c_[xx448.ravel(),yy448.ravel()],np.linalg.inv(A448))
        # One item-local PIL image serves both grids; no cross-item cache.
        with Image.open(contained_path(R,obj["mask_visible"],exists=True)) as visible_image:
            visible=_sample_binary_mask_image(visible_image,pixels,expected_wh=(record["width"],record["height"]))
            visible_support=int(_sample_binary_mask_image(visible_image,p448,
                expected_wh=(record["width"],record["height"])).sum())
        amodal=sample_binary_mask(obj["mask_amodal_expanded"],pixels,origin=obj["amodal_expanded_origin_px"],
                                  expected_wh=tuple(obj["amodal_expanded_shape_hw"][::-1]))
        masks.append(np.stack([visible,amodal]).reshape(2,112,112))
        camera=selected@T[:3,:3].T+T[:3,3]
        native_hom=camera@np.asarray(record["camera"]["K"]).T
        if len(selected) and ((camera[:,2]<=0).any() or np.max(np.linalg.norm(native_hom[:,:2]/native_hom[:,2:]-native[keep],axis=1))>.005):
            raise ValueError("Surface ray/XYZ projection no longer matches admitted native labels")
        hom=camera@(A@np.asarray(record["camera"]["K"])).T
        error=np.linalg.norm(hom[:,:2]/hom[:,2:]-uv[keep],axis=1)
        meta.update(surface_source=str(path.relative_to(R)),surface_points=len(selected),
                    dropped_surface_points=int((~keep).sum()),surface_projection_error_hm_px=float(error.max()) if len(error) else None)
        meta['target_visible_input_support_count']=visible_support
        meta['target_identity_valid']=True
        meta['gt_geometry_valid']=True
    targets={"sparse":sparse,"masks":torch.tensor(np.asarray(masks),dtype=torch.float32,device=device)}
    return rgb,targets,geometry,metadata
