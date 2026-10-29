"""Deterministic optional augmentation of predicted ROI448 masks only."""
import hashlib,json
import numpy as np

POLICY={"schema":"action4.sam_mask_augmentation.v1","pixel_units":"roi448",
    "dropout_probability":0.2,"perturb_probability_given_retained":0.25,
    "operations":["erosion","dilation","translation","holes"],
    "morphology_radius_min":1,"morphology_radius_max":3,"morphology_metric":"square_chebyshev",
    "translation_max_each_axis":3,"translation_nonzero":True,
    "holes_count_min":1,"holes_count_max":3,"holes_radius_min":1,"holes_radius_max":3,
    "holes_shape":"disk","holes_centres":"uniform_foreground_without_replacement",
    "rng":"numpy_PCG64_sha256_seed_image_instance_v1"}

def validate_policy(policy):
    if not isinstance(policy,dict) or json.dumps(policy,sort_keys=True,separators=(",",":"))!=json.dumps(POLICY,sort_keys=True,separators=(",",":")):
        raise ValueError("Unknown or changed SAM ROI448 augmentation policy")
    return dict(policy)

def decision(seed,image_id,instance_id,policy):
    validate_policy(policy)
    if type(seed) is not int or seed<0 or any(not isinstance(v,str) or not v for v in (image_id,instance_id)):
        raise ValueError("Explicit nonnegative augmentation seed/image/instance identity required")
    identity=hashlib.sha256(json.dumps({"seed":seed,"image_id":image_id,"instance_id":instance_id,"policy":policy},
        sort_keys=True,separators=(",",":")).encode()).hexdigest()
    rng=np.random.Generator(np.random.PCG64(int(identity[:32],16)))
    drop=bool(rng.random()<policy["dropout_probability"])
    perturb=False if drop else bool(rng.random()<policy["perturb_probability_given_retained"])
    op="dropout" if drop else "identity";params={}
    if perturb:
        op=policy["operations"][int(rng.integers(4))]
        if op in ("erosion","dilation"):params={"radius":int(rng.integers(1,4))}
        elif op=="translation":
            shifts=[(x,y) for x in range(-3,4) for y in range(-3,4) if x or y]
            dx,dy=shifts[int(rng.integers(len(shifts)))];params={"dx":dx,"dy":dy}
        else:
            count=int(rng.integers(1,4))
            params={"radii":[int(rng.integers(1,4)) for _ in range(count)],
                "selector_seed":int(rng.integers(0,2**63-1))}
    return {"policy":dict(policy),"policy_identity":hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        "seed":seed,"image_id":image_id,"instance_id":instance_id,"decision_identity":identity,
        "operation":op,"parameters":params,"dropout":drop,"perturb":perturb,"applied":False}

def apply(mask,valid_support,choice):
    """Copy input, perturb once, and re-clip to actual native-image support."""
    from scipy.ndimage import binary_erosion,binary_dilation
    if mask.shape!=(448,448) or valid_support.shape!=(448,448) or mask.dtype!=np.bool_ or valid_support.dtype!=np.bool_:
        raise ValueError("Bool ROI448 predicted mask/support required")
    validate_policy(choice["policy"])
    # Never accept caller-authored operator parameters outside the seeded policy.
    expected=decision(choice["seed"],choice["image_id"],choice["instance_id"],choice["policy"])
    if choice!=expected:raise ValueError("Augmentation decision differs from deterministic policy")
    log=json.loads(json.dumps(choice));m=mask.copy()&valid_support
    if not m.any():
        log["skip_reason"]="empty_or_unavailable_roi_mask"
        return m,log
    op=choice["operation"];p=choice["parameters"];log["applied"]=op!="identity"
    if op=="dropout":m[:]=False
    elif op in ("erosion","dilation"):
        fn=binary_erosion if op=="erosion" else binary_dilation
        m=fn(m,structure=np.ones((3,3),bool),iterations=p["radius"],border_value=0)
    elif op=="translation":
        moved=np.zeros_like(m);dx,dy=p["dx"],p["dy"]
        sy=slice(max(0,-dy),min(448,448-dy));sx=slice(max(0,-dx),min(448,448-dx))
        moved[max(0,dy):min(448,448+dy),max(0,dx):min(448,448+dx)]=m[sy,sx];m=moved
    elif op=="holes":
        points=np.argwhere(m);rng=np.random.Generator(np.random.PCG64(p["selector_seed"]))
        indices=rng.choice(len(points),size=min(len(points),len(p["radii"])),replace=False)
        centres=[]
        for index,radius in zip(indices,p["radii"]):
            cy,cx=map(int,points[index]);centres.append([cx,cy])
            y0,y1=max(0,cy-radius),min(448,cy+radius+1);x0,x1=max(0,cx-radius),min(448,cx+radius+1)
            yy,xx=np.ogrid[y0:y1,x0:x1];m[y0:y1,x0:x1]&=((xx-cx)**2+(yy-cy)**2>radius**2)
        log["actual_hole_centres_xy"]=centres
    m&=valid_support
    log["pixels_before"]=int((mask&valid_support).sum());log["pixels_after"]=int(m.sum())
    return m,log
