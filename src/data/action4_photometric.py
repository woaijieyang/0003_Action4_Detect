"""Deterministic train-only photometry adapted from GDR-Net's AAE recipe.

No geometric transform, synthetic occluder or label input. Original RGB and
letterbox padding stay immutable. This is not the full upstream augmentation.
"""
import numpy as np
import torch
from torchvision.transforms import functional as TF
from data import DataAdapter

DEFAULT = {"probability":.8,"operation_probability":.5,"add_range":[-25/255,25/255],
           "multiply_range":[.6,1.4],"contrast_range":[.5,2.2],"blur_sigma_max":1.2}

def sample_parameters(seed,config):
    rng=np.random.default_rng(int(seed));p=float(config["operation_probability"])
    parameters={"enabled":bool(rng.random()<config["probability"]),"seed":int(seed)}
    for name in ["blur","add","channel_multiply","multiply","contrast"]:
        parameters[name]=bool(rng.random()<p)
    parameters["sigma"]=float(rng.uniform(.05,config["blur_sigma_max"]))
    for name,interval,channel_probability in [
        ("add_value",config["add_range"],.3),
        ("channel_gain",config["multiply_range"],.5),
        ("contrast_gain",config["contrast_range"],.3)]:
        values=rng.uniform(*interval,size=3)
        if rng.random()>=channel_probability:values[:]=values[0]
        parameters[name]=values.tolist()
    parameters["gain"]=float(rng.uniform(*config["multiply_range"]))
    return parameters

@torch.no_grad()
def augment_prepared(images,source_sizes,seeds,config):
    if len(images)!=len(source_sizes) or len(images)!=len(seeds):raise ValueError("Batch identities disagree")
    if not 0<=config["probability"]<=1 or not 0<=config["operation_probability"]<=1:
        raise ValueError("Augmentation probabilities must be bounded")
    mean=images.new_tensor([.485,.456,.406])[:,None,None]
    std=images.new_tensor([.229,.224,.225])[:,None,None]
    result=images.clone();parameters=[]
    for index,(size,seed) in enumerate(zip(source_sizes,seeds)):
        params=sample_parameters(seed,config);parameters.append(params)
        if not params["enabled"]:continue
        _,info=DataAdapter.letterbox_affine(tuple(size),(images.shape[-1],images.shape[-2]))
        left,top,right,bottom=info["padding_ltrb"];rw,rh=info["resize_size"]
        # Process only image content. The exact normalized black padding survives.
        rgb=(images[index,:,top:top+rh,left:left+rw]*std+mean).clamp(0,1)
        if params["blur"]:rgb=TF.gaussian_blur(rgb,[5,5],[params["sigma"],params["sigma"]])
        if params["add"]:rgb=(rgb+rgb.new_tensor(params["add_value"])[:,None,None]).clamp(0,1)
        if params["channel_multiply"]:rgb=(rgb*rgb.new_tensor(params["channel_gain"])[:,None,None]).clamp(0,1)
        if params["multiply"]:rgb=(rgb*params["gain"]).clamp(0,1)
        if params["contrast"]:rgb=((rgb-.5)*rgb.new_tensor(params["contrast_gain"])[:,None,None]+.5).clamp(0,1)
        result[index,:,top:top+rh,left:left+rw]=(rgb-mean)/std
    return result,parameters
