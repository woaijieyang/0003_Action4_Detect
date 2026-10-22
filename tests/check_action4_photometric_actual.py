"""Actual full-resolution RGB augmentation and backward/update verification."""
import json,torch,numpy as np
from PIL import Image,ImageDraw
from data.action4_paths import PROJECT_ROOT as R
from data.action4_component_data import ComponentData
from data.action4_dataset import prepare_full_images
from data.action4_photometric import augment_prepared
from data.action4_epoch_batches import EpochRankBatches
from model.action4_release_bundle import warm_start
from model.training.action4_online_b0_batch import batch_losses
from model.training.action4_runtime import atomic_json

torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
cfg=json.loads((R/"configs/model/action4_b0_photometric_v001.json").read_text())
out=R/cfg["output"]
data=ComponentData(cfg["validation_receipt"],"train","B0")
indices=next(iter(EpochRankBatches(len(data),32,4,0,cfg["seed"])))
items=[data[i] for i in indices];labels_before=json.dumps([x["record"] for x in items],sort_keys=True)
seeds=[cfg["seed"]+i*104729 for i in indices]
x,aff=prepare_full_images([x["image"] for x in items],size=tuple(cfg["preprocess"]["full_size"]),device="cuda:0")
y,params=augment_prepared(x,[(int(i["image"].shape[2]),int(i["image"].shape[1])) for i in items],seeds,cfg["photometric"])
assert (y-x).abs().sum()>0 and torch.isfinite(y).all()
mean=x.new_tensor([.485,.456,.406])[None,:,None,None];std=x.new_tensor([.229,.224,.225])[None,:,None,None]
sheet=Image.new("RGB",(1024*2,768*2))
for i in range(2):
    for column,tensor in enumerate([x,y]):
        rgb=((tensor[i:i+1]*std+mean).clamp(0,1)[0].permute(1,2,0)*255).byte().cpu().numpy()
        im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
        for obj in items[i]["record"]["instances"]:
            if obj.get("ignore",False):continue
            b=np.array(obj["bbox_full_xyxy"]).reshape(2,2)
            h=np.c_[b,np.ones(2)]@aff[i].T
            draw.rectangle(h[:,:2].reshape(4).tolist(),outline="cyan",width=2)
        sheet.paste(im,(column*1024,i*768))
sheet.save(out/"photometric_preview.jpg",quality=92)
pixel_difference=float((y-x).abs().mean());del x,y
model,_,optimizer,parent=warm_start("models/B0/best.pt","B0",learning_rate=1e-6)
with torch.no_grad():control={k:float(v) for k,v in batch_losses(model,items,cfg,"cuda:0").items()}
optimizer.zero_grad(set_to_none=True)
losses=batch_losses(model,items,cfg,"cuda:0",augmentation_seeds=seeds)
loss=sum(losses.values());assert torch.isfinite(loss)
parameter=next(model.detector.backbone.parameters());before=parameter.detach().clone()
loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
assert not torch.equal(before,parameter)
assert labels_before==json.dumps([x["record"] for x in items],sort_keys=True)
report={"passed":True,"parent_best_step":parent["selected_step"],"real_microbatch":len(items),"sample_indices":indices,
    "label_records_unchanged":True,"augmentation_normalized_rgb_MAE":pixel_difference,"parameter_update_observed":True,
    "gradient_norm":float(norm),"control_losses":control,"augmented_losses":{k:float(v.detach()) for k,v in losses.items()},
    "parameters":params,"peak_memory_mib":torch.cuda.max_memory_allocated()/1048576,
    "scope":"actual RGB training dataflow/update engineering; no model accuracy claim; sealed weights unchanged"}
atomic_json(out/"PREFLIGHT.json",report);print(json.dumps({k:v for k,v in report.items() if k not in ["parameters","control_losses","augmented_losses"]}))
