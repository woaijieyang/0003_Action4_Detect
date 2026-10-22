import json,time,torch
from data.action4_paths import PROJECT_ROOT as R
from data.action4_component_data import ComponentData
from model.action4_online_b0 import OnlineB0
from model.training.action4_online_b0_batch import batch_losses
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
cfg=json.loads((R/"configs/model/action4_online_b0_v001.json").read_text())
model=OnlineB0(cfg["network"]).cuda()
model.detector.load_state_dict(torch.load(R/cfg["base_best"],map_location="cpu",weights_only=True)["model"])
model.refiner.load_state_dict(torch.load(R/cfg["adapter_best"],map_location="cpu",weights_only=True)["head"])
data=ComponentData(cfg["validation_receipt"],"train","B0");items=[data[i] for i in range(16)]
model.train();begin=time.monotonic()
losses=batch_losses(model,items,cfg,"cuda:0");total=sum(losses.values());total.backward()
torch.cuda.synchronize()
assert torch.isfinite(total)
assert next(model.detector.backbone.parameters()).grad.norm()>0
assert model.refiner.correction.weight.grad.norm()>0
print(json.dumps({"passed":True,"batch_images":16,"seconds_forward_backward":time.monotonic()-begin,"peak_memory_mib":torch.cuda.max_memory_allocated()/1048576,"total_loss":float(total.detach()),"scope":"actualRGB batch16 throughput/memory integration, not accuracy"}))
