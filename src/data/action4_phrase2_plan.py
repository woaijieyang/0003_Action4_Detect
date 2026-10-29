"""Exact phrase2 slots; independent deterministic attempts, no Blender at import."""
import hashlib,json,random,math
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R,contained_path

def seed_for(master,split,slot,attempt,stream="parameters"):
    raw=f"{master}|{split}|{slot}|{attempt}|{stream}".encode()
    return int.from_bytes(hashlib.blake2b(raw,digest_size=8).digest(),"little")

def draw_interval(rng,spec,outer,unit=None):
    q=rng.random() if unit is None else unit
    a,b=spec["total"];c,d=spec["core"]
    if not outer:return c+q*(d-c)
    length=(c-a)+(b-d);p=q*length
    # Sample only the disjoint complement; never redraw from total.
    return a+p if p<c-a else d+(p-(c-a))

def category(value,spec):
    a,b=spec["total"];c,d=spec["core"]
    if not a<=value<=b:return "outside"
    return "core" if c<=value<=d else "outer"

def slots(cfg):
    result=[];scenes=list(cfg["scene_counts"]);master=cfg["master_seed"];index=0
    for split,count in cfg["split_counts"].items():
        rng=random.Random(seed_for(master,split,0,0,"quota"))
        pos=[key for key,n in cfg["joint_position_quotas"][split].items() for _ in range(n)]
        scene=[scene for scene,n in zip(scenes,cfg["scene_counts_by_split"][split]) for _ in range(n)]
        left=[j for j,n in enumerate(cfg["occlusion_quota_by_side_by_split"][split]) for _ in range(n)];right=left.copy()
        for values in [pos,scene,left,right]:rng.shuffle(values)
        assert all(len(x)==count for x in [pos,scene,left,right])
        for i in range(count):
            result.append({"slot_id":index,"sample_id":f"p2_{split}_{i:05d}","split":split,"scene_id":scene[i],
                "outer_coordinates":pos[i],"occlusion_bins":{"left":left[i],"right":right[i]},
                "group_id":f"phrase2_{split}_{i:05d}","attempt":0})
            index+=1
    random.Random(seed_for(master,"all",0,0,"schedule")).shuffle(result)
    return result

def preview_ids(rows,count=128):
    selected=[];remaining=list(rows);covered=set()
    # Explicit coverage of each scene/position signature/side occlusion bin.
    while remaining and len(selected)<count:
        def tags(r):return {("scene",r["scene_id"]),("position",r["outer_coordinates"]),("left",r["occlusion_bins"]["left"]),("right",r["occlusion_bins"]["right"]),("split",r["split"])}
        best=max(range(len(remaining)),key=lambda i:len(tags(remaining[i])-covered))
        r=remaining.pop(best);selected.append(r["slot_id"]);covered|=tags(r)
        if len(selected)>40 and not any(tags(x)-covered for x in remaining):break
    for r in remaining:
        if len(selected)>=count:break
        selected.append(r["slot_id"])
    return selected

def write_plan(config="configs/data/action4_phrase2_v001.json"):
    cfg=json.loads(contained_path(R,config,exists=True).read_text());rows=slots(cfg)
    target=contained_path(R,cfg["output_root"],exists=False);target.mkdir(exist_ok=True)
    plan={"schema":"action4.phrase2_slots.v1","configuration":config,"count":len(rows),"preview_slot_ids":preview_ids(rows,cfg["preview_count"]),"slots":rows}
    p=target/"SLOTS.json"
    if p.exists():
        if json.loads(p.read_text())!=plan:raise ValueError("Existing slot identities differ; do not overwrite")
    else:
        with p.open("x") as f:json.dump(plan,f,indent=2)
    return plan
if __name__=="__main__":print(json.dumps({"slots":write_plan()["count"]}))
