"""Reuse unchanged validated splits; validate missing eval/test assets once."""
import json,hashlib,os
from data.action4_paths import PROJECT_ROOT as R,contained_path
from data.action4_dataset import Action4Dataset,is_ignored

def prepare(output,reuse=None,workers=12):
    target=contained_path(R,output,exists=False)
    if target.exists():raise FileExistsError("Receipt exists; reuse it or choose a new version")
    receipt={"schema":"action4.component_data_validation.v1","splits":{},"test_role":"historical test, not a newly blinded acceptance set"}
    if reuse:
        old=json.loads(contained_path(R,reuse,exists=True).read_text())
        for split,entry in old["splits"].items():
            manifest=contained_path(R,entry["manifest"],exists=True)
            if hashlib.sha256(manifest.read_bytes()).hexdigest()!=entry["sha256"]:raise ValueError("Reused manifest changed")
            for name,identity in entry["stats"].items():
                if Action4Dataset._file_identity(contained_path(R,name,exists=True))!=tuple(identity):
                    raise ValueError("Reused image identity changed; fresh validation required")
            receipt["splits"][split]=entry
        receipt["reused_receipt"]=reuse
    for split in ("train","eval","test"):
        if split in receipt["splits"]:continue
        manifest=R/f"data/{split}/annotations.jsonl"
        data=Action4Dataset(R,manifest,split=split,validation_workers=workers)
        if split=="test" and any(sum(not is_ignored(g) for g in r["instances"])!=2 for r in data.records):
            raise ValueError("This project's test split must contain only dual-camera images")
        receipt["splits"][split]={"manifest":str(manifest.relative_to(R)),"sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),"stats":data._validated_stats}
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix(".tmp");temporary.write_text(json.dumps(receipt));os.replace(temporary,target)
    print(json.dumps({"receipt":str(target.relative_to(R)),"images":{s:len(x["stats"]) for s,x in receipt["splits"].items()},"RGB_rehashed_splits":[s for s in ("train","eval","test") if not reuse or s not in old["splits"]]}))

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--reuse");p.add_argument("--workers",type=int,default=12);a=p.parse_args();prepare(a.output,a.reuse,a.workers)
