"""Explicit phrase2 dataset root; never falls back to the first-stage manifests."""
import json,numpy as np,torch
from PIL import Image
from data import DataAdapter
from data.action4_paths import PROJECT_ROOT as R,contained_path

class Phrase2Dataset:
    def __init__(self,split,*,dataset_root="data/phrase2",role="B0",require_complete=True):
        if dataset_root!="data/phrase2" or split not in ["train","eval","test"] or role not in ["A0","B0"]:raise ValueError("Explicit phrase2 root/split/role required")
        self.root=contained_path(R,dataset_root,exists=True);self.split=split;self.role=role
        if require_complete:
            commit=self.root/"DATASET_COMPLETE.json"
            if not commit.exists() or json.loads(commit.read_text()).get("status")!="FORMAL_DATASET_USABLE":
                raise ValueError("phrase2 has not passed final dataset release")
            admission=json.loads(commit.read_text())
            if not admission.get('similarity_dataset_identity'):
                raise ValueError('phrase2 requires a current split-global similarity admission')
            proof=json.loads(contained_path(R,admission['similarity_proof'],exists=True).read_text())
            if proof.get('status')!='PASS_GLOBAL_SIMILARITY_AUDIT' or proof.get('dataset_identity')!=admission['similarity_dataset_identity']:
                raise ValueError('Missing or stale phrase2 similarity proof')
            expected=admission['manifest_sha256'][split]
            actual=__import__('hashlib').sha256((self.root/split/'annotations.jsonl').read_bytes()).hexdigest()
            if actual!=expected:raise ValueError('Manifest changed after final similarity admission')
        path=contained_path(self.root,split+"/annotations.jsonl",exists=True)
        self.records=[json.loads(line) for line in path.read_text().splitlines()]
        counts=admission["splits"] if require_complete else {split:len(self.records)}
        if require_complete and len(self.records)!=counts[split]:raise ValueError("phrase2 split incomplete")
        if len({r["sample_id"] for r in self.records})!=len(self.records):raise ValueError("Duplicate sample IDs")
        for row in self.records:
            if row["dataset_root"]!=dataset_root or row["split"]!=split or row["status"]!="training_accepted" or [g["instance_id"] for g in row["instances"]]!=[1,2]:raise ValueError("Wrong dataset identity/qualification")
            if not contained_path(R,row["image"],exists=True).is_relative_to(self.root/split):raise ValueError("Image points to old or wrong split")
            DataAdapter(R).validate_sample(row,check_image=False)
        self.units=[(i,j) for i in range(len(self.records)) for j in range(2)] if role=="A0" else [(i,None) for i in range(len(self.records))]
    def __len__(self):return len(self.units)
    def __getitem__(self,index):
        i,j=self.units[index];record=self.records[i]
        with Image.open(contained_path(R,record["image"],exists=True)) as image:rgb=torch.from_numpy(np.array(image.convert("RGB"))).permute(2,0,1)
        return {"image":rgb,"record":record,"instance":j}

def surface_source():
    return {"admission":"data/phrase2/GEOMETRY_ADMISSION.json","regions":"data/action4_surface_regions_v001/centers.json",
            "label_roots":{s:f"data/phrase2/{s}/geometry" for s in ["train","eval","test"]}}
