"""Frozen pinned YOLOE adapter; import has no model/network side effects."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import numpy as np
from data.action4_paths import PROJECT_ROOT, contained_path

PREDICT_ARGS=dict(imgsz=1280,conf=.05,iou=1.,max_det=300,nms=None,agnostic_nms=False,retina_masks=True,save=False,verbose=False)
TEXT_CLASSES=('action camera','camera','smartwatch','wristwatch','wristband','bracelet','mobile phone')

def rgb_to_bgr(rgb):
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
        raise ValueError('Expected raw uint8 HWC RGB with exactly three channels')
    return np.ascontiguousarray(rgb[:,:,::-1])

def sha256(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def local_environment():
    for key,name in dict(HF_HOME='hf',TORCH_HOME='torch',XDG_CACHE_HOME='xdg',PIP_CACHE_DIR='pip',TMPDIR='tmp',CUDA_CACHE_PATH='cuda',YOLO_CONFIG_DIR='ultralytics').items():
        p=contained_path(PROJECT_ROOT,'.cache/yoloe/'+name); p.mkdir(parents=True,exist_ok=True); os.environ[key]=str(p)
    os.environ['YOLO_AUTOINSTALL']='False'


def verify_source_tree(repo,expected_commit):
    """Reject modified tracked source and untracked executable Python overlays."""
    repo=contained_path(PROJECT_ROOT,repo,exists=True)
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True)
    if git('rev-parse','HEAD').strip()!=expected_commit:raise ValueError('Pinned source mismatch')
    if git('status','--porcelain','--untracked-files=no').strip():raise ValueError('Pinned tracked tree is modified')
    untracked=git('ls-files','--others','--exclude-standard').splitlines()
    if any(Path(n).suffix in ('.py','.so','.pyd') for n in untracked):raise ValueError('Untracked importable source in pinned tree')
    paths=git('ls-files').splitlines()
    source_hashes={}
    for name in paths:
        if Path(name).suffix=='.py':
            path=contained_path(PROJECT_ROOT,repo/name,exists=True)
            source_hashes[name]=sha256(path)
    return dict(commit=expected_commit,tracked_tree_clean=True,python_files=source_hashes,
                python_tree_sha256=hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest())

def validate_prediction_arrays(boxes,scores,classes,masks,native_shape):
    b=np.asarray(boxes);s=np.asarray(scores);c=np.asarray(classes);n=len(b)
    if b.shape!=(n,4) or s.shape!=(n,) or c.shape!=(n,):raise ValueError('Prediction lengths/shapes disagree')
    if not np.isfinite(b).all() or not np.isfinite(s).all() or not np.isfinite(c).all():raise ValueError('Nonfinite prediction')
    if np.any(b[:,2:]<=b[:,:2]) or np.any((s<0)|(s>1)) or np.any(c<0) or np.any(c!=np.floor(c)):raise ValueError('Invalid boxes/scores/classes')
    if masks is None:
        if n:raise ValueError('Segmentation boxes have no masks')
    else:
        m=np.asarray(masks)
        if m.shape!=(n,*native_shape) or not np.isfinite(m).all() or np.any((m<0)|(m>1)):raise ValueError('Invalid native masks')

def squeeze_official_mask_channel(masks):
    """Remove only the singleton channel introduced by official LetterBox."""
    if not isinstance(masks,np.ndarray) or masks.ndim!=4 or masks.shape[-1]!=1:
        raise ValueError('Expected official mask pre_transform output[N,H,W,1]')
    return masks[...,0]

TEXT_SPEC_SCHEMA='action4.yoloe.text_spec.v1'

def default_text_spec():
    roles=('positive','positive','negative','negative','negative','negative','negative')
    return dict(schema=TEXT_SPEC_SCHEMA,profile_id='legacy_text_fixed7',claim_scope='camera_proxy_specificity_unverified',
        classes=[dict(class_id=i,text=t,evidence_role=roles[i],semantic_class='camera' if roles[i]=='positive' else t)
                 for i,t in enumerate(TEXT_CLASSES)])

def validate_text_spec(spec):
    if not isinstance(spec,dict) or set(spec)!={'schema','profile_id','claim_scope','classes'} or spec['schema']!=TEXT_SPEC_SCHEMA:
        raise ValueError('Explicit text spec schema required')
    if not isinstance(spec['profile_id'],str) or not spec['profile_id'].strip() or spec['claim_scope']!='camera_proxy_specificity_unverified':
        raise ValueError('Text profiles cannot assert unmeasured model specificity')
    rows=spec['classes']
    if not isinstance(rows,list) or not 1<=len(rows)<=64:raise ValueError('Expected1..64 declared text classes')
    for i,row in enumerate(rows):
        if not isinstance(row,dict) or set(row)!={'class_id','text','evidence_role','semantic_class'} or type(row['class_id']) is not int or row['class_id']!=i:
            raise ValueError('Class IDs must exactly match declared native NPZ order')
        if row['evidence_role'] not in ('positive','negative','neutral') or any(not isinstance(row[k],str) or not row[k].strip() for k in ('text','semantic_class')):
            raise ValueError('Every native class needs explicit text/role/semantic mapping')
    if len({r['text'] for r in rows})!=len(rows) or not any(r['evidence_role']=='positive' for r in rows):
        raise ValueError('Unique texts and at leastone positive class required')
    return json.loads(json.dumps(spec,sort_keys=True))

def text_spec_sha256(spec):
    return hashlib.sha256(json.dumps(validate_text_spec(spec),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def class_evidence_map(*,text=False,spec=None):
    if not text:
        if spec is not None:raise ValueError('Text spec supplied for visual profile')
        return {'0':dict(text='object0',evidence_role='positive',semantic_class='camera')}
    resolved=validate_text_spec(default_text_spec() if spec is None else spec)
    return {str(r['class_id']):{k:r[k] for k in ('text','evidence_role','semantic_class')} for r in resolved['classes']}

def evidence_availability(classes,*,text=False,spec=None):
    c=np.asarray(classes);mapping=class_evidence_map(text=text,spec=spec);ids=[int(k) for k in mapping]
    if not np.isin(c,ids).all():raise ValueError('Unknown native output class')
    positives=[int(k) for k,v in mapping.items() if v['evidence_role']=='positive']
    negatives=[int(k) for k,v in mapping.items() if v['evidence_role']=='negative']
    positive=bool(np.isin(c,positives).any())
    return dict(inference_response_available=1,target_evidence_available=int(positive),
        negative_evidence_available=int(np.isin(c,negatives).any()),status='raw_target_candidates' if positive else 'unknown_target',
        class_evidence_map=mapping,target_class_ids=positives,negative_class_ids=negatives)

def load_text_profile_contract(profile_path,*,contract_path=None,expected_assets=None,expected_contract_sha256=None):
    p=contained_path(PROJECT_ROOT,profile_path,exists=True)
    side=contained_path(PROJECT_ROOT,contract_path,exists=True) if contract_path is not None else p.with_suffix(p.suffix+'.contract.json')
    with np.load(p,allow_pickle=False) as data:names=data['names'].tolist()
    profile_sha=sha256(p)
    if side.exists():
        side=contained_path(PROJECT_ROOT,side,exists=True);contract=json.loads(side.read_text())
        if contract.get('schema')!='action4.yoloe.text_profile_contract.v1':raise ValueError('Unknown text profile contract')
        spec=validate_text_spec(contract['spec'])
        if contract['profile_sha256']!=profile_sha or contract['spec_sha256']!=text_spec_sha256(spec):raise ValueError('Text profile/spec identity changed')
        if expected_assets is not None and contract['assets']!=expected_assets:raise ValueError('Text profile model/encoder source mismatch')
        contract_sha=sha256(side)
        if expected_contract_sha256 is None and text_spec_sha256(spec)!=text_spec_sha256(default_text_spec()):raise ValueError('Nonlegacy text profile requires caller-frozen contract SHA256')
        if expected_contract_sha256 is not None and contract_sha!=expected_contract_sha256:raise ValueError('Frozen text contract SHA256 mismatch')
    else:
        if expected_contract_sha256 is not None:raise ValueError('Declared text contract is missing')
        # Backward compatibility is exact-name checked; a new seven-class profile
        # cannot silently inherit the historical0/1 semantics.
        spec=default_text_spec();contract_sha=None
        if names!=list(TEXT_CLASSES):raise ValueError('Nonlegacy text NPZ requires its frozen class-map contract')
    if names!=[r['text'] for r in spec['classes']]:raise ValueError('Native NPZ names/order do not match text class contract')
    return dict(spec=spec,spec_sha256=text_spec_sha256(spec),profile_sha256=profile_sha,
                contract_sha256=contract_sha,contract_path=str(side.relative_to(PROJECT_ROOT)) if side.exists() else None)

def select_role_predictions(result,*,score_threshold=.25,role='positive',id_prefix=None):
    """Native class IDs/roles are retained. Negative evidence never hard-vetoes."""
    if role not in ('positive','negative','neutral') or not np.isfinite(score_threshold) or not 0<=score_threshold<=1:
        raise ValueError('Invalid declared publication policy')
    mapping=result.get('class_evidence_map')
    if not isinstance(mapping,dict):raise ValueError('Explicit class evidence map required')
    prefix=result['reference_id'] if id_prefix is None else id_prefix
    selected=[]
    for j,(score,cls) in enumerate(zip(result['scores'],result['classes'])):
        if not np.isfinite(cls) or cls!=int(cls):raise ValueError('Native class ID must be finite integer')
        key=str(int(cls))
        if key not in mapping:raise ValueError('Output class missing from frozen mapping')
        row=mapping[key]
        if score>=score_threshold and row['evidence_role']==role:
            if result['masks'] is None:raise ValueError('Published segmentation output lacks native mask')
            selected.append(dict(id=str(prefix)+':'+str(j),source='yolo',class_id=int(cls),score=float(score),
                bbox=result['boxes'][j].tolist(),prompt_bbox=result['boxes'][j].tolist(),mask=result['masks'][j]>=.5,
                evidence_role=role,semantic_class=row['semantic_class'],text=row['text'],reference_id=result['reference_id'],
                profile_sha256=result['profile_sha256'],text_spec_sha256=result.get('text_spec_sha256')))
    return selected

def remap_prediction_semantics(predictions,project_class_map):
    """Explicit project class space for combining different native profiles.

    Native IDs, text, role and profile identity remain on every output record.
    """
    if not isinstance(project_class_map,dict) or not project_class_map or any(type(v) is not int or v<0 for v in project_class_map.values()) or len(set(project_class_map.values()))!=len(project_class_map):
        raise ValueError('Declare a unique nonnegative project ID for each semantic class')
    result=[]
    for p in predictions:
        if p['semantic_class'] not in project_class_map:raise ValueError('Unmapped semantic class')
        result.append(dict(p,native_class_id=p.get('native_class_id',p['class_id']),class_id=project_class_map[p['semantic_class']],class_id_space='project_semantic_v1'))
    return result

def validate_runtime_resources(lock,device):
    """Reject unregistered devices before imports, weight loading or allocation."""
    threads=lock.get('max_cpu_threads')
    if type(threads) is not int or not 1<=threads<=4:
        raise ValueError('Runtime max_cpu_threads must be an integer in1..4')
    if lock.get('cpu_only') is True:
        if str(device)!='cpu':raise ValueError('CPU-only runtime lock rejects non-CPU device')
        return threads
    mapping=lock.get('gpu_mapping',{})
    allowed_uuid='GPU-a77a9d4c-7642-7112-698c-b02673d254cb'
    if lock.get('schema')!='action4.yoloe_runtime_gpu1.v1' or str(device)!='cuda:0':
        raise ValueError('Explicit reviewed GPU1 runtime contract required')
    if mapping.get('physical_index')!=1 or mapping.get('physical_uuid')!=allowed_uuid or mapping.get('logical_device')!='cuda:0':
        raise ValueError('Unapproved physical/logical GPU mapping')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=allowed_uuid:
        raise ValueError('CUDA_VISIBLE_DEVICES must be exact approvedGPU1 UUID')
    if lock.get('max_gpu_memory_gib')!=8 or lock.get('torch_allocator_budget_gib')!=7:
        raise ValueError('GPU memory contract must reserve context margin under8GiB')
    return threads

def verify_runtime_lock(path='configs/model/action4_yoloe_runtime_lock_v001.json',*,device='cpu'):
    import sys
    import importlib
    lock=json.loads(contained_path(PROJECT_ROOT,path,exists=True).read_text())
    validate_runtime_resources(lock,device)
    if str(Path(sys.prefix))!=lock['environment_prefix']:
        raise ValueError('Use the locked isolated .venv/yoloe interpreter')
    for name,spec in lock['modules'].items():
        module=importlib.import_module(name)
        if str(Path(module.__file__).resolve())!=spec['path'] or str(getattr(module,'__version__','unknown'))!=spec['version']:
            raise ValueError('Runtime import identity changed: '+name)
    return lock

class FrozenYOLOE:
    def __init__(self, config='configs/model/action4_yoloe_assets_v001.json', *, device='cpu',runtime_lock='configs/model/action4_yoloe_runtime_lock_v001.json'):
        local_environment()
        self.runtime_identity=verify_runtime_lock(runtime_lock,device=device)
        self.config=json.loads(contained_path(PROJECT_ROOT,config,exists=True).read_text())
        self.source_identity={source['path']:verify_source_tree(source['path'],source['commit']) for source in self.config['sources']}
        self.assets={}
        for key,spec in self.config['assets'].items():
            p=contained_path(PROJECT_ROOT,spec['path'],exists=True)
            if sha256(p)!=spec['sha256']:raise ValueError('Asset SHA mismatch: '+key)
            self.assets[key]=p
        from ultralytics import YOLOE
        import ultralytics
        import clip
        origin=Path(ultralytics.__file__).resolve()
        if not origin.is_relative_to(contained_path(PROJECT_ROOT,self.config['sources'][0]['path'])):
            raise ValueError('Imported ultralytics is not pinned project source')
        if not Path(clip.__file__).resolve().is_relative_to(contained_path(PROJECT_ROOT,self.config['sources'][1]['path'])):
            raise ValueError('Imported CLIP is not pinned project source')
        # Official select_device can reset CPU threads; all execution modes use
        # the one reviewed thread budget, preserving the CPU-only contract.
        import torch
        import ultralytics.utils as yolo_utils
        import ultralytics.utils.torch_utils as yolo_torch_utils
        yolo_utils.NUM_THREADS=self.runtime_identity['max_cpu_threads']
        yolo_torch_utils.NUM_THREADS=self.runtime_identity['max_cpu_threads']
        torch.set_num_threads(self.runtime_identity['max_cpu_threads'])
        if str(device)=='cuda:0':
            if torch.cuda.device_count()!=1:raise ValueError('GPU contract requires exactly one visibleGPU')
            total=torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(self.runtime_identity['torch_allocator_budget_gib']*1024**3/total,0)
        self.device=device
        self.model=YOLOE(str(self.assets['detector']))
        self.model.model.to(device).eval().requires_grad_(False)
        self.dimension=int(self.model.model.model[-1].embed)

    def _check_profile(self,path,classes=1):
        p=contained_path(PROJECT_ROOT,path,exists=True)
        with np.load(p,allow_pickle=False) as f:
            e=f['embeddings']
            if e.shape!=(1,classes,self.dimension) or e.dtype!=np.float32 or not np.isfinite(e).all():
                raise ValueError('Invalid official embedding dimensions/dtype/values')
        return p

    def register_reference(self,rgb,box,profile_path):
        b=np.asarray(box,dtype=np.float32); h,w=rgb.shape[:2]
        if b.shape!=(4,) or not np.isfinite(b).all() or not(0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h):
            raise ValueError('Reference bbox must be in reference image coordinates')
        p=contained_path(PROJECT_ROOT,profile_path)
        if p.exists():raise FileExistsError(p)
        p.parent.mkdir(parents=True,exist_ok=True)
        reference=rgb_to_bgr(rgb)
        import torch
        with torch.inference_mode():
            self.model.predict(source=reference,refer_image=reference,visual_prompts={'bboxes':b[None,:],'cls':np.array([0],np.int64)},device=self.device,**PREDICT_ARGS)
            self.model.save_prompt_embeddings(p)
        self._check_profile(p)
        return dict(path=str(p.relative_to(PROJECT_ROOT)),sha256=sha256(p),shape=[1,1,self.dimension])

    def register_mask_reference(self,rgb,body_mask,profile_path):
        """Official mask prompt with a shape-only project bridge; no value edits."""
        import torch
        from ultralytics.models.yolo.yoloe.predict import YOLOEVPSegPredictor
        from ultralytics.data.augment import LoadVisualPrompt
        if not isinstance(body_mask,np.ndarray) or body_mask.dtype!=np.bool_ or body_mask.shape!=rgb.shape[:2] or not body_mask.any():
            raise ValueError('Expected nonempty reference bool mask matching RGB')
        p=contained_path(PROJECT_ROOT,profile_path)
        if p.exists():raise FileExistsError(p)
        p.parent.mkdir(parents=True,exist_ok=True);trace=[]
        class NativeMaskPredictor(YOLOEVPSegPredictor):
            def _process_single_image(self,dst_shape,src_shape,category,bboxes=None,masks=None):
                if bboxes is not None or masks is None:raise ValueError('Mask bridge requires masks without bboxes')
                original=np.stack(super().pre_transform(masks))
                padding_count=int(np.count_nonzero(original==114))
                original[original==114]=0  # unchanged official padding cleanup
                fixed=squeeze_official_mask_channel(original)
                trace.append(dict(official_shape=list(original.shape),fixed_shape=list(fixed.shape),
                    dtype=str(original.dtype),official_min=float(original.min()),official_max=float(original.max()),
                    nonzero_before=int(np.count_nonzero(original)),nonzero_after=int(np.count_nonzero(fixed)),
                    sum_before=float(original.sum()),sum_after=float(fixed.sum()),
                    exact_values_equal=bool(np.array_equal(original.reshape(fixed.shape),fixed)),
                    shares_storage=bool(np.shares_memory(original,fixed)),official_padding114_cleared=padding_count))
                return LoadVisualPrompt().get_visuals(category,dst_shape,bboxes=None,masks=fixed)
        predictor=NativeMaskPredictor(overrides={**self.model.overrides,**PREDICT_ARGS,'device':self.device,'task':'segment','mode':'predict','batch':1},_callbacks=self.model.callbacks)
        predictor.set_prompts({'cls':np.asarray([0],np.int64),'masks':body_mask[None].astype(np.float32)})
        predictor.setup_model(model=self.model.model,verbose=False);predictor.model.names={0:'object0'}
        with torch.inference_mode():
            embedding=predictor.get_vpe(rgb_to_bgr(rgb))
            self.model.set_classes(['object0'],embeddings=embedding)
            self.model.save_prompt_embeddings(p)
        self._check_profile(p)
        return dict(path=str(p.relative_to(PROJECT_ROOT)),sha256=sha256(p),shape=[1,1,self.dimension],mask_shape_bridge=trace)

    def register_text(self,profile_path,*,text_spec=None):
        import torch
        from ultralytics.nn.text_model import MobileCLIPTS
        spec=validate_text_spec(default_text_spec() if text_spec is None else text_spec)
        classes=[r['text'] for r in spec['classes']]
        p=contained_path(PROJECT_ROOT,profile_path);side=p.with_suffix(p.suffix+'.contract.json')
        if p.exists() or side.exists():raise FileExistsError(p)
        p.parent.mkdir(parents=True,exist_ok=True)
        self.model.model.clip_model=MobileCLIPTS(device=torch.device(self.device),weight=str(self.assets['text_encoder'])).eval().requires_grad_(False)
        with torch.inference_mode():
            pe=self.model.model.get_text_pe(classes,cache_clip_model=True)
            self.model.set_classes(classes,embeddings=pe)
            self.model.save_prompt_embeddings(p)
        self._check_profile(p,len(classes))
        contract=dict(schema='action4.yoloe.text_profile_contract.v1',spec=spec,spec_sha256=text_spec_sha256(spec),profile_sha256=sha256(p),
            assets={k:self.config['assets'][k]['sha256'] for k in ('detector','text_encoder')})
        temporary=side.with_suffix(side.suffix+'.part');temporary.write_text(json.dumps(contract,indent=2)+'\n');temporary.rename(side)
        return dict(path=str(p.relative_to(PROJECT_ROOT)),sha256=contract['profile_sha256'],classes=classes,
                    spec_sha256=contract['spec_sha256'],contract_path=str(side.relative_to(PROJECT_ROOT)),contract_sha256=sha256(side))

    def predict(self,rgb,*,profile_path,reference_id,text=False,text_contract_path=None,text_contract_sha256=None):
        import torch
        binding=load_text_profile_contract(profile_path,contract_path=text_contract_path,expected_contract_sha256=text_contract_sha256,expected_assets={k:self.config['assets'][k]['sha256'] for k in ('detector','text_encoder')}) if text else None
        if not text and (text_contract_path is not None or text_contract_sha256 is not None):raise ValueError('Text contract is invalid for visual inference')
        spec=binding['spec'] if binding is not None else None
        p=self._check_profile(profile_path,len(spec['classes']) if text else 1)
        with torch.inference_mode():
            self.model.load_prompt_embeddings(p)
            self.model.model.eval().requires_grad_(False)
            result=self.model.predict(source=rgb_to_bgr(rgb),device=self.device,**PREDICT_ARGS)[0]
        boxes=result.boxes.xyxy.cpu().numpy(); scores=result.boxes.conf.cpu().numpy(); classes=result.boxes.cls.cpu().numpy()
        masks=None if result.masks is None else result.masks.data.cpu().numpy()
        validate_prediction_arrays(boxes,scores,classes,masks,rgb.shape[:2])
        classes=classes.astype(np.int64)
        availability=evidence_availability(classes,text=text,spec=spec)
        return dict(boxes=boxes,masks=masks,scores=scores,classes=classes,reference_id=reference_id,
            **availability,
            source='yolo',profile_sha256=sha256(p),source_identity=self.source_identity,runtime_identity=self.runtime_identity,asset_identity=self.config,arguments=dict(PREDICT_ARGS),
            native_shape=list(rgb.shape[:2]),class_names=result.names,text_spec_sha256=binding['spec_sha256'] if binding else None,
            text_profile_contract=binding)
