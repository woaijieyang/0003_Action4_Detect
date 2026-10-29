"""Frozen geometry-head intrinsic sensitivity; neither calibration nor deployment."""
import json,os,hashlib,subprocess
import numpy as np
import torch
from torch import nn
from data.action4_paths import PROJECT_ROOT as ROOT,contained_path
from data.action4_surface_snapshot import load_snapshot
from model.action4_release_bundle import load_bundle
from model.action4_gdr_pose import GDRPose
from eval.action4_roi_sweep import read_binding,binding


class SavedSurface(nn.Module):
    def __init__(self,values):super().__init__();self.values=values
    def forward(self,features):return self.values


class SavedGeometry(nn.Module):
    forward=GDRPose.forward
    def __init__(self,pose,values):super().__init__();self.pose=pose;self.surface=SavedSurface(values)


def camera_variants(cfg,baseline_K):
    if 'camera_hypotheses' in cfg:
        if 'focal_native_px' in cfg:raise ValueError('Choose full K hypotheses or focal-only hypotheses')
        items=cfg['camera_hypotheses']
        if not isinstance(items,list) or not 1<=len(items)<=4:raise ValueError('Bounded explicit camera hypotheses required')
        result=[];names=set()
        for item in items:
            name=item['name'];K=np.asarray(item['K_native'],float)
            if not isinstance(name,str) or not name or name in names or K.shape!=(3,3) or not np.isfinite(K).all() or not np.array_equal(K[2],[0,0,1]) or K[0,0]<=0 or K[1,1]<=0:raise ValueError('Invalid named pinhole K')
            names.add(name);result.append((name,K))
        return result
    result=[]
    for focal in cfg['focal_native_px']:
        K=np.array(baseline_K,float,copy=True);K[0,0]=K[1,1]=focal;result.append((str(focal),K))
    return result


def run(config_path):
    cfg=json.loads(contained_path(ROOT,config_path,exists=True).read_text())
    if set(os.sched_getaffinity(0))!=set(cfg['cpus']) or not set(cfg['cpus'])<=set(range(24,96)):raise ValueError('Wrong CPU allocation')
    if cfg['gpu'] not in [4,5,6,7] or os.environ.get('CUDA_VISIBLE_DEVICES')!=cfg['gpu_uuid']:raise ValueError('Wrong physical GPU allocation')
    info=subprocess.check_output(['nvidia-smi','-i',str(cfg['gpu']),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    if info[0].strip()!=cfg['gpu_uuid'] or int(info[1])<12288:raise ValueError('GPU headroom unavailable')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory)
    out=contained_path(ROOT,cfg['output'],exists=False);out.mkdir(exist_ok=False)
    lock=read_binding(cfg['input_lock']);pair=read_binding(lock['formal_pair_gate']);a=pair['models']['A0']
    model,base,state=load_bundle(a['path'],'A0','cuda:0',allow_corner_candidate=True,expected_sha256=a['sha256'])
    model.eval().requires_grad_(False);base.eval().requires_grad_(False)
    if state['selected_step']!=cfg['expected_step']:raise ValueError('Wrong current parent')
    before=tuple((n,p._version) for n,p in list(model.named_parameters())+list(model.named_buffers()))
    tensor=lambda v:torch.from_numpy(np.asarray(v).copy()).float()[None].cuda()
    loaded=[];parity=[];records=[];calls=0
    with torch.inference_mode():
        for item in cfg['surfaces']:
            path=contained_path(ROOT,item['surface']['path'],exists=True)
            if binding(path)!=item['surface']:raise ValueError('Surface changed')
            snap=load_snapshot(path);z=snap.arrays;meta=snap.metadata
            if meta['models']['A0']['sha256']!=a['sha256'] or meta['prediction_id']!=f"video:{item['track_id']}:frame:{item['frame_id']}":raise ValueError('Wrong snapshot model/frame/track')
            values={key:tensor(z[name]) for key,name in [('xyz','xyz_normalized'),('regions','region_logits'),('masks','mask_logits')]}
            replay=SavedGeometry(model.head.pose,values).eval();proxy=torch.zeros((1,1,1,1),device='cuda:0');cad=tensor(z['canonical_m'])
            answer=replay(proxy,tensor(z['K112']),cad);calls+=1
            rot=answer['rotation'][0].cpu().numpy();t=answer['translation'][0].cpu().numpy()
            dr=float(np.max(np.abs(rot-z['rotation'])));dt=float(np.max(np.abs(t-z['translation_m'])))
            check=dict(frame_id=item['frame_id'],track_id=item['track_id'],rotation_max_abs=dr,translation_max_abs_m=dt)
            parity.append(check)
            if dr>1e-4 or dt>1e-5:
                (out/'BASELINE_FAILURE.json').write_text(json.dumps(check,indent=2));raise ValueError('Baseline cache replay failed; no focal variants allowed')
            hom=answer['camera_corners']@tensor(z['K_native']).transpose(1,2);uv=hom[:,:,:2]/hom[:,:,2:]
            native=uv[0].cpu().numpy();roi=answer['corners'][0].cpu().numpy();back=np.c_[roi,np.ones(8)]@np.linalg.inv(z['A112']).T;back=back[:,:2]/back[:,2:]
            err=float(np.max(np.abs(native-back)))
            if not np.isfinite(native).all() or err>1e-3:raise ValueError('Baseline native projection inconsistency')
            baseline=dict(rotation=rot.tolist(),translation_m=t.tolist(),corners_native=native.tolist(),K_native=z['K_native'].tolist(),K112=z['K112'].tolist(),ROI_backprojection_max_error_px=err)
            loaded.append((item,z,values,baseline))
        (out/'BASELINE_PARITY.json').write_text(json.dumps(dict(status='ALL_CACHED_BASELINES_PASS',records=parity),indent=2))
        for item,z,values,baseline in loaded:
            replay=SavedGeometry(model.head.pose,values).eval();proxy=torch.zeros((1,1,1,1),device='cuda:0');cad=tensor(z['canonical_m']);variants={}
            for name,K in camera_variants(cfg,z['K_native']):
                if np.array_equal(K,z['K_native']):
                    variants[name]=baseline
                    continue
                kr=z['A112']@K
                answer=replay(proxy,tensor(kr),cad);calls+=1
                camera=answer['camera_corners'];hom=camera@tensor(K).transpose(1,2);uv=hom[:,:,:2]/hom[:,:,2:]
                native=uv[0].cpu().numpy();roi=answer['corners'][0].cpu().numpy();back=np.c_[roi,np.ones(8)]@np.linalg.inv(z['A112']).T;back=back[:,:2]/back[:,2:]
                if not np.isfinite(native).all() or float(np.max(np.abs(native-back)))>1e-3:raise ValueError('ROI/native projection inconsistency')
                variants[name]=dict(rotation=answer['rotation'][0].tolist(),translation_m=answer['translation'][0].tolist(),corners_native=native.tolist(),K_native=K.tolist(),K112=np.asarray(kr).tolist(),ROI_backprojection_max_error_px=float(np.max(np.abs(native-back))))
            records.append(dict(frame_id=item['frame_id'],track_id=item['track_id'],surface=item['surface'],variants=variants))
    after=tuple((n,p._version) for n,p in list(model.named_parameters())+list(model.named_buffers()))
    if before!=after:raise ValueError('Frozen model mutated')
    report=dict(status='INTRINSICS_SENSITIVITY_NOT_CALIBRATION_OR_ACCURACY',records=records,models=pair['models'],
        focal_native_px=cfg.get('focal_native_px'),camera_hypotheses=cfg.get('camera_hypotheses'),pose_head_calls=calls,RGB_backbone_calls=0,SAM_calls=0,B0_calls=0,
        source_forward='existing GDRPose.forward with authenticated fixed surface outputs; K is not consumed upstream of surface',
        unchanged=['RGB crop','SAM auxiliaries','surface predictions','ROI affine','CAD','weights','raw published pipeline'],
        changed='K_native -> K112 -> camera rays AND existing relative-depth decode -> native projection',
        defaults_changed=False,GT_used=False,calibration_verified=False)
    (out/'RESULT.json').write_text(json.dumps(report,indent=2,allow_nan=False));return {k:v for k,v in report.items() if k!='records'}
