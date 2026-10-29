"""Preregistered64 training-recipe input forwards; no optimizer, backward or promotion."""
import copy,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path

def run():
    import numpy as np
    import torch
    from data.action4_paths import PROJECT_ROOT as ROOT,contained_path
    from eval.action4_roi_sweep import binding,read_binding
    from data.action4_stage2_data import Stage2Units
    from model.action4_sam_feature_adapter import FeatureIdentity,MaskAux
    from model.action4_sam_joint import stack_sam_items
    from model.training.action4_corner_targets import corner_targets_from_items
    from model.action4_corner_supervision import corner_loss_sums
    from model.action4_release_bundle import load_bundle
    out=contained_path(ROOT,'outputs/action4_corner_train_quality_probe_v001',exists=True)
    def write(name,value):
        with (out/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    def versions(module):
        return {kind+':'+name:{'version':value._version,'shape':list(value.shape),'dtype':str(value.dtype),'requires_grad':value.requires_grad} for kind,values in [('parameter',module.named_parameters()),('buffer',module.named_buffers())] for name,value in values}
    def tensor_ref(tensor):
        value=tensor.detach().cpu().contiguous().numpy()
        return {'shape':list(value.shape),'dtype':str(value.dtype),'sha256':hashlib.sha256(value.tobytes()).hexdigest()}
    if set(os.sched_getaffinity(0))!={60,61,62,63}:raise ValueError('CPU60-63 required')
    fresh=subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=uuid,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    gpu=fresh[0].strip()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu or int(fresh[1])<12288:raise ValueError('Fresh UUID/headroom mismatch')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    frozen=contained_path(ROOT,'runtime/action4_runs/p2_a0_corner_prediction15_formal80_v003/source',exists=True)
    cfgref=binding(frozen/'resolved_config.json');fullcfg=read_binding(cfgref);cfg=copy.deepcopy(fullcfg)
    indices=np.random.default_rng(20261028).choice(10400,size=64,replace=False).tolist()
    cfg.update(mode='integration',train_indices=indices,sam_cache_validation_workers=2)
    changed={k for k in set(cfg)|set(fullcfg) if cfg.get(k)!=fullcfg.get(k)}
    if changed!={'mode','train_indices','sam_cache_validation_workers'}:raise ValueError('Unexpected diagnostic recipe changes')
    modelref={'path':'models/candidates/action4_improvement/stage4_all8_trial_20261028/training_snapshot.pt','sha256':'5596d99347f5ebf65745bffb91fb82b2523080df6dd675a4a656cb404a955011'}
    manifest=read_binding(binding(frozen/'CORNER_EVAL_SOURCE.json'));verified={}
    for rel in ['src/data/action4_stage2_data.py','src/model/action4_release_bundle.py','src/model/action4_sam_joint.py','src/model/training/action4_corner_targets.py','src/model/action4_corner_supervision.py','src/data/action4_sam_cache_provider.py']:
        p=frozen/rel;sha=hashlib.sha256(p.read_bytes()).hexdigest()
        if sha!=manifest['files_sha256'][rel]:raise ValueError('Frozen source changed '+rel)
        verified[rel]=sha
    write('PREREGISTRATION.json',{'schema':'action4.sampled_train_quality_probe.v1','seed':20261028,'population_units':10400,'sample_size':64,'without_replacement':True,'indices_in_draw_order':indices,'index_min':min(indices),'index_max':max(indices),'dataset_epoch_argument':29,'microbatch':2,'full_config':cfgref,'changed_fields':sorted(changed),'diagnostic_only_integration_config':True,'actual_integration_acceptance_claim':False,'model':modelref,'source_manifest':binding(frozen/'CORNER_EVAL_SOURCE.json'),'verified_sources':verified,'gpu':{'physical':4,'uuid':gpu,'free_MiB':int(fresh[1]),'total_MiB':int(fresh[2]),'utilization_percent':int(fresh[3]),'allocator_cap_GiB':4},'CPU_affinity':sorted(os.sched_getaffinity(0)),'torch_threads':2,'cache_validation_workers':2,'optimizer':False,'backward':False,'additional_augmentation_ablation':False})
    write('DIAGNOSTIC_CONFIG.json',cfg)
    print(json.dumps({'stage':'data_validation','pid':os.getpid(),'indices':indices}),flush=True)
    started=time.monotonic();data=Stage2Units(cfg,sam_factories=(FeatureIdentity,MaskAux))
    if len(data)!=64 or len(data.data)!=10400 or data.indices!=indices:raise ValueError('Wrong population/subset domain')
    print(json.dumps({'stage':'strict_load','seconds':time.monotonic()-started}),flush=True)
    model,base,state=load_bundle(modelref['path'],'A0','cuda:0',allow_corner_candidate=True,expected_sha256=modelref['sha256'])
    if state['selected_step']!=6375 or state['quality_ready'] is not False or state['deployment_admitted'] is not False:raise ValueError('Wrong model state')
    model.eval().requires_grad_(False);base.eval().requires_grad_(False)
    before={'model':versions(model),'base':versions(base)};write('VERSIONS_BEFORE.json',before)
    write('MODEL_IDENTITY.json',{'model':modelref,'selected_step':state['selected_step'],'config_id':state['config_id'],'architecture':state['architecture'],'pose_method':state['pose_method'],'strict_load':True,'model_eval':not model.training,'base_eval':not base.training,'source_loader':str(Path(load_bundle.__code__.co_filename).relative_to(ROOT))})
    del state
    outputs=[];inputs=[];shape_records=[];augmentation=[];bce_sum=0.;eligible_count=0
    with torch.inference_mode():
        for first in range(0,64,2):
            items=[data[(i,29,True,64)] for i in range(first,min(first+2,64))]
            rois=torch.stack([x['roi'] for x in items]).to('cuda:0');K=torch.stack([x['geometry']['K'] for x in items]).to('cuda:0');cad=torch.stack([x['geometry']['canonical'] for x in items]).to('cuda:0')
            aux,identities,unused_gt,unused_supervised=stack_sam_items(items,'cuda:0',rois.dtype)
            # Only RGB, current predicted SAM inputs, intrinsics, fixed CAD and identities enter forward.
            packet=model(rois,K,cad,mask_aux=aux,identities=identities,return_corner=True)
            targets=corner_targets_from_items(items,device='cuda:0',dtype=rois.dtype)
            stats=corner_loss_sums(packet.corners,targets,epoch=29)
            err=torch.linalg.vector_norm(packet.corners.uv224-targets.uv224,dim=-1)
            actual=(targets.valid&(err<=.01*targets.diagonal224[:,None])).to(rois.dtype)
            if not torch.equal(stats.quality_target,actual) or not torch.equal(stats.quality_eligible,targets.known):raise ValueError('Actual quality BCE target mismatch')
            if packet.corners.quality_logits.shape!=(len(items),8):raise ValueError('Quality logits shape mismatch')
            bce_sum+=float(stats.sums['corner_quality_bce'].item());eligible_count+=int(stats.counts['corner_quality_bce'])
            q=packet.corners.q_corner.cpu().numpy();pv=packet.corners.p_valid.cpu().numpy()
            shape_records.append({'first_unit':first,'RGB':list(rois.shape),'quality_logits':list(packet.corners.quality_logits.shape),'quality_target':list(actual.shape),'quality_eligible':list(targets.known.shape),'quality_bce_count':stats.counts['corner_quality_bce']})
            for j,item in enumerate(items):
                ident=item['identity'];identity={k:getattr(ident,k) for k in ('image_id','instance_id','roi_id','pixel_space_id')};deps=item['sam_input']['dependencies'];aug=copy.deepcopy(deps.get('mask_augmentation',{}));augmentation.append(aug)
                manifest_id={'data_index':item['data_index'],'identity':identity,'augmentation_seed':item['augmentation_seed'],'RGB':tensor_ref(item['roi']),'K':tensor_ref(item['geometry']['K']),'fixed_CAD':tensor_ref(item['geometry']['canonical'])}
                for key in ('visible','distance','available','candidate_overlap','dropout'):
                    if hasattr(item['mask_aux'],key):manifest_id['mask_'+key]=tensor_ref(getattr(item['mask_aux'],key))
                manifest_id['manifest_id']=hashlib.sha256(json.dumps(manifest_id,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                manifest_id['sam_input']=copy.deepcopy(item['sam_input']);manifest_id['frozen_prediction_roi']=copy.deepcopy(item['metadata'].get('frozen_prediction_roi'));inputs.append(manifest_id)
                outputs.append({'data_index':item['data_index'],'input_manifest_id':manifest_id['manifest_id'],'identity':identity,'q_corner':q[j].tolist(),'p_valid':pv[j].tolist(),'uv224':packet.corners.uv224[j].cpu().tolist(),'error224':err[j].cpu().tolist(),'gt_diagonal224':float(targets.diagonal224[j]),'quality_target':actual[j].cpu().tolist(),'known':targets.known[j].cpu().tolist(),'valid':targets.valid[j].cpu().tolist(),'visible':item['geometry']['visible'].tolist(),'sam_dropout':bool(item['mask_aux'].dropout.reshape(-1)[0]),'sam_augmentation':aug})
            del packet,targets,stats,rois,K,cad,aux,unused_gt,unused_supervised,items
            print(json.dumps({'stage':'forward','units':first+2,'peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2}),flush=True)
    after={'model':versions(model),'base':versions(base)}
    if before!=after or model.training or base.training:raise ValueError('Model parameters/buffers versions or eval state changed')
    write('VERSIONS_AFTER.json',after);write('INPUT_TENSOR_MANIFEST.json',{'units':inputs});write('POINT_RECORDS.json',{'records':outputs});write('BCE_TENSOR_SHAPES.json',{'batches':shape_records,'actual_target_expression':'valid AND norm(uv224-target_uv224)<=0.01*diagonal224; detached; eligibility known','quality_bce_sum':bce_sum,'quality_bce_count':eligible_count,'quality_bce_mean':bce_sum/eligible_count if eligible_count else None})
    q=np.asarray([r['q_corner'] for r in outputs]);labels=np.asarray([r['quality_target'] for r in outputs],bool);known=np.asarray([r['known'] for r in outputs],bool);valid=np.asarray([r['valid'] for r in outputs],bool);visible=np.asarray([r['visible'] for r in outputs],bool);error=np.asarray([r['error224'] for r in outputs]);diag=np.asarray([r['gt_diagonal224'] for r in outputs])[:,None];selected=q>=.6
    def summary(mask):
        m=mask&known;n=int(m.sum())
        return {'points':n,'mean_q':float(q[m].mean()) if n else None,'correct_at_1pct':int((m&labels).sum()),'precision_at_1pct':float(labels[m].mean()) if n else None,'correct_at_2pct':int((m&valid&(error<=.02*diag)).sum()),'Brier':float(np.mean((q[m]-labels[m])**2)) if n else None}
    bins=[dict(lo=lo/10,hi=min((lo+1)/10,1),**summary((q>=lo/10)&(q<((lo+1)/10 if lo<9 else 1.000001)))) for lo in range(10)]
    devref=binding('outputs/action4_corner_quality_alignment_v001/REPORT.json');dev=read_binding(devref)
    result={'schema':'action4.sampled_train_quality_probe.v1','status':'64_TRAIN_RECIPE_UNITS_FORWARD_DIAGNOSTIC_COMPLETE_NOT_TRAIN_ACCEPTANCE','units':64,'points':512,'population_units':10400,'epoch_argument':29,'known_points':int(known.sum()),'valid_points':int(valid.sum()),'all_known':summary(known),'quality_selected':summary(selected),'selected_visible':summary(selected&visible),'selected_hidden':summary(selected&~visible),'all_visible':summary(visible),'all_hidden':summary(~visible),'bins':bins,'actual_sam_dropout_units':sum(r['sam_dropout'] for r in outputs),'augmentation_requested_dropout_units':sum(a.get('dropout',False) for a in augmentation),'augmentation_requested_perturb_units':sum(a.get('perturb',False) for a in augmentation),'augmentation_applied_units':sum(a.get('applied',False) for a in augmentation),'quality_BCE':{'sum':bce_sum,'eligible_known_points':eligible_count,'mean':bce_sum/eligible_count if eligible_count else None},'parameters_and_buffers_versions_unchanged':True,'peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2,'peak_reserved_MiB':torch.cuda.max_memory_reserved()/1024**2,'dev_comparison_source':devref,'full650_dev_stats':{k:dev[k] for k in ('all_known','quality_selected','selected_visible','selected_hidden','bins')},'caveat':'Preregistered64 sampled training units with original epoch29 augmentation versus full650 development predictions; no direct quantification of generalization gap. Forward eval mode, no optimizer/backward or model changes.','seconds':time.monotonic()-started}
    write('REPORT.json',result);print(json.dumps({'status':result['status'],'train_quality_selected':result['quality_selected'],'dev_quality_selected':dev['quality_selected'],'dropout':result['actual_sam_dropout_units'],'perturb':result['augmentation_requested_perturb_units']}),flush=True)
