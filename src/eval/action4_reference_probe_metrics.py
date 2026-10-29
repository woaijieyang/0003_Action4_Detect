"""Train-only fixed reference-cache diagnostic, not nativeYOLO or detector AP.

Tip-Adapter-style cache logits on existing frozen128D appearance. No model load.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from data.action4_paths import PROJECT_ROOT as R,contained_path


def sha(p):
    with contained_path(R,p,exists=True).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def metrics(scores,labels,threshold):
    scores=np.asarray(scores,float);labels=np.asarray(labels)
    if scores.shape!=labels.shape or not np.isfinite(scores).all() or not np.isin(labels,[0,1]).all():raise ValueError('Explicit eligible binary domain required')
    p=int((labels==1).sum());n=int((labels==0).sum());pred=scores>=threshold
    tp=int((pred&(labels==1)).sum());fp=int((pred&(labels==0)).sum());fn=p-tp
    precision=tp/(tp+fp) if tp+fp else None;recall=tp/p if p else None
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
    return {'count':len(labels),'positive':p,'negative':n,'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'F1':f1}


def auc(scores,labels):
    labels=np.asarray(labels);p=int((labels==1).sum());n=int((labels==0).sum())
    if not p or not n:return None
    ranks=rankdata(scores,method='average');return float((ranks[labels==1].sum()-p*(p+1)/2)/(p*n))


def normalize(x):
    x=np.asarray(x,np.float32)
    if x.ndim!=2 or x.shape[1]!=128 or not np.isfinite(x).all():raise ValueError('Finite128D appearance required')
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def bank_candidates(manifest):
    candidates=[]
    for row in manifest['records']:
        if row['split']!='train':continue
        if sha(row['file'])!=row['file_sha256']:raise ValueError('Train feature bytes changed')
        with np.load(contained_path(R,row['file'],exists=True),allow_pickle=False) as data:
            f=data['appearance'];y=data['labels'];g=data['target'];center=data['center_score'];grid=data['grid']
            valid=np.linalg.norm(f,axis=1)>1e-12
            indices=[]
            # At most one clean positive per training GT, one hard known
            # non-target per image. GT never creates eval predictions.
            for target in sorted(set(g[y==1].tolist())):
                hits=np.flatnonzero((y==1)&(g==target)&valid)
                if len(hits):indices.append(int(hits[np.argmax(center[hits])]))
            hits=np.flatnonzero((y==0)&valid)
            if len(hits):indices.append(int(hits[np.argmax(center[hits])]))
            for i in indices:candidates.append({'image_id':row['image_id'],'scene':row['scene'],'grid':int(grid[i]),'label':int(y[i]),'feature':f[i].copy()})
    return candidates


def image_domain_counts(row,labels):
    labels=np.asarray(labels)
    if labels.ndim!=1 or not np.isin(labels,[-1,0,1]).all():raise ValueError('Unknown proposal label domain')
    counts={'image_id':row['image_id'],'scene':row['scene'],'all_candidates':len(labels),
        'positive':int((labels==1).sum()),'negative':int((labels==0).sum()),'ignored_or_ambiguous':int((labels<0).sum()),'GT_count':row['GT_count']}
    counts['eligible']=counts['positive']+counts['negative']
    if counts['all_candidates']!=counts['eligible']+counts['ignored_or_ambiguous']:raise ValueError('Domain mass mismatch')
    for target,original in [('all_candidates','candidates'),('positive','positive'),('negative','negative'),('ignored_or_ambiguous','ignored_or_ambiguous')]:
        if counts[target]!=row[original]:raise ValueError('Feature manifest label counts differ')
    return counts


def score(root,output):
    root=contained_path(R,root,exists=True);out=contained_path(R,output,exists=False)
    if out.exists():raise FileExistsError('Preserve prior reference-probe scoring')
    manifest=json.loads((root/'feature_run/FEATURE_MANIFEST.json').read_text());cfg=json.loads((root/'CONFIG.json').read_text());pre=json.loads((root/'PREREGISTRATION.json').read_text())
    if manifest['status']!='FROZEN_FEATURE_EXTRACTION_COMPLETE_NOT_ADAPTER_TRAINING' or manifest['config_sha256']!=sha(root/'CONFIG.json'):raise ValueError('Incomplete/mismatched frozen extraction')
    rows=manifest['records'];train=[r for r in rows if r['split']=='train'];ev=[r for r in rows if r['split']=='eval']
    if len(train)!=192 or len(ev)!=650 or len({r['image_id'] for r in rows})!=842:raise ValueError('Full unique split domains required')
    rng=np.random.default_rng(cfg['seed']);available=bank_candidates(manifest);chosen=[];k=pre['reference_bank_per_class']
    for label in (0,1):
        pool=[x for x in available if x['label']==label]
        if len(pool)<k:raise ValueError('Insufficient eligible train references; do not duplicate or use eval')
        chosen.extend(pool[i] for i in rng.choice(len(pool),k,replace=False))
    keys=normalize(np.stack([x['feature'] for x in chosen]));values=np.eye(2,dtype=np.float32)[[x['label'] for x in chosen]]
    shuffled=values[rng.permutation(len(values))];features=[];labels=[];base=[];image_rows=[];full_domains=[]
    for row in ev:
        if sha(row['file'])!=row['file_sha256']:raise ValueError('Eval feature bytes changed')
        with np.load(contained_path(R,row['file'],exists=True),allow_pickle=False) as data:
            y=data['labels'];full_domains.append(image_domain_counts(row,y));keep=y>=0;image_rows.extend([row['image_id']]*int(keep.sum()));features.append(data['appearance'][keep]);labels.append(y[keep]);base.append(data['center_score'][keep])
    q=normalize(np.concatenate(features));y=np.concatenate(labels);baseline=np.concatenate(base).astype(np.float64);base_logit=np.log(np.clip(baseline,1e-6,1-1e-6)/np.clip(1-baseline,1e-6,1))
    affinity=q@keys.T;all_reports=[];score_cache={}
    for mode,V in [('true_train_labels',values),('shuffled_train_labels',shuffled)]:
        for beta in pre['beta_grid']:
            cache=np.exp(beta*(affinity-1))@V
            for alpha in pre['alpha_grid']:
                if alpha==0 and (beta!=pre['beta_grid'][0] or mode!='true_train_labels'):continue
                logit=base_logit+alpha*(cache[:,1]-cache[:,0]);prob=baseline.copy() if alpha==0 else 1/(1+np.exp(-np.clip(logit,-80,80)));key=(mode,beta,alpha);score_cache[key]=prob
                roc=auc(prob,y)
                for threshold in pre['threshold_grid']:all_reports.append(dict(metrics(prob,y,threshold),mode=mode,beta=beta,alpha=alpha,threshold=threshold,AUC=roc))
    def rank(r):return (r['F1'] if r['F1'] is not None else -1,-r['alpha'],-r['beta'],-r['threshold'])
    candidate=max([r for r in all_reports if r['mode']=='true_train_labels'],key=rank)
    baseline_best=max([r for r in all_reports if r['alpha']==0],key=rank)
    shuffled_best=max([r for r in all_reports if r['mode']=='shuffled_train_labels' or r['alpha']==0],key=rank)
    group_scores={};group_domains={};ids=np.asarray(image_rows)
    for scene in sorted({r['scene'] for r in ev}):
        selected=np.isin(ids,[r['image_id'] for r in ev if r['scene']==scene]);prob=score_cache[(candidate['mode'],candidate['beta'],candidate['alpha'])]
        group_scores[scene]=metrics(prob[selected],y[selected],candidate['threshold'])
        members=[d for d in full_domains if d['scene']==scene];group_domains[scene]={'images':len(members),**{k:sum(d[k] for d in members) for k in ('all_candidates','eligible','positive','negative','ignored_or_ambiguous','GT_count')}}
    out.mkdir(parents=True);np.savez_compressed(out/'TRAIN_REFERENCE_BANK.npz',keys=keys,values=values,shuffled_values=shuffled)
    members=[{k:v for k,v in x.items() if k!='feature'} for x in chosen]
    (out/'BANK_MEMBERS.json').write_text(json.dumps(members,indent=2)+'\n')
    report={'status':'DEVELOPMENT_PROPOSAL_REFERENCE_DIAGNOSTIC_COMPLETE_NOT_DETECTOR_ACCEPTANCE','feature_manifest_sha256':sha(root/'feature_run/FEATURE_MANIFEST.json'),'preregistration_sha256':sha(root/'PREREGISTRATION.json'),'metrics_source_sha256':sha(__file__),
        'reference_bank_sha256':sha(out/'TRAIN_REFERENCE_BANK.npz'),'bank_members_sha256':sha(out/'BANK_MEMBERS.json'),
        'reference_origin':'train_only','reference_members':len(chosen),'train_images':192,'eval_images':650,'eligible_eval_proposals':len(y),
        'ignored_or_ambiguous_eval_proposals':sum(r['ignored_or_ambiguous'] for r in ev),'all_eval_GT_instances':sum(r['GT_count'] for r in ev),
        'baseline_best_dev_threshold':baseline_best,'fixed_reference_best_dev':candidate,'label_shuffled_negative_control_best_dev':shuffled_best,
        'per_image_full_domains':full_domains,'per_scene_full_domains':group_domains,'per_scene_selected_config':group_scores,'all_finite_configs':all_reports,'selection':'full development F1, then loweralpha/lowerbeta/lowerthreshold; no test','negative_control_allows_shared_alpha0_baseline':True,'nativeYOLO99_changed':False,'full_detection_AP_or_dual_recall_measured':False,'real_video_gain_measured':False,
        'label_domain':'pure single-target candidates vs known non-target; mixed and invisible remain unmeasured, not deleted from original detection GT','learned_adapter_trained':False}
    (out/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['all_finite_configs','per_scene_selected_config','per_image_full_domains']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();score(a.root,a.output)
