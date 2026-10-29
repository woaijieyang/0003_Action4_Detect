"""Fixed-ID pose-associated metrics; scoring never changes inference outputs."""
import numpy as np
from data.action4_dataset import is_ignored

SCHEMA = 'action4.pose_projection_all8_002.v1'

def ratio(a,b):
    return float(a/b) if b else None

def bounds(points):
    return np.r_[points.min(0),points.max(0)]

from data.action4_instance_assignment import iou,match_geometry

def gt_groups(record,g,gs):
    """The original pose-score GT grouping, also shared by calibration provenance."""
    b=np.asarray(g['bbox_full_xyxy']);small=(b[2]-b[0])/record['width']<.08;heavy=g.get('occlusion_inframe',0)>=.4
    labels=['all','small' if small else 'non_small','high_occlusion' if heavy else 'lower_occlusion']
    if small and heavy:labels.append('small_high_occlusion')
    labels.append('scene:'+str(record.get('scene_id',record.get('scene_family','unknown'))))
    transform=g.get('canonical8_geometry',{}).get('object_to_capture_cv_m')
    if transform is not None:
        z=float(np.asarray(transform)[2,3]);labels.append('depth:'+('lt_0p6' if z<.6 else '0p6_1p0' if z<1. else 'ge_1p0'))
    if len(gs)==2:
        a,c=[np.asarray(x.get('bbox_visible_xyxy',x['bbox_full_xyxy'])) for x in gs]
        gap=np.linalg.norm(np.maximum(0,np.maximum(a[:2]-c[2:],c[:2]-a[2:])))
        near=gap<=.25*min(np.min(a[2:]-a[:2]),np.min(c[2:]-c[:2]))
        widths=[x['bbox_full_xyxy'][2]-x['bbox_full_xyxy'][0] for x in gs]
        unequal=max(widths)/min(widths)>=1.5
        labels.extend(['near_pair' if near else 'separated_pair','unequal_size' if unequal else 'similar_size'])
        if near and unequal and small and heavy:labels.append('small_unequal_near_heavy')
    return labels

def score_pose(predictions,records):
    mapping={p['sample_id']:p['instances'] for p in predictions}
    if len(mapping)!=len(predictions) or set(mapping)!={r['sample_id'] for r in records}:
        raise ValueError('Exactly one prediction record for every input image required')
    counts={'gt':0,'published':0,'matched':0,'invalid':0,'tp002':0,'tp005':0,'strict':0}
    rows=[];unmatched=[];errors=[];groups={}
    for record in records:
        sid=record['sample_id'];gs=[g for g in record['instances'] if not is_ignored(g)]
        ps=[p for p in mapping[sid] if p.get('published',False)]
        ids=[p['prediction_id'] for p in ps]
        if len(set(ids))!=len(ids):raise ValueError('Duplicate prediction identity')
        gtpoints=[np.asarray(g['corners_px'],dtype=float) for g in gs]
        if any(p.shape!=(8,2) or not np.isfinite(p).all() for p in gtpoints):raise ValueError('Invalid GT corners')
        valid=[];points=[];invalid=[]
        for j,p in enumerate(ps):
            uv=np.asarray(p.get('corners_px'),dtype=float)
            ok=p.get('pose_valid') is True and p.get('positive_depth_verified') is True and p.get('corner_ids')==list(range(8))
            ok=ok and uv.shape==(8,2) and np.isfinite(uv).all()
            if ok:valid.append(j);points.append(uv)
            else:invalid.append(j)
        pairs=match_geometry([bounds(p) for p in points],[bounds(p) for p in gtpoints],
                             [ids[i] for i in valid],[g['instance_id'] for g in gs])
        bygt={g:valid[p] for p,g in pairs};used=set(bygt.values());image_success=0
        counts['gt']+=len(gs);counts['published']+=len(ps);counts['matched']+=len(pairs);counts['invalid']+=len(invalid)
        for j,g in enumerate(gs):
            pi=bygt.get(j);err=None;success=False;success5=False
            if pi is not None:
                err=np.linalg.norm(np.asarray(ps[pi]['corners_px'])-gtpoints[j],axis=1)
                box=bounds(gtpoints[j]);diag=np.linalg.norm(box[2:]-box[:2])
                if diag<=0:raise ValueError('Zero GT diagonal')
                success=bool(err.max()<=.02*diag);success5=bool(err.max()<=.05*diag);errors.extend(err.tolist())
            counts['tp002']+=success;counts['tp005']+=success5;image_success+=success
            row={'sample_id':sid,'instance_id':g['instance_id'],'prediction_id':ids[pi] if pi is not None else None,
                 'errors_px':None if err is None else err.tolist(),'all8_at_002':success,'all8_at_005':success5,
                 'failure_reason':None if success else 'corner_error' if pi is not None else 'no_valid_pose_match'}
            rows.append(row)
            for label in gt_groups(record,g,gs):
                z=groups.setdefault(label,{'gt':0,'tp002':0});z['gt']+=1;z['tp002']+=success
        counts['strict']+=int(image_success==len(gs) and len(ps)==len(gs))
        unmatched.extend({'sample_id':sid,'prediction_id':ids[j],'reason':'invalid_pose' if j in invalid else 'unmatched_pose'} for j in range(len(ps)) if j not in used)
    n,p,t=counts['gt'],counts['published'],counts['tp002']
    for g in groups.values():g['recall']=ratio(g['tp002'],g['gt'])
    return {'score_schema':SCHEMA,'pose_gt_count':n,'pose_output_count':p,
            'pose_all8_at_002_success':t,'pose_all8_at_002_rate':ratio(t,n),
            'pose_all8_at_005_rate':ratio(counts['tp005'],n),'pose_all8_at_002_precision':ratio(t,p),
            'pose_all8_at_002_f1':ratio(2*t,n+p),'pose_fp':p-t,'pose_fn':n-t,
            'invalid_pose_count':counts['invalid'],'pose_instance_precision':ratio(counts['matched'],p),
            'pose_corner_p95_valid_matched_only':float(np.percentile(errors,95)) if errors else None,
            'strict_image_success_rate':ratio(counts['strict'],len(records)),
            'per_gt_failure_reason':rows,'unmatched_pose_predictions':unmatched,'pose_groups':groups}

def eligible(report,reference):
    if report.get('score_schema')!=SCHEMA or reference.get('score_schema')!=SCHEMA:raise ValueError('Mixed pose schemas')
    for key in ['pose_instance_precision','pose_all8_at_002_precision']:
        old=reference[key];new=report[key]
        if old is not None and (new is None or new+1e-12<old):return False
    for key,old in reference['pose_groups'].items():
        if old['gt'] and report['pose_groups'][key]['recall']+1e-12<old['recall']:return False
    return True

def rank(report,step):
    if report.get('score_schema')!=SCHEMA:raise ValueError('Wrong pose score schema')
    value=report['pose_all8_at_002_rate'];p95=report['pose_corner_p95_valid_matched_only']
    return (value if value is not None else -1.,-(report['pose_fn']+report['pose_fp']),
            -p95 if p95 is not None else -float('inf'),-step)
