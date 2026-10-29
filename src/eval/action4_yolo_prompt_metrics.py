"""Independent YOLO recognition, instance support and prompt scoring.

Recognition: frozen class mapping + GT bbox_full_xyxy / native prediction bbox
IoU>=0.5, max cardinality then total IoU then stable IDs. Masks never gate this
metric. Joint support/prompt metrics are separate and need visible GT masks.
"""
import hashlib
import json
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment

SCORE_SCHEMA='action4.yolo_prompt_metrics.v4'
JOINT_GATE_CONTRACT='visible_support_plus_prompt_joint_only_99'
METRIC_SEMANTICS={
    'recognition_precision_recall':'Independent diagnostic: class and full-GT-box IoU>=0.5; never an additional 99-percent veto.',
    'joint_precision_recall':'Class plus visible-mask support/purity/separation and prompt coverage/geometry; one-to-one matching. Native full-box IoU is not required.',
    'independent_dual_rate':'Preserved legacy support-only dual diagnostic; joint_dual_rate is the strict joint dual gate.',
    'joint_dual_rate':'Two distinct supports satisfy joint conditions with no extra published target outputs.',
    'strict_image_rate':'All evaluable targets satisfy joint conditions and no extra target outputs; unknown/ignore/support-unavailable images remain unmeasured.',
    'joint_99_observed':'Joint precision, joint recall, strict image and strict joint dual rates >=0.99; required groups and unknown-evidence restrictions retained.',
    'prediction_diagnostics':'Evidence labels, not inferred brand labels. No visible-GT overlap means background_or_unlabeled, not a proven non-DJI class.'}

def _ratio(n,d):return float(n/d) if d else None
def _area(b):return max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
def _valid_box(b):
    if b is None:return False
    b=np.asarray(b,float)
    return b.shape==(4,) and np.isfinite(b).all() and _area(b)>0
def _box(m):
    y,x=np.where(m);return np.array([x.min()-.5,y.min()-.5,x.max()+.5,y.max()+.5])
def _iou(a,b):
    c=[max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])]
    x=_area(c);return x/(_area(a)+_area(b)-x)
def _match(edges,quality,targets,preds):
    if not edges.size:return []
    bonus=5*min(edges.shape)+1
    rows,cols=linear_sum_assignment(np.where(edges,bonus+quality,-bonus),maximize=True)
    return [(targets[i]['id'],preds[j]['id']) for i,j in zip(rows,cols) if edges[i,j]]

def score_image(predictions,targets,*,image_id,class_protocol,groups=(),source_group=None,independent_unit_id=None):
    """Require an explicit frozen class protocol, never caller class_correct.

    protocol={protocol_id,prediction_class_map:{class_id:GT class name},
    target_classes:[...], frozen:true}. Its canonical SHA is recorded.
    GT requires annotation_status='valid'/'unknown'/'ignore', class_name and
    legal bbox_full_xyxy. Missing/invalid full boxes are unmeasured, not inferred
    from visible masks. All-occluded targets with legal boxes remain recognition
    GT. Masks may be missing/empty, making support unavailable rather than a TN.
    """
    if not isinstance(class_protocol,dict) or class_protocol.get('frozen') is not True or not class_protocol.get('protocol_id'):
        raise ValueError('Explicit frozen class protocol required')
    mapping={str(k):v for k,v in class_protocol['prediction_class_map'].items()}
    classes=set(class_protocol['target_classes'])
    if not mapping or not classes or not all(isinstance(v,str) for v in mapping.values()):raise ValueError('Invalid class mapping')
    identity=hashlib.sha256(json.dumps(class_protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    ps=sorted(predictions,key=lambda p:str(p['id']));ts=sorted(targets,key=lambda t:str(t['id']))
    for rows in (ps,ts):
        if len({str(r['id']) for r in rows})!=len(rows):raise ValueError('Duplicate IDs')
    for p in ps:
        if p.get('source')!='yolo':raise ValueError('Only YOLO outputs permitted')
        if 'class_correct' in p:raise ValueError('class_correct is forbidden; use frozen class mapping')
        if str(p.get('class_id')) not in mapping:raise ValueError('Unknown prediction class ID')
        if not _valid_box(p.get('bbox')) or not _valid_box(p.get('prompt_bbox')):raise ValueError('Invalid prediction boxes')
    ignored=[t['id'] for t in ts if t.get('annotation_status')=='ignore']
    unknown=[t['id'] for t in ts if t.get('annotation_status') not in ('valid','ignore') or
             (t.get('annotation_status')=='valid' and (not isinstance(t.get('class_name'),str) or not _valid_box(t.get('bbox_full_xyxy'))))]
    valid=[t for t in ts if t['id'] not in ignored+unknown]
    eligible=[t for t in valid if t['class_name'] in classes]
    # Published target set is defined by the protocol, retaining negative outputs separately.
    negative_predictions=[p['id'] for p in ps if mapping[str(p['class_id'])] not in classes]
    ps=[p for p in ps if mapping[str(p['class_id'])] in classes]
    shapes=set()
    for row in valid+ps:
        m=row.get('mask')
        if m is not None:
            m=np.asarray(m)
            if m.dtype!=np.bool_ or m.ndim!=2:raise ValueError('Native bool masks required')
            shapes.add(m.shape)
    if len(shapes)>1:raise ValueError('Native mask shape mismatch')
    visible=[t for t in eligible if t.get('mask') is not None and np.any(t['mask'])]
    all_visible=[t for t in valid if t.get('mask') is not None and np.any(t['mask'])]
    unavailable=[t['id'] for t in eligible if t.get('mask') is None or not np.any(t['mask'])]
    duplicate=set();seen={};duplicate_groups={}
    for j,p in enumerate(ps):
        if p.get('mask') is not None:
            key=np.asarray(p['mask']).tobytes()
            if key in seen:
                duplicate.add(j);duplicate_groups[seen[key]].append(p['id'])
            else:
                seen[key]=j;duplicate_groups[j]=[p['id']]
    rec=np.zeros((len(eligible),len(ps)),bool);rq=np.zeros(rec.shape)
    for i,t in enumerate(eligible):
        for j,p in enumerate(ps):
            rq[i,j]=_iou(t['bbox_full_xyxy'],p['bbox'])
            rec[i,j]=mapping[str(p['class_id'])]==t['class_name'] and rq[i,j]>=.5
    support=np.zeros((len(visible),len(ps)),bool);prompt=support.copy();joint=support.copy();quality=np.zeros(support.shape);pairs=[]
    for i,t in enumerate(visible):
        v=t['mask'];vb=_box(v);n=int(v.sum());yy,xx=np.where(v)
        for j,p in enumerate(ps):
            b=np.asarray(p['prompt_bbox'],float)
            coverage=float(((xx>=b[0])&(xx<b[2])&(yy>=b[1])&(yy<b[3])).mean())
            center=(b[:2]+b[2:])/2;pad=(vb[2:]-vb[:2])*.1
            box_ok=coverage>=.95 and _area(b)<=4*_area(vb) and bool(np.all(center>=vb[:2]-pad)&np.all(center<=vb[2:]+pad))
            s=p.get('mask');own=purity=leak=fraction=None;ok=False
            if s is not None and np.any(s):
                hits=[int((s&u['mask']).sum()) for u in all_visible];hit=int((s&v).sum())
                own=hit/n;purity=_ratio(hit,sum(hits));fraction=hit/int(s.sum())
                leak=max([hits[k]/int(u['mask'].sum()) for k,u in enumerate(all_visible) if u['id']!=t['id']],default=0.)
                ok=own>=.5 and purity is not None and purity>=.95 and leak<=.05 and fraction>=.8
            correct=mapping[str(p['class_id'])]==t['class_name']
            support[i,j]=bool(correct and ok)
            prompt[i,j]=bool(correct and box_ok)
            joint[i,j]=support[i,j] and prompt[i,j]
            quality[i,j]=(own or 0)+(purity or 0)+(fraction or 0)+coverage
            pairs.append(dict(target_id=t['id'],prediction_id=p['id'],coverage=coverage,own=own,purity=purity,leakage=leak,target_fraction=fraction,support_ok=bool(support[i,j]),prompt_ok=bool(prompt[i,j]),joint_ok=bool(joint[i,j])))
    rm=_match(rec,rq,eligible,ps);sm=_match(support,quality,visible,ps);pm=_match(prompt,quality,visible,ps);jm=_match(joint,quality,visible,ps)
    ng=len(eligible);npred=len(ps);known=not unknown and not ignored
    recognition_strict=(len(rm)==ng and npred==ng) if known else None
    strict=(len(jm)==ng and npred==ng) if known and not unavailable else None
    dual=ng==2 and known
    result=dict(score_schema=SCORE_SCHEMA,metric_semantics=dict(METRIC_SEMANTICS),image_id=image_id,groups=list(groups),source_group=source_group,independent_unit_id=independent_unit_id,
        class_protocol_sha256=identity,class_protocol_id=class_protocol['protocol_id'],gt_count=ng,prediction_count=npred,
        recognition_gt_ids=[t['id'] for t in eligible],recognition_unmeasured_gt_ids=unknown,ignored_gt_ids=ignored,
        support_gt_count=len(visible),support_unavailable_gt_ids=unavailable,unobservable_gt_ids=unavailable,
        missing_support_count=sum(p.get('mask') is None for p in ps),negative_prediction_ids=negative_predictions,
        duplicate_support_ids=[ps[j]['id'] for j in sorted(duplicate)],
        recognition_tp=len(rm),support_tp=len(sm),joint_tp=len(jm),prompt_coverage_tp=len(pm),
        recognition_matches=rm,support_matches=sm,joint_matches=jm,
        recognition_fp=npred-len(rm),recognition_fn=ng-len(rm),joint_fp=npred-len(jm),joint_fn=ng-len(jm),
        recognition_strict_image_success=recognition_strict,strict_image_success=strict,
        dual_eligible=dual,recognition_dual_success=recognition_strict if dual else None,
        independent_dual_success=(len(sm)==2 and npred==2) if dual and not unavailable else None,
        dual_success=strict if dual else None,pairs=pairs)
    rec_ids={p for _,p in rm};support_ids={p for _,p in sm};joint_ids={p for _,p in jm}
    diagnostics=[]
    for p in ps:
        pid=p['id'];mask=p.get('mask');overlaps=[]
        if known and mask is not None:
            overlaps=[dict(target_id=t['id'],class_name=t['class_name'],intersection_pixels=int((mask&t['mask']).sum()),visible_target_fraction=float((mask&t['mask']).sum()/t['mask'].sum())) for t in all_visible]
        merged=[x for x in overlaps if x['visible_target_fraction']>=.5]
        pair_rows=[x for x in pairs if x['prediction_id']==pid]
        localization_mismatch=known and pid in support_ids and pid not in rec_ids
        if not known:category='unmeasured_annotation_domain'
        elif localization_mismatch:category='full_box_localization_mismatch'
        elif pid in joint_ids:category='joint_supported_target'
        elif len(merged)>=2:category='merged_support'
        elif any(x['support_ok'] for x in pair_rows) and pid not in support_ids:category='duplicate_target_support'
        elif pid in support_ids:category='prompt_geometry_failure'
        elif unavailable:category='support_unmeasured'
        elif mask is None:category='missing_instance_support'
        elif not mask.any():category='empty_instance_support'
        elif not any(x['intersection_pixels'] for x in overlaps):category='background_or_unlabeled'
        elif not any(x['intersection_pixels'] and x['class_name'] in classes for x in overlaps):category='overlap_annotated_non_target'
        else:category='insufficient_or_contaminated_target_support'
        diagnostics.append(dict(prediction_id=pid,diagnostic_category=category,recognition_matched=pid in rec_ids if known else None,
            support_matched=pid in support_ids if known else None,joint_matched=pid in joint_ids if known else None,
            full_box_localization_mismatch=localization_mismatch if known else None,
            duplicate_support_group=next((g for g in duplicate_groups.values() if len(g)>1 and pid in g),[]),
            merged_support_evidence=merged if len(merged)>=2 else [],visible_gt_overlap_evidence=overlaps,
            non_dji_class_asserted=False))
    result.update(prediction_diagnostics=diagnostics,duplicate_support_groups=[g for g in duplicate_groups.values() if len(g)>1],
        unmatched_recognition_prediction_ids=[p['id'] for p in ps if p['id'] not in rec_ids] if known else None,
        unmatched_support_prediction_ids=[p['id'] for p in ps if p['id'] not in support_ids] if known else None,
        unmatched_joint_prediction_ids=[p['id'] for p in ps if p['id'] not in joint_ids] if known else None,
        unmatched_joint_gt_ids=[t['id'] for t in eligible if t['id'] not in {x for x,_ in jm}] if known else None)
    # Conservative graph-level eligibility: unknown/ignore annotation domains
    # cannot classify any published prediction as false positive or negative.
    result.update(scoring_eligible=known,input_prediction_count=len(predictions),
        non_target_prediction_count=len(negative_predictions),published_prediction_count=npred,
        scorable_prediction_count=npred if known else 0,
        excluded_prediction_count=0 if known else npred,
        scorable_gt_count=ng if known else 0,excluded_known_gt_count=0 if known else ng,
        unmeasured_reason=None if known else 'UNKNOWN_OR_IGNORE_ANNOTATION_DOMAIN')
    if not known:
        result['diagnostic_matches_not_scored']=dict(recognition=rm,support=sm,joint=jm)
        for key in ('recognition_tp','support_tp','joint_tp','prompt_coverage_tp',
                    'recognition_fp','recognition_fn','joint_fp','joint_fn',
                    'recognition_matches','support_matches','joint_matches'):
            result[key]=None
    return result

def summarize(records,*,required_groups=('wrist_interference','other_camera','near_pair','small_target'),unit_registry=None):
    """Observed gate and independent-unit uncertainty are explicitly separate.

    unit_registry optionally declares protocol_id, frozen, and units mapping
    unit_id -> {source_group,image_ids}. Every supplied record must occur exactly
    once; IDs must agree with record labels. Unit outcome is ALL images strict.
    This measures unit-strict outcomes, not an independent-instance recall bound.
    """
    if any(r.get('score_schema')!=SCORE_SCHEMA for r in records):raise ValueError('Legacy or mixed score records require explicit rescoring; historical v3 must remain unchanged')
    ids=[r['image_id'] for r in records]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate image IDs')
    if len({r['class_protocol_sha256'] for r in records})>1:raise ValueError('Mixed class protocols')
    def aggregate(rs):
        scored=[r for r in rs if r['scoring_eligible']]
        ng=sum(r['gt_count'] for r in scored);npred=sum(r['prediction_count'] for r in scored)
        published=sum(r['published_prediction_count'] for r in rs)
        known_gt=sum(r['gt_count'] for r in rs)
        rt=sum(r['recognition_tp'] for r in scored);jt=sum(r['joint_tp'] for r in scored);st=sum(r['support_tp'] for r in scored)
        nvisible=sum(r['support_gt_count'] for r in scored);pc=sum(r['prompt_coverage_tp'] for r in scored)
        def outcome(key,subset):
            vals=[r[key] for r in subset];known=[v for v in vals if v is not None]
            return dict(successes=sum(known),trials=len(known),rate=_ratio(sum(known),len(known)),unmeasured=len(vals)-len(known))
        strict=outcome('strict_image_success',rs);recstrict=outcome('recognition_strict_image_success',rs)
        duals=[r for r in rs if r['dual_eligible']];dual=outcome('independent_dual_success',duals);rd=outcome('recognition_dual_success',duals);jd=outcome('dual_success',duals)
        return dict(images=len(rs),gt_count=known_gt,prediction_count=published,published_prediction_count=published,
            input_prediction_count=sum(r['input_prediction_count'] for r in rs),
            non_target_prediction_count=sum(r['non_target_prediction_count'] for r in rs),
            scorable_prediction_count=npred,excluded_prediction_count=published-npred,
            scorable_gt_count=ng,excluded_known_gt_count=known_gt-ng,
            scored_images=len(scored),unmeasured_images=len(rs)-len(scored),support_gt_count=nvisible,
            recognition_tp=rt,support_tp=st,joint_tp=jt,recognition_precision=_ratio(rt,npred),recognition_recall=_ratio(rt,ng),
            joint_precision=_ratio(jt,npred),joint_recall=_ratio(jt,ng),prompt_coverage_tp=pc,prompt_coverage_rate=_ratio(pc,nvisible),
            support_purity_separation_rate=_ratio(st,nvisible),strict_successes=strict['successes'],strict_image_rate=strict['rate'],strict_image_outcome=strict,
            recognition_strict_image_outcome=recstrict,dual_images=len(duals),independent_dual_rate=dual['rate'],
            joint_dual_rate=jd['rate'],recognition_dual_rate=rd['rate'],recognition_dual_outcome=rd,independent_dual_outcome=dual,joint_dual_outcome=jd,
            missing_support_count=sum(r['missing_support_count'] for r in rs),unobservable_count=sum(len(r['support_unavailable_gt_ids']) for r in rs),
            unmeasured_gt_count=sum(len(r['recognition_unmeasured_gt_ids']) for r in rs),ignored_gt_count=sum(len(r['ignored_gt_ids']) for r in rs))
    out=aggregate(records);names=sorted(set(required_groups)|{g for r in records for g in r['groups']})
    out['groups']={g:aggregate([r for r in records if g in r['groups']]) for g in names}
    out['missing_groups']=[g for g in required_groups if not out['groups'][g]['images']]
    def passes(row):
        keys=['joint_precision','joint_recall','strict_image_rate']
        if row['dual_images']:keys+=['joint_dual_rate']
        return not(row['unobservable_count'] or row['unmeasured_gt_count'] or row['ignored_gt_count']) and all(row[k] is not None and row[k]>=.99 for k in keys)
    out['joint_99_observed']=bool(out['dual_images']) and not out['missing_groups'] and passes(out) and all(passes(out['groups'][g]) for g in required_groups)
    out['formal_99_pass']=False
    out['statistical_status']='INDEPENDENCE_NOT_ESTABLISHED';out['independent_units']=None;out['lower_bound_95']=None
    out['source_groups']={str(g):[r['image_id'] for r in records if r['source_group']==g] for g in {r['source_group'] for r in records}}
    if unit_registry is not None:
        if unit_registry.get('frozen') is not True or not unit_registry.get('protocol_id'):raise ValueError('Frozen unit registry required')
        units=unit_registry['units'];seen=[];outcomes=[];source_units=set()
        byid={r['image_id']:r for r in records}
        for uid,spec in units.items():
            group=spec['source_group'];image_ids=spec['image_ids']
            if not group or not image_ids:raise ValueError('Empty independent unit')
            if group in source_units:raise ValueError('One source group cannot be split into independent units')
            source_units.add(group)
            for iid in image_ids:
                r=byid[iid]
                if r['source_group']!=group or r['independent_unit_id']!=uid:raise ValueError('Unit/source mismatch')
            vals=[byid[i]['strict_image_success'] for i in image_ids]
            outcomes.append(dict(unit_id=uid,source_group=group,image_ids=image_ids,success=None if any(v is None for v in vals) else all(vals)))
            seen+=image_ids
        if len(seen)!=len(set(seen)) or set(seen)!=set(ids):raise ValueError('Registry must partition records')
        known=[o['success'] for o in outcomes if o['success'] is not None]
        out['unit_registry']=unit_registry;out['unit_outcomes']=outcomes
        out['unit_strict_interval']=independent_unit_interval(sum(known),len(known))
        out['statistical_status']='REGISTERED_UNIT_STRICT_ONLY';out['independent_units']=len(known)
    out['schema']=SCORE_SCHEMA
    out['metric_semantics']=dict(METRIC_SEMANTICS)
    out['prediction_diagnostic_counts']=dict(Counter(p['diagnostic_category'] for r in records for p in r['prediction_diagnostics']))
    out['unmatched_joint_predictions']=[dict(image_id=r['image_id'],prediction_ids=r['unmatched_joint_prediction_ids']) for r in records]
    return out

def independent_unit_interval(successes, trials):
    """One-sided 95% exact CP bound, ONLY for declared independent units."""
    from scipy.stats import beta
    if not isinstance(successes,int) or not isinstance(trials,int) or not 0<=successes<=trials:
        raise ValueError('Invalid independent-unit counts')
    lower=None if not trials else (0. if not successes else float(beta.ppf(.05,successes,trials-successes+1)))
    return dict(successes=successes,trials=trials,rate=_ratio(successes,trials),independent_units=trials,
                lower_bound_95=lower,status='BLOCKED_MISSING_GROUP' if not trials else
                ('PASS_OBSERVED_99_AND_LOWER_BOUND' if lower>=.99 else 'PASS_OBSERVED_99_STAT_UNPROVEN') if successes/trials>=.99 else 'FAIL_OBSERVED_99')
