import numpy as np
from model.action4_phase2_tracking_contracts import AnchorKey,Binding,AnchorRegistry,partition_geometry
from eval.action4_fixed_material_pose_diagnostic import solve_diagnostic,project


def fixture():
    rng=np.random.default_rng(30);x=rng.uniform(-.03,.03,(15,3))
    keys=tuple(AnchorKey("body",str(i),0) for i in range(15))
    registry=AnchorRegistry([Binding(k,tuple(v),"fixture",0,"test",i>=12) for i,(k,v) in enumerate(zip(keys,x))])
    K=np.array([[800.,0,320],[0,800,240],[0,0,1]])
    t=np.array([.01,.02,.6]);uv=project(x,K,np.eye(3),t)
    cad=np.array([[a,b,c] for a in [-.03,.03] for b in [-.02,.02] for c in [-.015,.015]])
    return registry,keys,uv,K,t,cad


def fit(reg,keys,uv):
    return partition_geometry(reg,keys,uv,np.ones(15,dtype=bool),np.zeros(15,dtype=bool),np.ones(15),tuple(i%4 for i in range(15)))[0]


def test_fixed_binding_fit_has_no_heldout_consumption():
    reg,keys,uv,K,t,cad=fixture();a=solve_diagnostic(fit(reg,keys,uv),K=K,initial_R=np.eye(3),initial_t=t,canonical=cad,threshold_px=2.)
    uv[12:]=np.nan;b=solve_diagnostic(fit(reg,keys,uv),K=K,initial_R=np.eye(3),initial_t=t,canonical=cad,threshold_px=2.)
    assert a==b
    assert a['status']=='diagnostic_candidate_only'
    assert np.max(np.abs(np.asarray(a['translation_m'])-t))<1e-6
    assert a['geometry_accepted'] is False and a['binding_confirmed'] is False
    assert len(a['solver_anchor_ids'])==12


def test_real_observations_change_solution_not_just_fields():
    reg,keys,uv,K,t,cad=fixture();a=solve_diagnostic(fit(reg,keys,uv),K=K,initial_R=np.eye(3),initial_t=t,canonical=cad,threshold_px=2.)
    uv[:12,0]+=8;b=solve_diagnostic(fit(reg,keys,uv),K=K,initial_R=np.eye(3),initial_t=t,canonical=cad,threshold_px=2.)
    assert b['status']=='diagnostic_candidate_only'
    assert np.linalg.norm(np.array(a['translation_m'])-b['translation_m'])>.002


def test_eleven_points_cannot_solve():
    reg,keys,uv,K,t,cad=fixture();obs=np.ones(15,dtype=bool);obs[0]=False
    p,_=partition_geometry(reg,keys,uv,obs,np.zeros(15,dtype=bool),np.ones(15),tuple(i%4 for i in range(15)))
    assert solve_diagnostic(p,K=K,initial_R=np.eye(3),initial_t=t,canonical=cad,threshold_px=2.)['reason']=='fewer_than12_fit_or4_grids'


def test_run_contract_rejects_wrong_heldout_and_cross_track():
    import copy
    from eval.action4_fixed_material_pose_diagnostic import validate_domain
    ids=[dict(track_id='body',anchor_id=str(i),generation=0) for i in range(15)]
    models={'A0':{'sha256':'fixture_a0'},'B0':{'sha256':'fixture_b0'}}
    ref={'path':'fixture_track','sha256':'fixture_track_identity'}
    surf={'path':'fixture_surface','sha256':'fixture_surface_identity'}
    cfg={'heldout_rule':'lexicographic_anchor_id_first3_before_any_solver'}
    item={'track':ref,'surface':surf,'heldout_anchor_ids':sorted(x['anchor_id'] for x in ids)[:3]}
    source={'track_id':'body','seed_record':{'frame_id':4,'seeds':[{'key':k} for k in ids],
             'source_surface':surf,'source_models':models},
            'identity':{'anchor_ids':ids,'frame_ids':[4,5,6],'frame_PTS':[1600,2000,2400]},
            'frame_observations':[{'frame_id':f,'pts':f*400} for f in [4,5,6]]}
    cycle={'track_id':'body','source_track':ref,'records':[{'anchor_id':k['anchor_id']} for k in ids]}
    arrays={'uv_native':np.zeros((3,15,2)),'is_query':np.zeros((3,15),bool)}
    meta={'models':models,'prediction_id':'video:body:frame:4','video_context':{'frame_id':4}}
    validate_domain(cfg,item,source,cycle,arrays,meta)
    for kind in ['heldout','cycle','surface','frames']:
        ii,ss,cc,mm=map(copy.deepcopy,[item,source,cycle,meta])
        if kind=='heldout':ii['heldout_anchor_ids']=[]
        if kind=='cycle':cc['source_track']={'path':'other','sha256':'valid_but_other'}
        if kind=='surface':mm['prediction_id']='video:track_01:frame:4'
        if kind=='frames':ss['frame_observations'][1]['pts']+=1
        try:validate_domain(cfg,ii,ss,cc,arrays,mm)
        except ValueError:pass
        else:raise AssertionError(kind+' not rejected')
