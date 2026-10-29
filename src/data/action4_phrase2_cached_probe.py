"""Bounded actual Cycles-cache experiment; two different poses, no formal data writes."""
from pathlib import Path
import sys,json,hashlib,time,os,sqlite3,numpy as np
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))

def run():
    from data.action4_phrase2_acceleration import install
    install(__file__)
    from data.action4_phrase2_runtime import Phrase2Runtime
    from data.action4_phrase2_surface import SurfaceDeriver
    from data import action4_phrase2_pixels as pixels
    from data import action4_synthesis as synth
    from data import action4_phrase2_amodal as amodal
    from data.action4_phrase2_visibility import canonical_visibility
    cfg=json.loads((R/'configs/data/action4_phrase2_v001.json').read_text());root=R/cfg['output_root'];out=root/'persistent_probe_v002';out.mkdir(exist_ok=False)
    previous=json.loads((root/'persistent_probe_v001/COMPLETE.json').read_text())
    records=[json.loads((R/row['candidate']).read_text())['baseline'] for row in previous['rows']]
    db=sqlite3.connect(root/'slots.sqlite3')
    for scene in ['wall','cloth_lift10_full_garment']:
        raw=db.execute("select record from slots where status='accepted' and scene=? order by updated desc limit 1",(scene,)).fetchone()[0]
        records.append(json.loads(raw))
    db.close()
    runtime=Phrase2Runtime('table_full_garment');deriver=SurfaceDeriver();default=bool(runtime.scene.render.use_persistent_data)
    runtime.scene.render.use_persistent_data=True
    timings=[];active={'case':None};orig_device=synth._render_device;orig_render=pixels._render_call;orig_amodal=amodal.render_crop
    def device(scene,settings):
        t=time.monotonic();v=orig_device(scene,settings);timings.append({'case':active['case'],'stage':'device_selection','seconds':time.monotonic()-t});return v
    def render(scene,write_still):
        t=time.monotonic();v=orig_render(scene,write_still);timings.append({'case':active['case'],'stage':'RGB' if write_still else 'visible_masks','seconds':time.monotonic()-t});return v
    def crop(*args,**kwargs):
        t=time.monotonic();v=orig_amodal(*args,**kwargs);timings.append({'case':active['case'],'stage':'amodal_crop','seconds':time.monotonic()-t});return v
    synth._render_device=device;pixels._render_call=render;amodal.render_crop=crop
    ids={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in pixels.REQUIRED_SOURCE_PATHS};canonical=runtime.metadata['canonical_definition'];ids[canonical['source_manifest']['path']]=canonical['source_manifest']['sha256'];results=[]
    for i,baseline in enumerate(records):
        payload=json.loads((R/baseline['source_candidate_record']).read_text());proposal=payload['proposal'];
        if baseline['scene_id']!=runtime.name:runtime=Phrase2Runtime(baseline['scene_id'])
        runtime.apply(proposal);actual=runtime.export_corners();assert actual['max_proposal_vs_actual_error_px']<=.005
        dest=out/f'pose_{i}';dest.mkdir();data=dest/'data';data.mkdir();meta=dest/'runtime';meta.mkdir();active['case']=i
        settings={'device':'CUDA','denoiser':'OPTIX','samples':32,'mask_samples':1,'amodal_mode':'crop','amodal_margin_px':16,'seed':proposal['entry']['seed'],'worker_gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']}
        t=time.monotonic();value=pixels.render_frame(scene=runtime.scene,entry=proposal['entry'],targets=runtime.targets(),canonical=canonical,K=runtime.K.tolist(),data_out=data,runtime_out=meta,output_role='diagnostic',source_identities=ids,render_settings=settings,verify_scene=runtime.verify);record=value['record'];record=deriver.derive(record,data/'geometry');elapsed=time.monotonic()-t
        vis=[]
        for obj,base in zip(record['instances'],baseline['instances']):
            g=obj['canonical8_geometry'];flags=canonical_visibility(np.array(record['corners_3d_mm'])/1000,np.array(g['object_to_capture_cv_m']),g['ray_clear_to_virtual_point'],g['point_inframe'])
            vis.append({'same_corners':bool(np.allclose(obj['corners_px'],base['corners_px'],atol=.005,rtol=0)),'same_visibility':flags['visible']==base['corner_visible'],'mask_byte_identity':{k:obj[k+'_sha256']==base[k+'_sha256'] for k in ['mask_visible','mask_amodal_expanded']}})
        (dest/'CANDIDATE.json').write_text(json.dumps({'record':record,'proposal':proposal,'slot':payload['slot'],'baseline':baseline},allow_nan=False))
        results.append({'case':i,'sample_id':baseline['sample_id'],'candidate':str((dest/'CANDIDATE.json').relative_to(R)),'seconds':elapsed,'baseline_production_seconds':payload['render_seconds'],'geometry':vis,'persistent_after_render':bool(runtime.scene.render.use_persistent_data)})
        (out/'PROGRESS.json').write_text(json.dumps({'default_persistent_data':default,'rows':results,'phase_timings':timings,'formal_accepted':False},indent=2));print('pose',i,'seconds',elapsed,'baseline_production_seconds',payload['render_seconds'],flush=True)
    (out/'COMPLETE.json').write_text(json.dumps({'default_persistent_data':default,'rows':results,'phase_timings':timings,'formal_accepted':False,'only_tested_change':'persistent GPU session and once-initialized assigned device, validated each frame; original geometry/render/mask algorithms unchanged'},indent=2))
if __name__=='__main__':run()
