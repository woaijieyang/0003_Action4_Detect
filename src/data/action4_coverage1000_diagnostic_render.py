"""Few diagnostic RGBs of measured constraints, NEVER dataset admission."""
from pathlib import Path
import sys,json,os,hashlib,time
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))

def run(scene):
    from data.action4_phrase2_acceleration import install
    install(__file__)
    from data.action4_phrase2_runtime import Phrase2Runtime
    from data import action4_phrase2_pixels as pixels
    from data.action4_phrase2_surface import SurfaceDeriver
    if scene=='wall':
        source=R/'outputs/coverage1000_v001/feasibility_v005';rows=[]
        for path in sorted(source.glob('wall_*_proposals.json')):rows.extend(json.loads(path.read_text()))
        rows=[r for r in rows if r['proposal']['phrase2']['coverage_group']=='E'][:1]
        cases=[{'proposal':r['proposal'],'counterfactual':False} for r in rows]
    else:
        data=json.loads((R/'outputs/coverage1000_v001/constraint_diagnostic_v001'/(scene+'.json')).read_text())
        rows=[r for r in data['rows'] if r['counterfactual_success'] and r['group']=='E'][:2 if scene=='table_full_garment' else 1]
        cases=[{'proposal':r['counterfactual_success']['proposal'],'counterfactual':True,'elbow_quality_conditions':r['counterfactual_success']['elbow_quality_conditions']} for r in rows]
    out=R/'outputs/coverage1000_v001/constraint_visual_v002'/scene;out.mkdir(parents=True,exist_ok=False)
    runtime=Phrase2Runtime(scene);deriver=SurfaceDeriver();canonical=runtime.metadata['canonical_definition']
    ids={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in pixels.REQUIRED_SOURCE_PATHS};ids[canonical['source_manifest']['path']]=canonical['source_manifest']['sha256']
    receipts=[]
    for i,case in enumerate(cases):
        proposal=case['proposal'];runtime.apply(proposal);entry=proposal['entry'].copy();entry['sample_id']='diagnostic_'+entry['sample_id']
        root=out/f'case{i:02d}';root.mkdir();(root/'pixels').mkdir();(root/'runtime').mkdir();(root/'PROPOSAL.json').write_text(json.dumps(case))
        settings={'device':'CUDA','denoiser':'OPTIX','samples':32,'mask_samples':1,'amodal_mode':'crop','amodal_margin_px':16,'seed':entry['seed'],'worker_gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']}
        begin=time.monotonic();result=pixels.render_frame(scene=runtime.scene,entry=entry,targets=runtime.targets(),canonical=canonical,K=runtime.K.tolist(),data_out=root/'pixels',runtime_out=root/'runtime',output_role='diagnostic',source_identities=ids,render_settings=settings,verify_scene=runtime.verify)
        record=deriver.derive(result['record'],root/'geometry');(root/'RECORD.json').write_text(json.dumps(record))
        receipts.append({'image':record['image'],'record':str((root/'RECORD.json').relative_to(R)),'counterfactual':case['counterfactual'],'seconds':time.monotonic()-begin,'occlusion':[x['occlusion_inframe'] for x in record['instances']]})
    (out/'COMPLETE.json').write_text(json.dumps({'diagnostic_only':True,'never_formally_admitted':True,'receipts':receipts},indent=2));print(json.dumps(receipts),flush=True)

if __name__=='__main__':run(sys.argv[sys.argv.index('--')+1])
