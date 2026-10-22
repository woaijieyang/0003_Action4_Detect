"""Four-cell diagnostic interventions; never publishes training data."""
from pathlib import Path
import json,os,hashlib
import numpy as np
from data.action4_template_runtime import TemplateRuntime,bounded
from data.action4_synthesis import _render_device,_set_denoiser


def render(config_path):
    import bpy
    root=Path('/data/zy/vln/a-test')
    if Path.cwd().resolve()!=root:raise ValueError('Wrong project directory')
    cfg=json.loads(bounded(config_path).read_text());out=bounded(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'RENDER_COMPLETE.json').exists():raise FileExistsError('Diagnostic already rendered')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=cfg['gpu_uuid']:raise ValueError('GPU visibility mismatch')
    rows=[]
    for case in cfg['cases']:
        raw=bounded(case['parameters_ref']['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=case['parameters_ref']['sha256']:raise ValueError('Source proposal changed')
        runtime=TemplateRuntime(case['template']);proposal=json.loads(raw);targets=runtime.apply(proposal);actual=runtime.export_corners()
        original_flags=dict(runtime.expected_flags);target_names={obj.name for item in targets.values() for obj in item['parts']};patch_names={obj.name for obj in runtime.patches.values()}
        scene=runtime.scene;scene.render.engine='CYCLES';scene.render.threads_mode='FIXED';scene.render.threads=cfg['threads'];scene.render.resolution_percentage=100
        scene.cycles.samples=64;scene.cycles.seed=case['source_seed'];scene.cycles.max_bounces=6;scene.cycles.diffuse_bounces=2;scene.cycles.glossy_bounces=2;scene.cycles.transparent_max_bounces=4;scene.cycles.use_adaptive_sampling=False
        settings={'device':'CUDA','denoiser':'OPTIX','worker_gpu_uuid':cfg['gpu_uuid']};device=_render_device(scene,settings);_set_denoiser(scene,settings);scene.cycles.use_denoising=True
        scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.image_settings.color_depth='8'
        for cameras,cloth in ((True,True),(True,False),(False,True),(False,False)):
            expected=dict(original_flags)
            for name in target_names:
                hidden,index=original_flags[name];expected[name]=(hidden or not cameras,index)
            for name in patch_names:
                hidden,index=original_flags[name];expected[name]=(hidden or not cloth,index)
            for name,(hidden,index) in expected.items():scene.objects[name].hide_render=hidden;scene.objects[name].pass_index=index
            # Explicit counterfactual visibility contract; original geometry/light checks retained.
            runtime.expected_flags=expected;bpy.context.view_layer.update();runtime.verify(scene)
            filename=f"{case['sample_id']}_camera{int(cameras)}_cloth{int(cloth)}.png";path=out/filename
            if path.exists():raise FileExistsError(path)
            scene.render.filepath=str(path);bpy.ops.render.render(write_still=True);runtime.verify(scene)
            rows.append({'sample_id':filename[:-4],'source_sample':case['sample_id'],'template':case['template'],'image':str(path.relative_to(root)),'cameras_present':cameras,'patches_present':cloth,'reference_boxes':case['reference_boxes'] if cameras else [],'max_corner_export_error_px':actual['max_proposal_vs_actual_error_px'],'actual_device':device})
            (out/'RENDER_PROGRESS.json').write_text(json.dumps({'completed':len(rows),'total':12}))
        runtime.expected_flags=original_flags
    (out/'RENDER_COMPLETE.json').write_text(json.dumps({'status':'DIAGNOSTIC_RENDERED_NOT_DATASET','cases':rows,'training_use':False},indent=2))
