"""Thin diagnostic-only orchestration over existing Action4 render helpers."""
import argparse,contextlib,json,math,os,subprocess,sys
from pathlib import Path
from data import action4_global_environment as env,action4_global_scene as integrated
from data import action4_synthesis as synth,action4_amodal as amodal
from data import action4_canonical_export as canonical_export
from data.action4_devices import ALLOWED_GPU_UUIDS,worker_environment
ROOT=env.ROOT
OUTPUTS=ROOT/'outputs/action4_realism_reassessment_v001/global_render_adapter_v001'
SOURCES={'integrated_asset','integrated_contract','integrated_config','integrated_module','environment_module','render_module','synthesis_module','amodal_module','devices_module','adapter_module','blender','global_render_module','canonical_export_module'}
RENDER_FIELDS={'device','denoiser','samples','mask_samples','amodal_mode','amodal_margin_px','seed'}


def validate_config(cfg):
    if set(cfg)!={'schema','mode','dataset_generation_allowed','sources','resolution_wh','threads','render','inspection_poses'} or cfg['schema']!='action4.global_diagnostic_preview.v2' or cfg['mode']!='diagnostic_only' or cfg['dataset_generation_allowed'] is not False:
        raise ValueError('Diagnostic-only unfitted preview required; production forbidden')
    if set(cfg['sources'])!=SOURCES:raise ValueError('Complete frozen input identity required')
    for ref in cfg['sources'].values():
        if set(ref)!={'path','sha256'} or env.sha(env.bounded(ref['path'],True))!=ref['sha256']:raise ValueError('Preview source/input changed')
    for name,module_path in (('global_render_module',Path(__file__)),('canonical_export_module',Path(canonical_export.__file__))):
        if env.bounded(cfg['sources'][name]['path'],True)!=module_path.resolve():raise ValueError('Loaded export module path differs from frozen source')
    if cfg['resolution_wh']!=[3248,2464] or type(cfg['threads']) is not int or cfg['threads']!=2:raise ValueError('Native resolution and explicit two-thread budget required')
    r=cfg['render']
    if set(r)!=RENDER_FIELDS or r['device'] not in ('CUDA','OPTIX') or type(r['samples']) is not int or r['samples']!=64 or type(r['mask_samples']) is not int or r['mask_samples']!=1 or type(r['seed']) is not int:
        raise ValueError('Native RGB64/mask1 explicit GPU settings required')
    synth._denoiser_name(r);amodal.validate_settings(r)
    shots=cfg['inspection_poses']
    if not isinstance(shots,list) or [x.get('id') for x in shots]!=['whole_table','left_wrist','right_wrist']:raise ValueError('Exactly three named engineering inspection poses required')
    for shot in shots:
        if set(shot)!={'id','pose','derivation','required_visible_target_ids'} or not isinstance(shot['derivation'],str) or not shot['derivation']:raise ValueError('Pose derivation required')
        env.validate_pose(shot['pose'])
        required={'whole_table':[],'left_wrist':[1],'right_wrist':[2]}[shot['id']]
        if shot['required_visible_target_ids']!=required:raise ValueError('Engineering wrist inspection must require its target visibility')
    c=json.loads(env.bounded(cfg['sources']['integrated_contract']['path'],True).read_text())
    if c['saved_asset']!=cfg['sources']['integrated_asset'] or c['source_module_sha256']!=cfg['sources']['integrated_module']['sha256']:raise ValueError('Integrated asset/source identity mismatch')
    if c['config']!=json.loads(env.bounded(cfg['sources']['integrated_config']['path'],True).read_text()) or c['dataset_generation_allowed'] is not False:raise ValueError('Frozen integrated config/scope mismatch')
    integrated.validate_config(c['config']);integrated.verify_canonical_definition(c['config'],c['canonical8'])
    if c['environment']['config']['native_image']['resolution_wh']!=cfg['resolution_wh']:raise ValueError('Observing K/resolution mismatch')
    return cfg,c


def output_path(value,*,fresh=True):
    p=env.bounded(value)
    if not p.is_relative_to(OUTPUTS) or p==OUTPUTS:raise ValueError('Diagnostics only in assigned outputs subtree, never data/train')
    if fresh and p.exists():raise FileExistsError('Fresh diagnostic output required; no overwrite')
    return p


def _write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def target_parts(scene):
    result={}
    for side,index in (('left',1),('right',2)):
        core=scene.objects[side+'_Action4_core'];closure={core,*core.children_recursive}
        parts=[o for o in closure if o.type=='MESH' and o.pass_index==index]
        if core not in parts:raise ValueError('Target core index mismatch')
        result[index]={'core':core,'parts':parts}
    owned={o for r in result.values() for o in r['parts']}
    if any(o.pass_index not in (0,1,2) or (o.pass_index in (1,2) and o not in owned) for o in scene.objects):raise ValueError('Non-target object carries target label')
    if any(o.pass_index!=0 for o in scene.objects if o.name.startswith(integrated.PREFIX)):raise ValueError('Accessory target-label fusion')
    return result


RENDER_ATTRS=('resolution_x','resolution_y','resolution_percentage','pixel_aspect_x','pixel_aspect_y','filepath','use_border','use_crop_to_border','border_min_x','border_max_x','border_min_y','border_max_y','threads_mode','threads','film_transparent')
CYCLES_ATTRS=('device','samples','use_denoising','denoiser','seed','max_bounces','diffuse_bounces','glossy_bounces','transparent_max_bounces','use_adaptive_sampling')


def state_snapshot(scene):
    camera=scene.camera
    return {'render':{k:getattr(scene.render,k) for k in RENDER_ATTRS},
            'cycles':{k:getattr(scene.cycles,k) for k in CYCLES_ATTRS},
            'image':{k:getattr(scene.render.image_settings,k) for k in ('file_format','color_mode','color_depth')},
            'camera_sensor_width':camera.data.sensor_width,
            'camera_channels':{k:list(getattr(camera,k)) for k in ('location','rotation_euler','rotation_quaternion','rotation_axis_angle','scale')},
            'hide':{o.name:[o.hide_render,o.hide_viewport] for o in scene.objects},
            'use_nodes':scene.use_nodes,'index_pass':scene.view_layers[0].use_pass_object_index,
            'frame':scene.frame_current}


@contextlib.contextmanager
def temporary_render_state(scene):
    """Always restore temporary active render/camera/hide state, including failure."""
    import bpy
    if scene.use_nodes or (scene.node_tree and len(scene.node_tree.nodes)):
        raise ValueError('Diagnostic adapter requires frozen input without active compositor nodes')
    old=state_snapshot(scene)
    try:yield
    finally:
        for k,v in old['render'].items():setattr(scene.render,k,v)
        for k,v in old['cycles'].items():setattr(scene.cycles,k,v)
        for k,v in old['image'].items():setattr(scene.render.image_settings,k,v)
        scene.camera.data.sensor_width=old['camera_sensor_width']
        for k,v in old['camera_channels'].items():setattr(scene.camera,k,v)
        for name,(render,viewport) in old['hide'].items():scene.objects[name].hide_render=render;scene.objects[name].hide_viewport=viewport
        if scene.node_tree:scene.node_tree.nodes.clear()
        scene.use_nodes=old['use_nodes'];scene.view_layers[0].use_pass_object_index=old['index_pass']
        scene.frame_set(old['frame']);bpy.context.view_layer.update()


@contextlib.contextmanager
def isolated_expanded(scene,parts):
    """Same 3W/3H, sensor96 convention as synthesis; exact restoration on any failure."""
    import bpy
    old_hide={o.name:o.hide_render for o in scene.objects}
    w,h=scene.render.resolution_x,scene.render.resolution_y;sensor=scene.camera.data.sensor_width
    samples,denoise=scene.cycles.samples,scene.cycles.use_denoising
    try:
        for o in scene.objects:
            if o.type not in ('CAMERA','LIGHT'):o.hide_render=o not in parts
        scene.render.resolution_x=3*w;scene.render.resolution_y=3*h;scene.camera.data.sensor_width=3*sensor
        scene.cycles.samples=1;scene.cycles.use_denoising=False;bpy.context.view_layer.update()
        yield
    finally:
        for name,v in old_hide.items():scene.objects[name].hide_render=v
        scene.render.resolution_x=w;scene.render.resolution_y=h;scene.camera.data.sensor_width=sensor
        scene.cycles.samples=samples;scene.cycles.use_denoising=denoise;bpy.context.view_layer.update()


def project_canonical(scene,core,corners,*,return_ray_clear=False):
    """Original integer-centre projection and ray visibility; hidden scaffolds excluded."""
    import bpy,numpy as np
    from mathutils import Vector
    from bpy_extras.object_utils import world_to_camera_view
    w,h=scene.render.resolution_x,scene.render.resolution_y;camera=scene.camera;projected=[];visible=[];ray_clear=[]
    contributors=set(integrated.contributing_objects(scene))
    excluded=[o for o in scene.objects if o.type=='MESH' and o not in contributors]
    old={o.name:o.hide_viewport for o in excluded}
    try:
        for o in excluded:o.hide_viewport=True
        bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get()
        for corner in corners:
            point=core.matrix_world@Vector(corner);v=world_to_camera_view(scene,camera,point)
            if v.z<=0:raise ValueError('Canonical corner behind observer')
            projected.append([float(v.x*w-.5),float((1-v.y)*h-.5)])
            direction=point-camera.matrix_world.translation;distance=direction.length
            hit,location,*_=scene.ray_cast(deps,camera.matrix_world.translation,direction.normalized(),distance=distance+1e-6)
            clear=bool(not hit or (location-camera.matrix_world.translation).length>=distance-1e-5)
            ray_clear.append(clear)
            visible.append(bool(clear and 0<=v.x<=1 and 0<=v.y<=1))
    finally:
        for name,value in old.items():scene.objects[name].hide_viewport=value
        bpy.context.view_layer.update()
    pts=np.asarray(projected)
    if (pts[:,0]<-w-.5).any() or (pts[:,0]>2*w-.5).any() or (pts[:,1]<-h-.5).any() or (pts[:,1]>2*h-.5).any():raise ValueError('Expanded canvas cannot contain target')
    return (projected,visible,ray_clear) if return_ray_clear else (projected,visible)


def canonical_geometry(scene,core,canonical,index,points,visible,ray_clear):
    """Metadata only; trusted current evaluated transforms and original legacy pixels."""
    import bpy
    context={'canonical':canonical,'instance_id':index,
             'object_to_world_m':[list(r) for r in core.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world],
             'camera_to_world_blender_m':[list(r) for r in scene.camera.matrix_world],
             'K':env.actual_K(scene),'image_size_wh':[scene.render.resolution_x,scene.render.resolution_y],
             'near_plane_m':scene.camera.data.clip_start,'ray_clear_to_virtual_point':ray_clear,
             'legacy_corners_px':points,'legacy_corner_visible':visible}
    value=canonical_export.build_export(**context)
    canonical_export.verify_export(value,**context)
    return value


def inspection_visibility_guard(scene,targets,corners,required_ids):
    """Explicit inspection semantics only: reject cameras inside solids and hidden wrist targets."""
    import bpy,numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    from bpy_extras.object_utils import world_to_camera_view
    camera=scene.camera;origin=camera.matrix_world.translation.copy()
    deps=bpy.context.evaluated_depsgraph_get();containers=[]
    contributors=integrated.contributing_objects(scene)
    for obj in contributors:
        evaluated=obj.evaluated_get(deps)
        # Cheap evaluated bounds filter; only candidate solids need a parity test.
        box=[evaluated.matrix_world@Vector(p) for p in evaluated.bound_box]
        if not all(min(p[i] for p in box)<=origin[i]<=max(p[i] for p in box) for i in range(3)):continue
        mesh=evaluated.to_mesh()
        try:
            tree=BVHTree.FromPolygons([evaluated.matrix_world@v.co for v in mesh.vertices],[list(p.vertices) for p in mesh.polygons])
            votes=[]
            for d in (Vector((1,.173,.319)).normalized(),Vector((.127,1,.237)).normalized(),Vector((.193,.271,1)).normalized()):
                point=origin.copy();count=0
                for _ in range(128):
                    hit,normal,index,distance=tree.ray_cast(point,d,100)
                    if hit is None:break
                    count+=1;point=hit+d*1e-6
                else:raise ValueError('Diagnostic containment ray did not terminate')
                votes.append(count%2)
            containers.append({'object':obj.name,'inside_votes':votes})
            if votes==[1,1,1]:raise ValueError('Engineering observer is inside render contributor: '+obj.name)
        finally:evaluated.to_mesh_clear()
    rays={}
    a=np.asarray(corners);center=a.mean(axis=0)
    # Original ordered cuboid face indices; label order itself remains untouched.
    representatives=[('center',center)]+[(name,a[ids].mean(axis=0)) for name,ids in
        [('zmin',[0,1,2,3]),('zmax',[4,5,6,7]),('ymin',[0,1,4,5]),('ymax',[2,3,6,7]),('xmin',[0,3,4,7]),('xmax',[1,2,5,6])]]
    for j in required_ids:
        target=targets[j];rows=[]
        for name,point in representatives:
            world=target['core'].matrix_world@Vector(point);direction=world-origin
            hit,location,normal,face,obj,matrix=scene.ray_cast(deps,origin,direction.normalized(),distance=100)
            projected=world_to_camera_view(scene,camera,world)
            visible=bool(hit and obj in target['parts'] and projected.z>0 and 0<=projected.x<=1 and 0<=projected.y<=1)
            rows.append({'point':name,'first_hit':obj.name if hit else None,'distance_m':float((location-origin).length) if hit else None,'target_visible':visible})
        rays[j]=rows
        if not any(row['target_visible'] for row in rows):raise ValueError('Required engineering wrist target fully hidden on centre/surface probes: '+str(j))
    return {'camera_solid_candidates':containers,'required_target_first_hits':rays,'claim':'Engineering inspection prerequisite, not a production visibility threshold'}


def crop_geometry_preflight(scene,core,parts,corners):
    import bpy
    from mathutils import Vector
    from bpy_extras.object_utils import world_to_camera_view
    w,h=scene.render.resolution_x,scene.render.resolution_y;camera=scene.camera
    def project(point):
        p=world_to_camera_view(scene,camera,point);return [float(p.x*w),float(p.y*h),float(p.z)]
    bounds=amodal.crop_bounds([project(core.matrix_world@Vector(c)) for c in corners],w,h)
    counts={}
    for part in parts:
        evaluated=part.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=evaluated.to_mesh()
        try:
            amodal.validate_geometry_coverage([project(evaluated.matrix_world@v.co) for v in mesh.vertices],bounds,w,h)
            counts[part.name]=len(mesh.vertices)
        finally:evaluated.to_mesh_clear()
    return {'expanded_crop_bottomleft_xyxy':list(bounds),'evaluated_target_vertices':counts}


def mask_measurements(visible,full,width,height):
    """Original synthesis visible/amodal definitions and fixed rejection/ignore thresholds."""
    import numpy as np
    from data.action4_adapter import DataAdapter
    if visible.shape!=(height,width) or full.shape!=(3*height,3*width) or visible.dtype!=np.bool_ or full.dtype!=np.bool_:raise ValueError('Native binary mask shapes required')
    inside=full[height:2*height,width:2*width];av,ai,af=int(visible.sum()),int(inside.sum()),int(full.sum())
    if af==0:raise ValueError('Empty amodal mask')
    bad=int((visible&~inside).sum())
    if bad>max(4,.005*ai):raise ValueError('Mask boundary mismatch exceeds fixed tolerance')
    if ai==0:return {'excluded':True,'reason':'outside_image','area_full_amodal_px':af}
    affine,_=DataAdapter.letterbox_affine((width,height),(1024,768));support=av*float(affine[0,0]*affine[1,1]);ignored=support<16 or av/ai<.1
    result={'area_visible_px':av,'area_inframe_amodal_px':ai,'area_full_amodal_px':af,
            'occlusion_inframe':float(max(0,1-av/ai)),'occlusion_inframe_raw':float(1-av/ai),'truncation':float(1-ai/af),
            'mask_nonnested_boundary_pixels':bad,'visible_area_at_fullinput_px':support,
            'ignore':bool(ignored),'ignore_reason':'low_visible_support' if ignored else None}
    yy,xx=np.nonzero(visible)
    if len(xx):result['bbox_visible_xyxy']=[float(xx.min()-.5),float(yy.min()-.5),float(xx.max()+.5),float(yy.max()+.5)]
    return result


def execute_blender(config_path,out,threads,*,gpu_uuid=None,probe_only=False):
    import bpy,numpy as np
    from data import action4_render as base
    cfg,c=validate_config(json.loads(env.bounded(config_path,True).read_text()));out=output_path(out,fresh=False)
    if threads!=cfg['threads'] or (probe_only and gpu_uuid is not None) or (not probe_only and gpu_uuid not in ALLOWED_GPU_UUIDS):raise ValueError('Explicit diagnostic resource identity mismatch')
    bpy.ops.wm.open_mainfile(filepath=str(env.bounded(cfg['sources']['integrated_asset']['path'],True)),load_ui=False,use_scripts=False)
    scene=bpy.context.scene;bpy.context.view_layer.update();integrated.verify(scene,c);targets=target_parts(scene)
    corners=c['canonical8']['corners_3d_m'];before=state_snapshot(scene);records=[]
    try:
        with temporary_render_state(scene):
            scene.render.threads_mode='FIXED';scene.render.threads=threads;scene.frame_set(1)
            scene.cycles.samples=64;scene.cycles.seed=cfg['render']['seed'];scene.cycles.max_bounces=6
            scene.cycles.diffuse_bounces=2;scene.cycles.glossy_bounces=2;scene.cycles.transparent_max_bounces=4;scene.cycles.use_adaptive_sampling=False
            scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.image_settings.color_depth='8'
            scene.render.use_border=False;scene.render.use_crop_to_border=False;scene.render.film_transparent=False
            device=None;denoiser=None;nodes=None
            if not probe_only:
                device=synth._render_device(scene,{**cfg['render'],'worker_gpu_uuid':gpu_uuid});denoiser=synth._set_denoiser(scene,cfg['render'])
                for folder in ('rgb','masks','scratch'):(out/folder).mkdir()
                nodes=base._mask_nodes(out/'masks')
            for shot in cfg['inspection_poses']:
                env.apply_capture_pose(scene,shot['pose'],c['environment']);integrated.verify(scene,c)
                sid=shot['id'];instances=[];excluded=[];projections={};geometry_exports={}
                inspection=inspection_visibility_guard(scene,targets,corners,shot['required_visible_target_ids'])
                for j,target in targets.items():
                    projected,visible,ray_clear=project_canonical(scene,target['core'],corners,return_ray_clear=True)
                    geometry_exports[j]=canonical_geometry(scene,target['core'],c['canonical8'],j,projected,visible,ray_clear)
                    projections[j]=(projected,visible)
                    with isolated_expanded(scene,target['parts']):
                        coverage=crop_geometry_preflight(scene,target['core'],target['parts'],corners)
                    instances.append({'instance_id':j,'target_parts':[o.name for o in target['parts']],'corners_px':projected,'corner_visible':visible,'canonical8_geometry':geometry_exports[j],**coverage})
                if probe_only:
                    records.append({'sample_id':sid,'pose':shot['pose'],'K':env.actual_K(scene),'instances':instances,'inspection_visibility':inspection,'rendered':False})
                    continue
                for node in nodes:node.mute=True
                scene.cycles.samples=64;scene.cycles.use_denoising=True;scene.render.filepath=str(out/'rgb'/(sid+'.png'))
                bpy.ops.render.render(write_still=True)
                for j,node in enumerate(nodes,1):node.mute=False;node.file_slots[0].path=f'{sid}_i{j}_visible_'
                scene.cycles.samples=1;scene.cycles.use_denoising=False;bpy.ops.render.render(write_still=False)
                rows=[]
                for j,target in targets.items():
                    vp=out/'masks'/f'{sid}_i{j}_visible_0001.png';ap=out/'masks'/f'{sid}_i{j}_amodal_for_{j}_0001.png'
                    for k,node in enumerate(nodes,1):node.file_slots[0].path=f'{sid}_i{k}_amodal_for_{j}_'
                    with isolated_expanded(scene,target['parts']):
                        if cfg['render']['amodal_mode']=='crop':
                            full,info=amodal.render_crop(scene,scene.camera,target['core'],target['parts'],corners,nodes,j,ap,out/'scratch'/f'{sid}_i{j}')
                        else:
                            bpy.ops.render.render(write_still=False);full=base._mask_array(ap);info={'mode':'full'}
                    measurements=mask_measurements(base._mask_array(vp),full,*cfg['resolution_wh'])
                    if measurements.pop('excluded',False):
                        excluded.append({'instance_id':j,'object_id':1,'canonical8_geometry':geometry_exports[j],'training_accepted':False,**measurements});continue
                    pts,visible=projections[j];a=np.asarray(pts)
                    rows.append({'instance_id':j,'object_id':1,'corners_px':pts,'corner_visible':visible,
                        'canonical8_geometry':geometry_exports[j],
                        'bbox_full_xyxy':[float(a[:,0].min()),float(a[:,1].min()),float(a[:,0].max()),float(a[:,1].max())],
                        'mask_visible':str(vp.relative_to(ROOT)),'mask_amodal_expanded':str(ap.relative_to(ROOT)),
                        'mask_visible_sha256':env.sha(vp),'mask_amodal_expanded_sha256':env.sha(ap),
                        'amodal_expanded_origin_px':[-3248,-2464],'amodal_expanded_shape_hw':[7392,9744],
                        'amodal_render':info,**measurements})
                records.append({'schema_version':'action4.v1','sample_id':sid,'status':'diagnostic_render_not_training_accepted','dataset_generation_allowed':False,
                    'image':str((out/'rgb'/(sid+'.png')).relative_to(ROOT)),'width':3248,'height':2464,'instances':rows,'excluded_instances':excluded,
                    'inspection_visibility':inspection,'canonical8':c['canonical8'],'camera':{'K':env.actual_K(scene),'camera_to_world_m':[list(r) for r in scene.camera.matrix_world],'calibration_status':'unfitted engineering inspection'},
                    'render_settings':{**cfg['render'],'actual_device':device,'actual_denoiser':denoiser},'rgb_sha256':env.sha(out/'rgb'/(sid+'.png'))})
    finally:
        if state_snapshot(scene)!=before:raise RuntimeError('Temporary diagnostic state was not fully restored')
        integrated.verify(scene,c)
    report={'status':'PASS_CPU_PREFLIGHT_NO_RENDER' if probe_only else 'DIAGNOSTIC_RENDER_COMPLETE_NOT_ACCEPTED','dataset_generation_allowed':False,
            'rendered':not probe_only,'source_config_sha256':env.sha(env.bounded(config_path,True)),'asset':cfg['sources']['integrated_asset'],'source_identities':cfg['sources'],
            'target_ownership':{j:[o.name for o in r['parts']] for j,r in targets.items()},
            'hidden_scaffolds_excluded':[s+'_mount_initial_support' for s in ('left','right')],'restoration_verified':True,'records':records}
    _write(out/('preflight.json' if probe_only else 'diagnostic_report.json'),report)
    if not probe_only:
        with (out/'manifest.jsonl').open('x') as f:
            for record in records:f.write(json.dumps(record,allow_nan=False)+chr(10))
    return report


def launch(config_path,out,threads,*,gpu_uuid=None,probe_only=False):
    cfg,_=validate_config(json.loads(env.bounded(config_path,True).read_text()));out=output_path(out)
    if threads!=cfg['threads'] or (probe_only and gpu_uuid is not None) or (not probe_only and gpu_uuid not in ALLOWED_GPU_UUIDS):raise ValueError('Pinned GPU4-7 or explicit CPU preflight and frozen thread budget required')
    child_env=worker_environment(None if probe_only else gpu_uuid)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):child_env[key]=str(threads)
    cmd=[str(env.bounded(cfg['sources']['blender']['path'],True)),'--factory-startup','--disable-autoexec','-noaudio','--background','--threads',str(threads),'--python-exit-code','1','--python',str(ROOT/'scripts/action4_global_render.py'),'--','--inside-blender','--config',str(env.bounded(config_path,True)),'--output',str(out),'--threads',str(threads)]
    if probe_only:cmd.append('--probe-only')
    else:cmd.extend(['--gpu-uuid',gpu_uuid])
    out.mkdir(parents=True);_write(out/'launch.json',{'command':cmd,'probe_only':probe_only,'gpu_uuid':gpu_uuid,'config_sha256':env.sha(env.bounded(config_path,True)),'adapter_sha256':env.sha(Path(__file__))})
    with (out/'blender.log').open('xb') as log:process=subprocess.run(cmd,cwd=ROOT,env=child_env,stdout=log,stderr=subprocess.STDOUT)
    if process.returncode:raise RuntimeError('Diagnostic Blender failed; preserved '+str(out/'blender.log'))
    return json.loads((out/('preflight.json' if probe_only else 'diagnostic_report.json')).read_text())


def main():
    args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--threads',required=True,type=int);p.add_argument('--gpu-uuid');p.add_argument('--probe-only',action='store_true');p.add_argument('--inside-blender',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args(args)
    if a.inside_blender and 'bpy' not in sys.modules:raise ValueError('Internal Blender entry cannot launch GPU work directly')
    result=(execute_blender if a.inside_blender else launch)(a.config,a.output,a.threads,gpu_uuid=a.gpu_uuid,probe_only=a.probe_only)
    print(json.dumps({'status':result['status'],'rendered':result['rendered'],'dataset_generation_allowed':False}))
