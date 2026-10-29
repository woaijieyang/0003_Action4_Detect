"""Independent body-only reference bank, isolated from training data."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from .action4_paths import PROJECT_ROOT,contained_path

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def mask_bbox(coverage,margin=1):
    a=np.asarray(coverage)
    if a.ndim!=2 or not np.isfinite(a).all() or np.any((a<0)|(a>1)):raise ValueError('Invalid coverage')
    y,x=np.where(a>=.5)
    if not len(x):raise ValueError('Empty body coverage')
    h,w=a.shape
    return [max(0,float(x.min())-.5-margin),max(0,float(y.min())-.5-margin),
            min(w,float(x.max())+.5+margin),min(h,float(y.max())+.5+margin)]

def recipe_views():
    active={(0,0):'front',(180,0):'back',(270,0):'left',(90,0):'right',(0,30):'frontupper',(180,30):'backupper'}
    return [dict(reference_id=f'az{a:03d}_el{e:+03d}_light{l}',azimuth=a,elevation=e,light=l,
                 active_name=active.get((a,e)) if l==0 else None)
            for e in (-30,0,30) for a in range(0,360,45) for l in range(2)]


def validate_recipe(cfg):
    if cfg.get('gray_backgrounds')!=[64,128,192]:raise ValueError('Exactly three registered gray backgrounds required')
    expected=['front','back','left','right','frontupper','backupper']
    if cfg.get('active_views')!=expected:raise ValueError('Six registered active views required')

def formal_bank_complete(cfg,records,*,preview):
    """Completion requires48 distinct mother images,144 RGBs and six views."""
    validate_recipe(cfg)
    if preview or len(records)!=48:return False
    expected={v['reference_id']:v for v in recipe_views()}
    if len({r['reference_id'] for r in records})!=48 or {r['reference_id'] for r in records}!=set(expected):return False
    paths=[];active=[]
    for r in records:
        if r.get('preview') or r.get('resolution')!=1024 or r.get('samples')!=128:return False
        if r.get('active_name')!=expected[r['reference_id']]['active_name']:return False
        if r.get('active_name'):active.append(r['active_name'])
        rgb=r.get('rgb',[])
        if len(rgb)!=3 or sorted(x['gray'] for x in rgb)!=[64,128,192]:return False
        artifacts=[(r['rgba'],r['rgba_sha256']),(r['mask'],r['mask_sha256'])]+[(x['path'],x['sha256']) for x in rgb]
        for name,sha in artifacts:
            p=contained_path(PROJECT_ROOT,name,exists=True)
            if digest(p)!=sha:return False
        coverage=np.load(contained_path(PROJECT_ROOT,r['mask'],exists=True),allow_pickle=False)
        if coverage.shape!=(1024,1024) or not np.isfinite(coverage).all() or not np.any(coverage>=.5):return False
        paths.extend(x['path'] for x in rgb)
    return len(set(paths))==144 and sorted(active)==sorted(cfg['active_views'])

def select_reference_views(reference_ids=None,*,limit=1):
    views=recipe_views()
    if reference_ids is None:
        if type(limit) is not int or not 1<=limit<=48:raise ValueError('limit must be1..48')
        return views[:limit]
    if not isinstance(reference_ids,(list,tuple)) or not reference_ids or not all(isinstance(x,str) for x in reference_ids):
        raise ValueError('reference_ids must be a nonempty sequence of IDs')
    if len(set(reference_ids))!=len(reference_ids):raise ValueError('Duplicate reference IDs')
    byid={v['reference_id']:v for v in views}
    if any(x not in byid for x in reference_ids):raise ValueError('Unknown reference ID')
    return [byid[x] for x in reference_ids]


def build(config_path,*,limit=1,preview=True,reference_ids=None):
    """Explicit Blender invocation only. Default is one bounded CPU preview.

    Front is canonical -Y, right +X, up +Z (action4_render.render_asset_views).
    Derive material copies; never save or overwrite the frozen foundation.
    """
    views=select_reference_views(reference_ids,limit=limit)
    import bpy
    from mathutils import Vector
    from bpy_extras.object_utils import world_to_camera_view
    from .action4_blender_foundation import load_r4_core
    config_file=contained_path(PROJECT_ROOT,config_path,exists=True)
    cfg=json.loads(config_file.read_text())
    validate_recipe(cfg)
    source=contained_path(PROJECT_ROOT,cfg['foundation_config'],exists=True)
    binding=json.loads(source.read_text())['camera']
    for key in ('foundation','asset_manifest'):
        item=binding[key]; path=contained_path(PROJECT_ROOT,item['path'],exists=True)
        if digest(path)!=item['sha256']:raise ValueError('Frozen source SHA mismatch: '+key)
    out=contained_path(PROJECT_ROOT,cfg['preview_output'] if preview else cfg['output'])
    if out.exists():raise FileExistsError('Use a new bank output; refusing overwrite '+str(out))
    out.mkdir(parents=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene
    core=load_r4_core(binding['asset_dir'],binding['foundation']['path'],'Action4_reference_body',1)
    for obj in list(scene.objects):
        if obj!=core:obj.hide_render=True
    overrides=[]
    for slot in core.material_slots:
        mat=slot.material
        if not mat or not mat.use_nodes:continue
        photos=[n for n in mat.node_tree.nodes if n.type=='TEX_IMAGE' and n.image and 'rear_screen' in n.image.name]
        if photos:
            derived=mat.copy();slot.material=derived
            for n in derived.node_tree.nodes:
                if n.type=='TEX_IMAGE' and n.image and 'rear_screen' in n.image.name:
                    for socket in n.outputs:
                        for link in list(socket.links):
                            target=link.to_socket;derived.node_tree.links.remove(link)
                            if target.type=='RGBA':target.default_value=(.007,.011,.014,1.)
                            elif target.type=='VALUE':target.default_value=0.
            overrides.append(mat.name)
    if not overrides:raise ValueError('No rear display texture found; material semantics require review')
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=8 if preview else 128
    scene.cycles.use_denoising=True;scene.render.threads_mode='FIXED';scene.render.threads=4
    size=256 if preview else 1024
    scene.render.resolution_x=scene.render.resolution_y=size;scene.render.resolution_percentage=100
    scene.render.film_transparent=True
    scene.view_settings.view_transform='Standard';scene.view_settings.look='None'
    scene.view_settings.exposure=0;scene.view_settings.gamma=1
    scene.world=bpy.data.worlds.new('neutral_world');scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.15,.15,.15,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=.5
    camdata=bpy.data.cameras.new('reference_camera');cam=bpy.data.objects.new('reference_camera',camdata)
    scene.collection.objects.link(cam);scene.camera=cam;camdata.type='ORTHO';camdata.clip_start=.001;camdata.clip_end=10
    lightdata=bpy.data.lights.new('large_area','AREA');lightdata.energy=float(cfg.get('area_light_energy_w',30));lightdata.size=.3
    light=bpy.data.objects.new('large_area',lightdata);scene.collection.objects.link(light)
    light_mode=cfg.get('lighting_frame','world')
    if light_mode not in ('world','camera_azimuth'):raise ValueError('Unknown lighting_frame')
    fill_energy=float(cfg.get('camera_fill_energy_w',0))
    if not 0<=fill_energy<=.6:raise ValueError('Fill budget must be0..0.6W')
    fill=None
    if fill_energy:
        fd=bpy.data.lights.new('soft_camera_fill','AREA');fd.energy=fill_energy;fd.size=.4
        fill=bpy.data.objects.new('soft_camera_fill',fd);scene.collection.objects.link(fill)
    scene.use_nodes=True;nodes=scene.node_tree.nodes;nodes.clear()
    layers=nodes.new('CompositorNodeRLayers')
    composite=nodes.new('CompositorNodeComposite');scene.node_tree.links.new(layers.outputs['Image'],composite.inputs[0])
    rgb_outputs=[]
    for gray in cfg['gray_backgrounds']:
        # Alpha Over in scene-linear space; display transform applied once on save.
        c=gray/255.;linear=c/12.92 if c<=.04045 else ((c+.055)/1.055)**2.4
        alpha=nodes.new('CompositorNodeAlphaOver');alpha.inputs[0].default_value=1
        alpha.inputs[1].default_value=(linear,linear,linear,1)
        scene.node_tree.links.new(layers.outputs['Image'],alpha.inputs[2])
        output=nodes.new('CompositorNodeOutputFile');output.base_path=str(out/'rgb')
        output.format.file_format='PNG';output.format.color_mode='RGB';output.format.color_depth='8'
        scene.node_tree.links.new(alpha.outputs[0],output.inputs[0]);rgb_outputs.append((gray,output))
    records=[]
    for spec in views:
        az=math.radians(spec['azimuth']);el=math.radians(spec['elevation'])
        direction=Vector((math.sin(az)*math.cos(el),-math.cos(az)*math.cos(el),math.sin(el)))
        cam.location=direction*.3;cam.rotation_euler=(-cam.location).to_track_quat('-Z','Y').to_euler()
        camdata.ortho_scale=.1;bpy.context.view_layer.update()
        coords=[world_to_camera_view(scene,cam,core.matrix_world@Vector(c)) for c in core.bound_box]
        fraction=max(max(c.x for c in coords)-min(c.x for c in coords),max(c.y for c in coords)-min(c.y for c in coords))
        camdata.ortho_scale*=fraction/.75
        lx=.15 if spec['light'] else -.15;ly=-.15
        if light_mode=='camera_azimuth':
            light.location=(lx*math.cos(az)-ly*math.sin(az),lx*math.sin(az)+ly*math.cos(az),.25)
        else:light.location=(lx,ly,.25)
        light.rotation_euler=(-light.location).to_track_quat('-Z','Y').to_euler()
        if fill is not None:
            fill.location=direction*.25;fill.rotation_euler=(-fill.location).to_track_quat('-Z','Y').to_euler()
        rid=spec['reference_id']
        for gray,node in rgb_outputs:node.file_slots[0].path=f'{rid}_gray{gray}_'
        scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA';scene.render.image_settings.color_depth='8'
        (out/'rgba').mkdir(exist_ok=True);rgba=out/'rgba'/f'{rid}.png';scene.render.filepath=str(rgba)
        bpy.ops.render.render(write_still=True)
        # Reload saved straight-alpha PNG. Coverage is alpha, not object-index pass.
        image=bpy.data.images.load(str(rgba),check_existing=False)
        pixels=np.asarray(image.pixels[:],np.float32).reshape(size,size,4)[::-1]
        coverage=pixels[:,:,3].copy();bbox=mask_bbox(coverage)
        (out/'masks').mkdir(exist_ok=True);mp=out/'masks'/f'{rid}.npy';np.save(mp,coverage,allow_pickle=False)
        rgb=[]
        for gray,node in rgb_outputs:
            p=out/'rgb'/f'{rid}_gray{gray}_{scene.frame_current:04d}.png'
            if not p.is_file():raise ValueError('Missing compositor RGB')
            rgb.append(dict(gray=gray,path=str(p.relative_to(PROJECT_ROOT)),sha256=digest(p)))
        record=dict(**spec,source='blender_canonical_asset',preview=preview,rgba=str(rgba.relative_to(PROJECT_ROOT)),rgba_sha256=digest(rgba),
            mask=str(mp.relative_to(PROJECT_ROOT)),mask_sha256=digest(mp),bbox_xyxy=bbox,rgb=rgb,
            camera_matrix_world=[list(row) for row in cam.matrix_world],ortho_scale=camdata.ortho_scale,
            body_fraction=max(bbox[2]-bbox[0]-2,bbox[3]-bbox[1]-2)/size,
            resolution=size,samples=scene.cycles.samples,device='CPU',area_light_energy_w=lightdata.energy,lighting_frame=light_mode,main_light_location=list(light.location),main_light_size_m=lightdata.size,fill_energy_w=fill_energy,fill_location=list(fill.location) if fill else None,fill_size_m=fill.data.size if fill else None,alpha='PNG straight; scene-linear compositor AlphaOver',
            config_sha256=digest(config_file),generator_sha256=digest(Path(__file__)),mesh_sha256=digest(contained_path(PROJECT_ROOT,Path(binding['asset_dir'])/'mesh_v001.npz',exists=True)),
            view_transform=scene.view_settings.view_transform,look=scene.view_settings.look,exposure=scene.view_settings.exposure,
            material_overrides=overrides,hidden_decal=True,foundation=binding['foundation'],asset_manifest=binding['asset_manifest'])
        records.append(record)
        with (out/'manifest.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    return dict(output=str(out),rendered=len(records),preview=preview,formal_bank_complete=formal_bank_complete(cfg,records,preview=preview))
