"""Reusable complete-scene adapter. Imports have no Blender or render side effects.

Applies v005 geometry to existing complete templates. Actual scene exports are
independent of numerical proposal points. This module does not accept datasets.
"""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path('/data/zy/vln/a-test')
TEMPLATES={'wall':'wall','table_full_garment':'table_v002','cloth_lift10_full_garment':'cloth_v002'}
CLOTH_MATERIAL='wearing_garment_table_garment_denim'
LAYOUT='configs/data/action4_template_environment_v001.json'

def bounded(value):
    p=Path(value);p=(p if p.is_absolute() else ROOT/p).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Project boundary')
    return p

def read(value):return json.loads(bounded(value).read_text())
def verified_read(ref):
    raw=bounded(ref['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('Changed template source: '+ref['path'])
    return json.loads(raw)


def validate_proposal_binding(proposal,metadata,anchor,name):
    from data import action4_parameter_sampling as sampler
    if proposal.get('parameter_contract_revision')!='broad_joint_v005' or proposal['background_request']['template']!=name:raise ValueError('Explicit matching template and parameter revision required')
    provenance=proposal['provenance'];canonical=metadata['canonical_definition']
    if provenance['anchor_ref']!=metadata['baseline_parameters_ref'] or provenance['anchor_parameter_digest']!=sampler.digest(anchor):raise ValueError('Proposal anchor identity differs from verified template')
    if proposal['canonical_source_ref']!=canonical['source_manifest'] or proposal['canonical8_m']!=canonical['corners_3d_m'] or proposal['corner_ids']!=list(range(8)):raise ValueError('Proposal immutable canonical8 differs')
    if proposal['K_native']!=metadata['K_native'] or proposal['W_from_capture']!=metadata['W_from_capture']:raise ValueError('Proposal capture frame differs')
    if proposal['target_presence']!=proposal['entry']['target_presence'] or proposal['entry']['target_count']!=sum(proposal['target_presence']):raise ValueError('Proposal presence mismatch')
    if proposal['entry']['split']=='test' and proposal['target_presence']!=[True,True]:raise ValueError('Test requires both cameras')
    for key in ('area_light_gain','exposure_offset_ev'):
        value=proposal['environment_controls'][key]
        if type(value) not in (int,float) or not np.isfinite(value) or key=='area_light_gain' and value<=0:raise ValueError('Finite valid photometry required')
    if set(proposal['occluder_proposals'])!={side for side,present in zip(('left','right'),proposal['target_presence']) if present}:raise ValueError('Exact present occluder proposal identities required')
    for i,side in enumerate(('left','right')):
        declared=proposal['sides'][side];fixed=anchor['sides'][side]
        for key in ('A_brace_from_CAD','S_skin_relative_to_brace','F_body_from_brace'):
            if declared[key]!=fixed[key]:raise ValueError('Frozen installation differs: '+side+'/'+key)
        pose=proposal['actor_pose_capture_by_side'][side];B=sampler.rigid(pose['brace_B_capture']);C=sampler.rigid(pose['core_C_capture'])
        if not np.allclose(B@np.array(fixed['A_brace_from_CAD']),C,atol=1e-9,rtol=0):raise ValueError('B/A/C mismatch')
        source=anchor['actor_pose_capture_by_side'][side];Bbase=np.linalg.inv(np.array(source['whole_actor_delta_capture']))@np.array(source['brace_B_capture'])
        D=sampler.rigid(pose['whole_actor_delta_capture'])
        if not np.allclose(D,B@np.linalg.inv(Bbase),atol=1e-9,rtol=0):raise ValueError('Whole actor delta inconsistent with verified base')
        if not np.allclose(metadata['actor_groups'][side]['base_brace_capture_m'],source['brace_B_capture'],atol=1e-9,rtol=0):raise ValueError('Template base is not verified anchor brace')
        if declared['present']!=proposal['target_presence'][i]:raise ValueError('Side presence mismatch')
        for key in ('brace_B_capture','core_C_capture','whole_actor_delta_capture'):
            if declared[key]!=(pose[key] if declared['present'] else None):raise ValueError('Duplicated actor pose disagrees')
        if declared['present']:
            metric=proposal['derived']['by_side'][side];native=sampler.array(metric['canonical8_native_px'],(8,2));capture=sampler.array(metric['canonical8_capture_m'],(8,3))
            expected=np.array(canonical['corners_3d_m'])@C[:3,:3].T+C[:3,3]
            if not np.allclose(capture,expected,atol=1e-9,rtol=0):raise ValueError('Derived capture corners differ from pose')
            if np.any(capture[:,2]<=metadata['near_plane_m']):raise ValueError('Nonprojectable target corners')
            hom=capture@np.array(metadata['K_native']).T
            if not np.allclose(native,hom[:,:2]/hom[:,2,None],atol=1e-6,rtol=0):raise ValueError('Derived native corners differ from pose')
            spec=proposal['occluder_proposals'][side]
            if spec['status']!='GEOMETRY_PROPOSAL_RENDER_APPLICATION_PENDING':raise ValueError('Present target requires complete occluder geometry')
            world=sampler.array(spec['vertices_world_m'],(4,3));v=sampler.array(spec['vertices_capture_m'],(4,3));uv=sampler.array(spec['vertices_native_px'],(4,2));W=np.array(metadata['W_from_capture'])
            if np.any(v[:,2]<=metadata['near_plane_m']) or not np.allclose(world,v@W[:3,:3].T+W[:3,3],rtol=0,atol=1e-9):raise ValueError('Invalid occluder world/capture geometry')
            hom=v@np.array(metadata['K_native']).T
            if not np.allclose(uv,hom[:,:2]/hom[:,2,None],rtol=0,atol=1e-6):raise ValueError('Invalid occluder projection')
    return True

class TemplateRuntime:
    def __init__(self,name):
        import bpy
        from data import action4_modular_render as camera
        if name not in TEMPLATES:raise ValueError('Known existing template required')
        self.name=name;self.metadata_path='data/action4_scene_templates/v007_candidate/'+TEMPLATES[name]+'/template.json'
        self.metadata=read(self.metadata_path);self.anchor=verified_read(self.metadata['baseline_parameters_ref'])
        blend=bounded(self.metadata['blend_ref']['path'])
        if hashlib.sha256(blend.read_bytes()).hexdigest()!=self.metadata['blend_ref']['sha256']:raise ValueError('Changed complete template')
        bpy.ops.wm.open_mainfile(filepath=str(blend));self.scene=bpy.context.scene
        if self.scene.camera.name!=self.metadata['camera_name']:raise ValueError('Template capture camera changed')
        for image in bpy.data.images:
            if image.source=='FILE' and not image.packed_file and image.filepath:bounded(bpy.path.abspath(image.filepath))
        for lib in bpy.data.libraries:bounded(bpy.path.abspath(lib.filepath))
        if not np.allclose(camera.actual_K(self.scene),self.metadata['K_native'],rtol=0,atol=1e-6):raise ValueError('Template intrinsics changed')
        self.K=np.asarray(self.metadata['K_native']);self.W=np.asarray(self.metadata['W_from_capture'])
        self.layout=read(LAYOUT)
        from mathutils import Matrix
        wall=self.scene.objects[self.layout['wall']['object']]
        delta=np.eye(4);delta[:3,3]=self.W[:3,:3]@np.array(self.layout['wall']['translation_capture_m'])
        wall.matrix_world=Matrix((delta@np.array(wall.matrix_world)).tolist());wall.dimensions=self.layout['wall']['dimensions_world_m']
        group=self.layout['templates'][name];delta=np.eye(4);delta[:3,3]=self.W[:3,:3]@np.array(group['translation_capture_m'])
        for objname in group['roots']:
            obj=self.scene.objects[objname];obj.matrix_world=Matrix((delta@np.array(obj.matrix_world)).tolist())
        bpy.context.view_layer.update()
        self.world={o.name:np.array(o.matrix_world) for o in self.scene.objects}
        self.parents={o.name:o.parent.name if o.parent else None for o in self.scene.objects}
        self.hide={o.name:bool(o.hide_render) for o in self.scene.objects}
        self.pass_indices={o.name:int(o.pass_index) for o in self.scene.objects}
        self.bindings=None
        parent_ref=self.metadata['parent_manifest_ref'];parent_bytes=bounded(parent_ref['path']).read_bytes()
        if hashlib.sha256(parent_bytes).hexdigest()!=parent_ref['sha256']:raise ValueError('Changed template presence source')
        if parent_bytes:
            for line in parent_bytes.splitlines():
                row=json.loads(line)
                if row['sample_id']==self.metadata['sample_id']:
                    self.bindings=row['modular_backend']['scene_contract']['actor_bindings'];break
        if self.bindings is None:raise ValueError('Missing original presence bindings')
        del parent_bytes
        for side,group in self.metadata['actor_groups'].items():
            if not set(self.bindings[side]['hidden_when_absent'])<=set(group['objects']):raise ValueError('Presence binding outside actor')
            for name in group['objects']:
                if name not in self.scene.objects:raise ValueError('Missing actor object')
            for name in group['roots']:
                if self.parents[name] in group['objects']:raise ValueError('Actor root is not top-level')
        self.base_lights={o.name:float(o.data.energy)/self.anchor['environment_controls']['area_light_gain'] for o in self.scene.objects if o.type=='LIGHT'}
        self.base_exposure=float(self.scene.view_settings.exposure)-self.anchor['environment_controls']['exposure_offset_ev']
        self.additional_source_refs=[]
        material=bpy.data.materials.get(CLOTH_MATERIAL)
        if material is None:
            material_metadata='data/action4_scene_templates/v007_candidate/table_v002/template.json'
            metadata_bytes=bounded(material_metadata).read_bytes();source=json.loads(metadata_bytes)
            self.additional_source_refs=[{'path':material_metadata,'sha256':hashlib.sha256(metadata_bytes).hexdigest()},source['blend_ref']]
            asset=bounded(source['blend_ref']['path'])
            if hashlib.sha256(asset.read_bytes()).hexdigest()!=source['blend_ref']['sha256']:raise ValueError('Changed cloth material source')
            with bpy.data.libraries.load(str(asset),link=False) as (available,selected):
                if CLOTH_MATERIAL not in available.materials:raise ValueError('Verified cloth material missing')
                selected.materials=[CLOTH_MATERIAL]
            material=selected.materials[0]
        self.patches={}
        for side in ('left','right'):
            mesh=bpy.data.meshes.new('random_cloth_'+side+'_mesh');mesh.from_pydata([(0,0,0)]*4,[],[(0,1,2,3)]);mesh.update()
            uv=mesh.uv_layers.new(name='UVMap')
            for loop,coord in zip(uv.data,[(0,0),(1,0),(1,1),(0,1)]):loop.uv=coord
            obj=bpy.data.objects.new('random_cloth_'+side,mesh);self.scene.collection.objects.link(obj);obj.data.materials.append(material)
            obj.hide_render=True;obj.pass_index=0;self.patches[side]=obj
        self.current=None;self.expected=None

    def apply(self,proposal):
        import bpy
        from mathutils import Matrix
        validate_proposal_binding(proposal,self.metadata,self.anchor,self.name)
        if not np.array_equal(proposal['K_native'],self.K) or not np.array_equal(proposal['W_from_capture'],self.W):raise ValueError('Camera calibration is frozen')
        for name in self.world:
            obj=self.scene.objects[name];obj.hide_render=self.hide[name];obj.pass_index=self.pass_indices[name]
        expected={name:matrix.copy() for name,matrix in self.world.items()}
        for i,side in enumerate(('left','right')):
            group=self.metadata['actor_groups'][side]
            B=np.array(proposal['actor_pose_capture_by_side'][side]['brace_B_capture'])
            delta=self.W@B@np.linalg.inv(self.W@np.array(group['base_brace_capture_m']))
            for name in group['roots']:self.scene.objects[name].matrix_world=Matrix((delta@self.world[name]).tolist())
            for name in group['objects']:expected[name]=delta@self.world[name]
            if not proposal['target_presence'][i]:
                for name in self.bindings[side]['hidden_when_absent']:
                    self.scene.objects[name].hide_render=True;self.scene.objects[name].pass_index=0
            for name in self.bindings[side].get('always_hidden',[]):self.scene.objects[name].hide_render=True
            patch=self.patches[side];spec=proposal['occluder_proposals'].get(side)
            patch.hide_render=spec is None or spec.get('status')!='GEOMETRY_PROPOSAL_RENDER_APPLICATION_PENDING'
            if not patch.hide_render:
                for vertex,position in zip(patch.data.vertices,spec['vertices_world_m']):vertex.co=position
                patch.data.update()
        for name,value in self.base_lights.items():self.scene.objects[name].data.energy=value*proposal['environment_controls']['area_light_gain']
        self.scene.view_settings.exposure=self.base_exposure+proposal['environment_controls']['exposure_offset_ev']
        bpy.context.view_layer.update();self.current=proposal;self.expected=expected
        self.expected_flags={o.name:(bool(o.hide_render),int(o.pass_index)) for o in self.scene.objects}
        self.verify(self.scene)
        return self.targets()

    def targets(self):
        from data import action4_modular_pixels as pixels
        return {i+1:{'core':self.scene.objects[pixels.KNOWN_TARGET_PARTS[i+1]['core']],
                        'parts':[self.scene.objects[name] for name in pixels.KNOWN_TARGET_PARTS[i+1].values()]}
                for i,present in enumerate(self.current['target_presence']) if present}

    def verify(self,scene):
        from data import action4_modular_render as camera
        if scene is not self.scene or self.current is None:raise ValueError('No applied template state')
        if not np.allclose(camera.actual_K(scene),self.K,rtol=0,atol=1e-6):raise ValueError('Actual K changed')
        if not np.allclose(np.array(scene.camera.matrix_world),self.world[self.metadata['camera_name']],rtol=0,atol=2e-6):raise ValueError('Capture pose changed')
        if set(scene.objects.keys())!=set(self.world)|{p.name for p in self.patches.values()}:raise ValueError('Unexpected scene objects')
        for obj in scene.objects:
            if (bool(obj.hide_render),int(obj.pass_index))!=self.expected_flags[obj.name]:raise ValueError('Applied visibility or instance index changed: '+obj.name)
        for name,expected in self.expected.items():
            obj=scene.objects[name]
            if (obj.parent.name if obj.parent else None)!=self.parents[name]:raise ValueError('Actor hierarchy changed')
            if not np.allclose(np.array(obj.matrix_world),expected,rtol=0,atol=3e-6):raise ValueError('Actual actor/static transform differs: '+name)
        for side,patch in self.patches.items():
            spec=self.current['occluder_proposals'].get(side)
            if spec and spec.get('vertices_world_m') is not None:
                if not np.allclose([v.co[:] for v in patch.data.vertices],spec['vertices_world_m'],atol=2e-6,rtol=0):raise ValueError('Occluder vertices not applied')
        for name,base in self.base_lights.items():
            if not np.isclose(scene.objects[name].data.energy,base*self.current['environment_controls']['area_light_gain'],rtol=1e-6,atol=1e-5):raise ValueError('Light parameter not applied')
        if abs(scene.view_settings.exposure-self.base_exposure-self.current['environment_controls']['exposure_offset_ev'])>1e-6:raise ValueError('Exposure parameter not applied')
        return {'status':'ACTUAL_TRANSFORMS_MATCH','collision_checked':False}

    def export_corners(self):
        import bpy
        from data import action4_global_render as renderer
        from data import action4_canonical_export as exporter
        canonical=self.metadata['canonical_definition'];corners=np.array(canonical['corners_3d_m']);exports={};error=0.
        for ident,target in self.targets().items():
            points,visible,rays=renderer.project_canonical(self.scene,target['core'],corners,return_ray_clear=True)
            kwargs=dict(canonical=canonical,instance_id=ident,
                object_to_world_m=[list(row) for row in target['core'].evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world],
                camera_to_world_blender_m=[list(row) for row in self.scene.camera.matrix_world],K=self.K.tolist(),
                image_size_wh=[self.scene.render.resolution_x,self.scene.render.resolution_y],near_plane_m=self.scene.camera.data.clip_start,
                ray_clear_to_virtual_point=rays,legacy_corners_px=points,legacy_corner_visible=visible)
            value=exporter.build_export(**kwargs);exporter.verify_export(value,**kwargs)
            side=('left','right')[ident-1];expected=np.array(self.current['derived']['by_side'][side]['canonical8_native_px'])
            actual=np.asarray(points,float)
            if actual.shape!=(8,2) or expected.shape!=(8,2) or not np.isfinite(actual).all() or not np.isfinite(expected).all():raise ValueError('Nonfinite actual/proposed corners')
            current_error=float(np.max(np.abs(actual-expected)))
            if not np.isfinite(current_error):raise ValueError('Nonfinite projection error')
            error=max(error,current_error);exports[str(ident)]=value
        if error>.005:raise ValueError('Actual Blender/numerical projection differs: '+str(error))
        return {'canonical8_by_instance':exports,'max_proposal_vs_actual_error_px':error,'collision_checked':False,'training_accepted':False}
