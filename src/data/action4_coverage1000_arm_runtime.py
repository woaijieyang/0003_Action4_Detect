"""Admitted template runtime + one-time versioned upper-arm overlay.

The base runtime and its frozen assets are unchanged. Mesh replacement occurs
once before building collision/visibility caches, never during frame sampling.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from data.action4_phrase2_runtime import Phrase2Runtime
from data.action4_paths import PROJECT_ROOT as R,contained_path

class CoverageArmRuntime(Phrase2Runtime):
    def __init__(self,name,require_visual_admission=False,asset_version='v001'):
        import bpy
        from data.action4_modular_assembly import world_geometry
        super().__init__(name)
        if asset_version not in ['v001','v002']:raise ValueError('Unknown coverage arm asset version')
        root=R/('data/action4_scene_templates/coverage1000_arm_'+asset_version)
        report=json.loads((root/'ASSET_GEOMETRY.json').read_text())
        if report['status']!='GEOMETRY_PASS_VISUAL_REVIEW_REQUIRED':raise ValueError('Missing static geometry proof')
        meta_path=root/name/'template.json';meta=json.loads(meta_path.read_text())
        if meta['coverage_arm_parent_metadata']!=self.metadata_path or meta['blend_ref']!=self.metadata['blend_ref']:raise ValueError('Overlay base scene differs')
        for key in ['canonical_definition','K_native','W_from_capture','actor_groups','baseline_parameters_ref']:
            if meta[key]!=self.metadata[key]:raise ValueError('Overlay modified a frozen body/camera/mount field: '+key)
        ref=meta['coverage_arm_overlay_ref'];path=contained_path(R,ref['path'],exists=True)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=ref['sha256']:raise ValueError('Overlay identity changed')
        if require_visual_admission:
            admission=json.loads((R/'outputs/coverage1000_v001/arm_repair_v001/ROOT_ARM_VISUAL_REVIEW.json').read_text())
            if (admission.get('status')!='PASS_ARM_ASSET_VISUAL_REVIEW' or admission.get('overlay_sha256_by_scene',{}).get(name)!=ref['sha256']
                or admission.get('metadata_sha256_by_scene',{}).get(name)!=hashlib.sha256(meta_path.read_bytes()).hexdigest()):raise ValueError('Arm asset not visually admitted')
        with np.load(path) as overlay:
            for side in ['left','right']:
                obj=self.scene.objects[side+'_forearm_continuation']
                old=np.array([v.co[:] for v in obj.data.vertices])
                if not np.array_equal(old,overlay[side+'_source_vertices_local_m']):raise ValueError('Overlay belongs to different original arm')
                mesh=obj.data.copy()
                for vertex,value in zip(mesh.vertices,overlay[side+'_vertices_local_m']):vertex.co=value
                mesh.update();mesh.use_auto_smooth=True;mesh.normals_split_custom_set(overlay[side+'_loop_normals'].tolist());obj.data=mesh
        bpy.context.view_layer.update()
        self.metadata=meta;self.metadata_path=str(meta_path.relative_to(R));self.arm_repairs=meta['arm_repairs']
        profile=meta.get('coverage_light_profile')
        if profile:
            for obj in self.scene.objects:
                if obj.type=='LIGHT' and obj.data.type=='AREA':
                    matrix=obj.matrix_world.copy();matrix.translation.z+=profile['world_z_offset_m'];obj.matrix_world=matrix
                    self.world[obj.name]=np.array(matrix);self.base_lights[obj.name]*=profile['energy_multiplier'];obj.data.energy*=profile['energy_multiplier']
            bpy.context.view_layer.update()
        for side in ['left','right']:
            inverse=np.linalg.inv(self.W@np.array(meta['actor_groups'][side]['base_brace_capture_m']))
            obj=self.scene.objects[side+'_forearm_continuation'];local_to_B=inverse@np.array(obj.matrix_world)
            vertices=np.array([v.co[:] for v in obj.data.vertices]);ids=self.arm_repairs[side]['terminal_vertex_indices']
            self.arm_terminal_points_B[side]=vertices[ids]@local_to_B[:3,:3].T+local_to_B[:3,3]
            self.arm_terminal_B[side]=self.arm_terminal_points_B[side].mean(0)
            self.upperarm_points_B[side]=vertices@local_to_B[:3,:3].T+local_to_B[:3,3]
            points=[]
            for object_name in meta['actor_groups'][side]['objects']:
                actor=self.scene.objects[object_name]
                if actor.type not in ['MESH','CURVE'] or actor.hide_render or object_name in self.bindings[side].get('always_hidden',[]):continue
                v,_=world_geometry(actor);points.extend((v@inverse[:3,:3].T+inverse[:3,3]).tolist())
            self.local_actor[side]=np.asarray(points)
        # Semantic replacement of the old "hide every elbow" heuristic is
        # allowed only for this geometry-certified, immutable continuous mesh.
        # Terminal out-of-frame, all contacts, masks and source rules remain.
        self.coverage_continuous_junction_geometry=True
        self.coverage_overlay_ref=ref
        self.coverage_asset_version=asset_version
