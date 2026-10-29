"""Render actual new-arm feasibility examples for root visual review only."""
from pathlib import Path
import json,os,sys,time,hashlib
import numpy as np
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))

def run(scene,sss_probe=False,normals_probe=False,albedo_probe=False,light_probe=False,pose_final=False):
    from data.action4_phrase2_acceleration import install
    install(__file__)
    from data.action4_coverage1000_arm_runtime import CoverageArmRuntime
    from data.action4_coverage1000_sampling import propose
    from data.action4_phrase2_geometry import LocalArmVisibility,junction_outside_frame
    from data.action4_phrase2_collision import Phrase2CollisionChecker
    from data.action4_phrase2_scene_visibility import SceneVisibility
    from data import action4_phrase2_pixels as pixels
    from data.action4_phrase2_surface import SurfaceDeriver
    from data.action4_coverage1000_contract import actual_accept
    cfg_path='configs/data/action4_coverage1000_arm_pose_light_v001.json' if pose_final else 'configs/data/action4_coverage1000_feasibility_arm_v001.json'
    cfg=json.loads((R/cfg_path).read_text())
    plan=json.loads((R/'outputs/coverage1000_v001/SLOTS_DRAFT.json').read_text());slots={s['slot_id']:s for s in plan['slots']}
    catalog=json.loads((R/cfg['existing_parameter_catalog']).read_text());rows=[]
    scan='feasibility_arm_pose_v001' if pose_final else 'feasibility_arm_v001'
    for p in sorted((R/'outputs/coverage1000_v001'/scan).glob(scene+'_*.json')):
        if p.name.endswith('_proposals.json'):continue
        for row in json.loads(p.read_text())['rows']:
            if row['accepted'] and slots[row['slot_id']]['split']=='train':rows.append(row)
    folder='visual_probe_pose_light' if pose_final else 'visual_probe_light' if light_probe else 'visual_probe_albedo' if albedo_probe else 'visual_probe_normals' if normals_probe else 'visual_probe_sss0' if sss_probe else 'visual_probe_v002'
    out=R/'outputs/coverage1000_v001/arm_repair_v001'/folder/scene;out.mkdir(parents=True,exist_ok=False)
    runtime=CoverageArmRuntime(scene,asset_version=cfg.get('arm_asset_version','v001'));trees={s:LocalArmVisibility(runtime,s) for s in ['left','right']}
    if light_probe:
        for obj in runtime.scene.objects:
            if obj.type=='LIGHT' and obj.data.type=='AREA':
                matrix=obj.matrix_world.copy();matrix.translation.z+=1.2;obj.matrix_world=matrix
                runtime.world[obj.name]=np.array(matrix);runtime.base_lights[obj.name]*=4.;obj.data.energy*=4.
    if albedo_probe:
        skin=runtime.scene.objects['left_forearm_natural_hand'].material_slots[0].material.copy();skin.name='coverage_diagnostic_albedo'
        nodes=skin.node_tree.nodes;links=skin.node_tree.links;shader=next(n for n in nodes if n.bl_idname=='ShaderNodeBsdfPrincipled');output=next(n for n in nodes if n.bl_idname=='ShaderNodeOutputMaterial')
        emission=nodes.new('ShaderNodeEmission');emission.inputs['Strength'].default_value=1.
        colorlink=next(l for l in links if l.to_node==shader and l.to_socket.name=='Base Color')
        links.new(colorlink.from_socket,emission.inputs['Color'])
        for link in list(links):
            if link.to_node==output and link.to_socket.name=='Surface':links.remove(link)
        links.new(emission.outputs['Emission'],output.inputs['Surface'])
        for side in ['left','right']:
            for suffix in ['forearm_natural_hand','forearm_continuation']:runtime.scene.objects[side+'_'+suffix].material_slots[0].material=skin
    if normals_probe:
        import bpy
        for side in ['left','right']:
            obj=runtime.scene.objects[side+'_forearm_continuation'];hand=runtime.scene.objects[side+'_forearm_natural_hand'];old=obj.data
            positions=np.array([v.co[:] for v in old.vertices]);faces=[list(f.vertices) for f in old.polygons]
            mesh=bpy.data.meshes.new('coverage_normal_probe_'+side);mesh.from_pydata(positions.tolist(),[],faces);mesh.update()
            mesh.materials.append(hand.material_slots[0].material);mesh.use_auto_texspace=False;mesh.texspace_location=hand.data.texspace_location;mesh.texspace_size=hand.data.texspace_size
            for face in mesh.polygons:face.use_smooth=True
            mesh.use_auto_smooth=True;mesh.auto_smooth_angle=np.pi;mesh.calc_normals_split();hand.data.calc_normals_split()
            hv=np.array([v.co[:] for v in hand.data.vertices]);mapping=np.linalg.norm(positions[:44,None]-hv[None],axis=-1).argmin(1)
            source={int(v):[] for v in mapping}
            for loop in hand.data.loops:
                if loop.vertex_index in source:source[loop.vertex_index].append(np.array(loop.normal))
            rim={i:np.mean(source[int(v)],axis=0) for i,v in enumerate(mapping)};rim={i:n/np.linalg.norm(n) for i,n in rim.items()}
            centers=positions.reshape(24,44,3).mean(1);s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(centers,axis=0),axis=1))]
            normals=[]
            for loop in mesh.loops:
                i,j=divmod(loop.vertex_index,44);weight=max(0.,1-s[i]/.06)**2
                n=(1-weight)*np.array(loop.normal)+weight*rim[j];normals.append((n/np.linalg.norm(n)).tolist())
            mesh.normals_split_custom_set(normals);obj.data=mesh
    if sss_probe:
        skin=runtime.scene.objects['left_forearm_natural_hand'].data.materials[0].copy();skin.name='coverage_diagnostic_skin_sss0'
        for node in skin.node_tree.nodes:
            if node.bl_idname=='ShaderNodeBsdfPrincipled':node.inputs['Subsurface'].default_value=0.
        for side in ['left','right']:
            for suffix in ['forearm_natural_hand','forearm_continuation']:runtime.scene.objects[side+'_'+suffix].data.materials[0]=skin
    checker=Phrase2CollisionChecker(runtime);visibility=SceneVisibility(runtime,trees);deriver=SurfaceDeriver();canonical=runtime.metadata['canonical_definition']
    ids={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in pixels.REQUIRED_SOURCE_PATHS};ids[canonical['source_manifest']['path']]=canonical['source_manifest']['sha256']
    ids[runtime.coverage_overlay_ref['path']]=runtime.coverage_overlay_ref['sha256'];ids[runtime.metadata_path]=hashlib.sha256((R/runtime.metadata_path).read_bytes()).hexdigest()
    selected=[]
    for group in (['C'] if sss_probe or normals_probe or albedo_probe or light_probe else ['C','E','A']):
        candidates=sorted([row for row in rows if row['group']==group],key=lambda row:(row.get('small_width_band') or 0,row.get('high_band') or 0))
        found=None
        for row in candidates:
            slot=slots[row['slot_id']];proposal,error=propose(runtime,cfg,slot,row['attempt'],catalog,local_visibility=None)
            if error:raise ValueError('Same-seed admitted geometry no longer reproducible: '+error)
            if proposal['phrase2']['pair_metrics']!=row['pair']:raise ValueError('Same-seed pair geometry differs')
            size=np.array([cfg['width'],cfg['height']])
            elbows={s:bool(junction_outside_frame(runtime,s,np.array(proposal['actor_pose_capture_by_side'][s]['brace_B_capture']),size)) for s in ['left','right']}
            if group in ['C','E'] and all(elbows.values()):continue
            runtime.apply(proposal);collision=checker.check()
            if not collision['accepted']:raise ValueError('Reproduced actual collision differs')
            found=(slot,proposal,elbows,collision);break
        if found:selected.append(found)
    if not selected:raise ValueError('No actual accepted new-arm poses available for visual probe')
    receipts=[]
    for n,(slot,proposal,elbows,collision) in enumerate(selected):
        runtime.apply(proposal);sources=visibility.measure(proposal);actual=runtime.export_corners()
        entry=proposal['entry'].copy();entry['sample_id']='arm_probe_'+entry['sample_id']
        root=out/f'{n:02d}_{slot["group"]}';root.mkdir();(root/'pixels').mkdir();(root/'runtime').mkdir()
        (root/'FRAME_SNAPSHOT.json').write_text(json.dumps({'slot':slot,'proposal':proposal,'sources':sources,'collision':collision,'actual':actual,'elbows_outside':elbows,'diagnostic_shader_sss_zero':sss_probe},indent=2))
        settings={'device':'CUDA','denoiser':'OPTIX','samples':32,'mask_samples':1,'amodal_mode':'crop','amodal_margin_px':16,'seed':entry['seed'],'worker_gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']}
        begin=time.monotonic();result=pixels.render_frame(scene=runtime.scene,entry=entry,targets=runtime.targets(),canonical=canonical,K=runtime.K.tolist(),data_out=root/'pixels',runtime_out=root/'runtime',output_role='diagnostic',source_identities=ids,render_settings=settings,verify_scene=runtime.verify)
        record=result['record'];record['occlusion_sources']=sources
        ok,why,stats=actual_accept(slot,record)
        record=deriver.derive(record,root/'geometry')
        (root/'RECORD.json').write_text(json.dumps(record));receipts.append({'slot':slot,'image':record['image'],'record':str((root/'RECORD.json').relative_to(R)),
            'snapshot':str((root/'FRAME_SNAPSHOT.json').relative_to(R)),'actual_quota_provisional_pass':ok,'actual_quota_reason':why,'measurements':stats,'pending_host_png_and_surface_validation':True,
            'elbows_outside':elbows,'seconds':time.monotonic()-begin,'diagnostic_not_formal128':True})
        (out/'PROGRESS.json').write_text(json.dumps(receipts))
    (out/'COMPLETE.json').write_text(json.dumps({'status':'ARM_RGB_REVIEW_REQUIRED','receipts':receipts,'asset_overlay':runtime.coverage_overlay_ref},indent=2));print(json.dumps(receipts),flush=True)

if __name__=='__main__':
    args=sys.argv[sys.argv.index('--')+1:];run(args[0],'sss0' in args[1:],'normals' in args[1:],'albedo' in args[1:],'light' in args[1:],'pose_final' in args[1:])
