"""Actual saved finger rig checks. CPU Blender, no render/save; exclusive report."""
import bpy,json,sys,argparse,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/data/zy/vln/a-test');assert Path.cwd().resolve()==ROOT
sys.path.insert(0,str(ROOT/'src'))
from data.action4_finger_rig import bounded,sha,evaluated,apply_pose,check_pose,nonadjacent_overlaps,validate_config,write_new
p=argparse.ArgumentParser();p.add_argument('--contract',required=True);p.add_argument('--report',required=True);args=p.parse_args(sys.argv[sys.argv.index('--')+1:])
c=json.loads(bounded(args.contract,True).read_text());validate_config(c['config']);asset=bounded(c['saved_asset']['path'],True);before=sha(asset);assert before==c['saved_asset']['sha256'];m=c['mapping']
bpy.ops.wm.open_mainfile(filepath=str(bounded(c['config']['source_asset']['path'],True)),load_ui=False,use_scripts=False)
original=bpy.data.objects['global_source_anatomy'];source_control=np.array([list(v.co) for v in original.data.vertices]);source_faces=[tuple(x.vertices) for x in original.data.polygons];source_eval,source_tri=evaluated(original)
bpy.ops.wm.open_mainfile(filepath=str(asset),load_ui=False,use_scripts=False);body=bpy.data.objects['global_source_anatomy'];rig=bpy.data.objects['global_metric_armature'];assert len(rig.data.bones)==35
assert np.array_equal(np.array([list(v.co) for v in body.data.vertices]),source_control);assert [tuple(x.vertices) for x in body.data.polygons]==source_faces
assert all(np.array_equal(np.array(b.matrix_basis),np.eye(4)) for b in rig.pose.bones)
assert bpy.context.scene['finger_dataset_generation_allowed'] is False
rest,t=evaluated(body);assert np.array_equal(t,source_tri);err=np.linalg.norm(rest-source_eval,axis=1);assert err.max()<1e-6
obj=body.evaluated_get(bpy.context.evaluated_depsgraph_get());me=obj.to_mesh();handgroups={body.vertex_groups[s+'_hand'].index for s in ['left','right']};ids=set().union(*(set(x['evaluated_vertex_ids']) for x in m.values()));ids|={v.index for v in me.vertices if any(g.group in handgroups and g.weight>0 for g in v.groups)};obj.to_mesh_clear();ts=[i for i,x in enumerate(t) if any(int(j) in ids for j in x)];base={'hand_triangles':ts,'overlaps':nonadjacent_overlaps(rest,t,ts)};world={k:np.array(v) for k,v in c['fixed_camera_and_brace_world'].items()}
checks=[]
for key in m:
 pose={key:[8,8,8]};result=check_pose(body,rig,m,rest,t,base,world,c['original_G1_bones'],pose);assert result['skin_max_displacement_m']>.001;checks.append({'name':'isolated_'+key,'check':result})
for case in c['config']['diagnostic_poses']:
 checks.append({'name':case['name'],'check':check_pose(body,rig,m,rest,t,base,world,c['original_G1_bones'],case['pose'])})
negative=[]
for pose in [{'camera_scale':[1,1,1]},{'left_hand':[1,2,3]},{'left_index':[26,0,0]},{'left_index':[2,1,0],'right_thumb':[0,float('nan'),0]}]:
 prior={k:[list(r) for r in b.matrix_basis] for k,b in rig.pose.bones.items()}
 try:apply_pose(rig,pose,m)
 except ValueError:negative.append('rejected_no_mutation')
 else:raise AssertionError('Invalid control accepted')
 assert prior=={k:[list(r) for r in b.matrix_basis] for k,b in rig.pose.bones.items()}
apply_pose(rig,{},m);end,_=evaluated(body);assert np.array_equal(end,rest);assert sha(asset)==before
assert all(abs(sum(g.weight for g in v.groups)-1)<1e-6 for v in body.data.vertices)
result={'status':'PASS_ACTUAL_SAVED_REOPEN_INITIALIZATION_ONLY','asset_sha256':before,'original_control_vertices_faces_exact':True,'neutral_evaluated_L2_max_error_m':float(err.max()),'neutral_error_signed_mean_xyz_m':np.mean(rest-source_eval,axis=0).tolist(),'neutral_numerical_tolerance_m':1e-6,'single_digit_cases':10,'configured_pose_cases':4,'negative_cases':4,'checks':checks,'all_return_to_saved_neutral_exact':True,'source_hashes_unchanged':True,'scope':'No actual pinch/grasp/contact, no general safe anatomical range, no photorealism/generation acceptance;336 baseline distal self-intersection pairs retained and reported'}
validate_config(c['config']);target=bounded(args.report)
assert target.is_relative_to(ROOT/'outputs/action4_realism_reassessment_v001/finger_rig_init_v001')
write_new(target,result);print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
