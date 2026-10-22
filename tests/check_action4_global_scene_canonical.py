"""Minimal saved-scene GIS1 regression; no labels/render/GPU."""
import bpy,json,sys,copy
from pathlib import Path
R=Path('/data/zy/vln/a-test');sys.path.insert(0,str(R/'src'))
from data import action4_global_scene as m
O=R/'outputs/action4_realism_reassessment_v001/global_scene_init_v001/fix_v004'
c=json.loads((R/'data/action4_scene_assets/global_scene_init_v001/build_v004/integrated_contract.json').read_text())
bpy.ops.wm.open_mainfile(filepath=str(R/c['saved_asset']['path']),load_ui=False,use_scripts=False)
scene=bpy.context.scene;bpy.context.view_layer.update();m.verify(scene,c)
d=m.canonical_definition(c['config']);assert c['canonical8']==d
assert 'canonical8_from_frozen_core_bounds' not in c['original_invariants']
assert 'NOT training labels' in c['original_invariants']['diagnostic_envelope_purpose']
negatives=[]
for keys in [('corner_order','corners_3d_mm','corners_3d_m'),('corners_3d_m',)]:
 bad=copy.deepcopy(c);perm=[0,4,3,7,1,5,2,6]
 for k in keys:bad['canonical8'][k]=[d[k][i] for i in perm]
 assert sorted(map(tuple,bad['canonical8']['corners_3d_m']))==sorted(map(tuple,d['corners_3d_m']))
 try:m.verify(scene,bad)
 except ValueError as ex:negatives.append({'fields':keys,'same_unordered_points':True,'rejected':str(ex)})
 else:raise AssertionError('Ordered permutation accepted')
old=json.loads((R/'data/action4_scene_assets/global_scene_init_v001/build_v003/integrated_contract.json').read_text())
before=copy.deepcopy(old['original_invariants'])
before['diagnostic_unordered_envelope']=before.pop('canonical8_from_frozen_core_bounds')
before['diagnostic_envelope_purpose']=c['original_invariants']['diagnostic_envelope_purpose']
assert before==c['original_invariants']
assert old['assembly']['assembly_priors']==c['assembly']['assembly_priors']
assert old['config']['initial_priors']==c['config']['initial_priors']
assert m.env.sha(R/old['saved_asset']['path'])==old['saved_asset']['sha256']
assert all(m.env.sha(R/ref['path'])==ref['sha256'] for ref in c['config']['sources'].values())
result={'status':'PASS_SAVED_ORDERED_CANONICAL_GIS1','negative_cases':negatives,'source_manifest':d['source_manifest'],'corner_order':d['corner_order'],'millimetres_to_metres_factor':.001,'original_geometry_weights_bones_A_invariants_preserved':True,'assembly_priors_unchanged':True,'old_v003_asset_unchanged':True,'source_inputs_unchanged':True,'training_labels_generated':False,'dataset_generation_allowed':False,'render_calls':0,'gpu_used':False}
(O/'canonical_regression.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
