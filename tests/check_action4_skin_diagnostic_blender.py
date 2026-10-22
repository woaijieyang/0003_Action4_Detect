"""Actual CPU scene/strict-invariant/failure checks; never render or save assets."""
import bpy
import contextlib
import json
import sys
from pathlib import Path
R = Path('/data/zy/vln/a-test'); sys.path.insert(0, str(R / 'src'))
from data import action4_skin_diagnostic as m
O = m.OUTPUTS / 'implementation_v001'
cfg = json.loads((R / 'configs/data/action4_skin_diagnostic_v001.json').read_text())
preview, contract, shot, evidence = m.validate_config(cfg)
source_before = {ref['path']: m.env.sha(ref['path']) for ref in cfg['sources'].values()}
asset = preview['sources']['integrated_asset']
bpy.ops.wm.open_mainfile(filepath=str(R / asset['path']), load_ui=False, use_scripts=False)
scene = bpy.context.scene; bpy.context.view_layer.update(); m.integrated.verify(scene, contract)
full_before = m.comparison_state(scene)
# Strict original invariant must reject the controlled temporary variant.
with m.constant_photo_input(scene, contract, cfg['sources']['photo'], evidence['linear_rgb_mean_float64']) as (receipt, expected):
    try: m.integrated.verify(scene, contract)
    except ValueError as exc: strict_rejection = str(exc)
    else: raise AssertionError('Original verification was improperly relaxed')
assert m.comparison_state(scene) == full_before
m.integrated.verify(scene, contract)
# Invalid means fail before unlinking anything.
for bad in ([float('nan'), .1, .1], [1.1, .1, .1], [.1, .1]):
    try:
        with m.constant_photo_input(scene, contract, cfg['sources']['photo'], bad): pass
    except ValueError: pass
    else: raise AssertionError('Invalid mean accepted')
    assert m.comparison_state(scene) == full_before
# Inject failure after the actual input replacement inside the real CPU orchestration.
real_context = m.constant_photo_input
@contextlib.contextmanager
def injected(*args):
    with real_context(*args) as state:
        raise RuntimeError('INJECTED_AFTER_SKIN_INPUT_REPLACEMENT')
        yield state
failure_dir = m.OUTPUTS / 'failure_probe_v001'; failure_dir.mkdir(exist_ok=False)
m.constant_photo_input = injected
try:
    try: m.execute_blender('configs/data/action4_skin_diagnostic_v001.json', failure_dir, 2, probe_only=True)
    except RuntimeError as exc: assert str(exc) == 'INJECTED_AFTER_SKIN_INPUT_REPLACEMENT'
    else: raise AssertionError('Injected failure did not propagate')
finally: m.constant_photo_input = real_context
scene = bpy.context.scene
assert m.comparison_state(scene) == full_before
m.integrated.verify(scene, contract)
failure = json.loads((failure_dir / 'failure.json').read_text())
assert failure['status'] == 'FAILED_SKIN_DIAGNOSTIC' and len(failure['completed_variants']) == 1
assert not (failure_dir / 'preflight.json').exists() and not (failure_dir / 'diagnostic_report.json').exists()
assert not list(failure_dir.glob('*.png'))
assert all(m.env.sha(p) == h for p, h in source_before.items()) and m.env.sha(asset['path']) == asset['sha256']
result = {'status': 'PASS_ACTUAL_CPU_SINGLE_INPUT_AND_EXCEPTION_RESTORE',
          'strict_original_verify_rejects_temporary_ablation': strict_rejection,
          'invalid_mean_cases_no_mutation': 3, 'exact_full_scene_state_restored_after_injected_failure': True,
          'failed_attempt_has_no_success_report_or_RGB': True, 'failure_record_preserved': str((failure_dir / 'failure.json').relative_to(R)),
          'controlled_input': receipt, 'source_and_asset_hashes_unchanged': True,
          'render_calls': 0, 'GPU_used': False, 'dataset_generation_allowed': False, 'CPU_threads': 2}
m.write_new(O / 'cpu_exception_checks.json', result)
print(json.dumps(result, indent=2))
