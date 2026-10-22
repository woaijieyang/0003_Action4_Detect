"""Single-variable skin photo-colour diagnostic; never a production material edit.

The outer project-Python launcher recomputes the source sRGB patch mean. Blender
uses that hash-bound evidence without installing Pillow into its runtime.
"""
import argparse
import contextlib
import copy
import json
import math
import subprocess
import sys
import traceback
from pathlib import Path
from data import action4_global_render as renderer
from data import action4_global_environment as env
from data import action4_global_scene as integrated

ROOT = env.ROOT
OUTPUTS = ROOT / 'outputs/action4_realism_reassessment_v001/skin_appearance_audit_v001'
GPU_UUID = 'GPU-8c93aadc-2a0a-665c-46c3-dbd3f063e0d8'
VARIANTS = ['original_photo', 'constant_linear_mean']
METHOD = 'decode RGB8 sRGB; linearize each channel before float64 arithmetic mean over all pixels'
SOURCES = {'preview_config', 'renderer_module', 'foundation_module', 'photo', 'patch_audit',
           'mean_evidence', 'diagnostic_module', 'diagnostic_cli'}


def mean_linear_rgb(rgb):
    import numpy as np
    array = np.asarray(rgb)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) <= 0:
        raise ValueError('An unchanged RGB8 source patch is required')
    srgb = array.astype(np.float64) / 255.0
    linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
    return linear.mean(axis=(0, 1), dtype=np.float64).tolist()


def derive_mean(photo_path):
    import numpy as np
    from PIL import Image
    path = env.bounded(photo_path, True)
    with Image.open(path) as im:
        if im.mode != 'RGB' or im.format != 'PNG':
            raise ValueError('Frozen source must be RGB PNG; no implicit conversion')
        values = np.asarray(im)
        size = list(im.size)
        mean = mean_linear_rgb(values)
    return {'source': {'path': str(path.relative_to(ROOT)), 'sha256': env.sha(path)},
            'source_color_space': 'sRGB', 'size_wh': size, 'pixel_count': size[0] * size[1],
            'method': METHOD, 'linear_rgb_mean_float64': mean,
            'transfer': 'c/12.92 if c<=0.04045 else ((c+0.055)/1.055)**2.4; c=byte/255',
            'note': 'Source-photo mean, not albedo, measured skin BRDF or a final skin material'}


def validate_config(cfg, *, recompute_mean=False):
    keys = {'schema', 'mode', 'dataset_generation_allowed', 'sources', 'shot_id',
            'variants', 'threads', 'gpu_uuid', 'mix_factor'}
    if set(cfg) != keys or cfg['schema'] != 'action4.skin_colour_diagnostic.v1' or cfg['mode'] != 'single_variable_diagnostic' or cfg['dataset_generation_allowed'] is not False:
        raise ValueError('Single-variable diagnostic scope required')
    if cfg['shot_id'] != 'left_wrist' or cfg['variants'] != VARIANTS or type(cfg['threads']) is not int or cfg['threads'] != 2 or cfg['gpu_uuid'] != GPU_UUID or type(cfg['mix_factor']) not in (int, float) or cfg['mix_factor'] != .65:
        raise ValueError('Frozen shot, two variants, mix .65 and GPU5/CPU2 required')
    if set(cfg['sources']) != SOURCES:
        raise ValueError('Complete frozen sources required')
    for ref in cfg['sources'].values():
        if set(ref) != {'path', 'sha256'} or env.sha(env.bounded(ref['path'], True)) != ref['sha256']:
            raise ValueError('Skin diagnostic source identity changed')
    preview, contract = renderer.validate_config(json.loads(env.bounded(cfg['sources']['preview_config']['path'], True).read_text()))
    if preview['render']['samples'] != 64 or preview['render']['denoiser'] != 'OPTIX':
        raise ValueError('Original native RGB64/OPTIX policy required')
    # Keep the original THREE-shot validator intact; only select one validated shot.
    shot = next(row for row in preview['inspection_poses'] if row['id'] == cfg['shot_id'])
    if contract['config']['sources']['skin_forearm'] != cfg['sources']['photo']:
        raise ValueError('Photo is not the actual integrated skin source')
    evidence = json.loads(env.bounded(cfg['sources']['mean_evidence']['path'], True).read_text())
    if evidence['source'] != cfg['sources']['photo'] or evidence['method'] != METHOD or evidence['source_color_space'] != 'sRGB':
        raise ValueError('Mean evidence is not bound to the original sRGB patch')
    mean = evidence['linear_rgb_mean_float64']
    if not isinstance(mean, list) or len(mean) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in mean):
        raise ValueError('Finite linear RGB mean required')
    if recompute_mean and derive_mean(cfg['sources']['photo']['path']) != evidence:
        raise ValueError('Source linear mean recomputation differs from frozen evidence')
    return preview, contract, shot, evidence


def output_path(value, *, fresh=True):
    path = env.bounded(value)
    if path == OUTPUTS or not path.is_relative_to(OUTPUTS):
        raise ValueError('Skin diagnostic outputs must stay in the assigned audit subtree')
    if fresh and path.exists():
        raise FileExistsError('Fresh diagnostic run required; no overwrite')
    return path


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def photo_binding(scene, photo_ref):
    """Require the exact sole photo->Mix.Color2->Principled.BaseColor connection."""
    import hashlib
    body = scene.objects['global_source_anatomy']
    if len(body.data.materials) != 1:
        raise ValueError('Expected the one frozen body skin material')
    material = body.data.materials[0]
    if material.get('photo_texture_source') != photo_ref['path'] or not material.use_nodes:
        raise ValueError('Skin source metadata changed')
    users = [o.name for o in scene.objects if hasattr(o.data, 'materials') and material in list(o.data.materials)]
    if users != [body.name]:
        raise ValueError('Skin material must not be shared with another scene object')
    tree = material.node_tree
    candidates = []
    for node in tree.nodes:
        image = getattr(node, 'image', None)
        if node.type == 'TEX_IMAGE' and image and image.packed_file and hashlib.sha256(bytes(image.packed_file.data)).hexdigest() == photo_ref['sha256']:
            candidates.append(node)
    if len(candidates) != 1:
        raise ValueError('Exactly one packed source photo image node required')
    texture = candidates[0]
    if texture.image.colorspace_settings.name != 'sRGB' or texture.projection != 'BOX' or abs(texture.projection_blend - .25) > 1e-7:
        raise ValueError('Original photo colour space/projection changed')
    links = [link for link in tree.links if link.from_node == texture and link.from_socket == texture.outputs['Color']]
    if len(links) != 1:
        raise ValueError('Exactly one photo colour consumer required')
    link = links[0]
    mix = link.to_node
    if mix.type != 'MIX_RGB' or mix.blend_type != 'MIX' or link.to_socket != mix.inputs[2] or mix.inputs[0].is_linked or abs(mix.inputs[0].default_value - .65) > 1e-7:
        raise ValueError('Original Mix .65 photo colour input is not intact')
    shader = tree.nodes.get('Principled BSDF')
    if shader is None or len(shader.inputs['Base Color'].links) != 1 or shader.inputs['Base Color'].links[0].from_node != mix:
        raise ValueError('Photo mix must feed the original skin Base Color')
    return material, texture, mix, link


@contextlib.contextmanager
def constant_photo_input(scene, contract, photo_ref, mean):
    """Temporarily replace ONE linked colour input; original verify stays strict."""
    import bpy
    import numpy as np
    if len(mean) != 3 or not np.isfinite(mean).all() or min(mean) < 0 or max(mean) > 1:
        raise ValueError('Finite source-linear mean required')
    integrated.verify(scene, contract)
    material, texture, mix, link = photo_binding(scene, photo_ref)
    tree = material.node_tree
    socket = mix.inputs[2]
    old_default = tuple(socket.default_value)
    before = env._material_state(material)
    link_key = (texture.name, list(texture.outputs).index(link.from_socket), mix.name, 2)
    receipt = {'material': material.name, 'object': 'global_source_anatomy',
               'removed_link': list(link_key), 'input_index': 2,
               'requested_linear_rgba_float64': list(mean) + [1.0],
               'original_input_default_rgba': list(old_default),
               'mix_factor_actual': float(mix.inputs[0].default_value),
               'scope': 'Single colour-input ablation only; not a final skin replacement'}
    try:
        tree.links.remove(link)
        socket.default_value = tuple(mean) + (1.0,)
        bpy.context.view_layer.update()
        receipt['actual_linear_rgba_float32'] = list(socket.default_value)
        np.testing.assert_allclose(receipt['actual_linear_rgba_float32'], receipt['requested_linear_rgba_float64'], rtol=0, atol=3e-8)
        expected = copy.deepcopy(before)
        expected['nodes']['links'] = [row for row in expected['nodes']['links'] if tuple(row) != link_key]
        node = next(row for row in expected['nodes']['nodes'] if row['name'] == mix.name)
        next(row for row in node['inputs'] if row['index'] == 2)['value'] = list(socket.default_value)
        if env._material_state(material) != expected:
            raise ValueError('Shader mutation exceeded the one intended colour input')
        yield receipt, expected
        if env._material_state(material) != expected:
            raise ValueError('Unexpected shader change during comparison')
    finally:
        for remaining in list(socket.links):
            tree.links.remove(remaining)
        socket.default_value = old_default
        tree.links.new(texture.outputs['Color'], socket)
        bpy.context.view_layer.update()
        if env._material_state(material) != before:
            raise RuntimeError('Original skin material was not restored exactly')
        integrated.verify(scene, contract)


def comparison_state(scene):
    """Same-state evidence, except the intentional output pathname."""
    runtime = renderer.state_snapshot(scene)
    runtime['render'].pop('filepath')
    runtime['animated_seed'] = scene.cycles.use_animated_seed
    return {'fixed_scene': env.fixed_snapshot(scene), 'runtime': runtime,
            'world_matrices': {o.name: [list(row) for row in o.matrix_world] for o in scene.objects},
            'pose_bones': {b.name: [list(row) for row in b.matrix_basis] for b in scene.objects['global_metric_armature'].pose.bones}}


def _configure_rgb(scene, preview):
    scene.render.threads_mode = 'FIXED'; scene.render.threads = 2; scene.frame_set(1)
    scene.cycles.samples = 64; scene.cycles.seed = preview['render']['seed']
    scene.cycles.use_animated_seed = False
    scene.cycles.max_bounces = 6; scene.cycles.diffuse_bounces = 2; scene.cycles.glossy_bounces = 2
    scene.cycles.transparent_max_bounces = 4; scene.cycles.use_adaptive_sampling = False
    scene.render.image_settings.file_format = 'PNG'; scene.render.image_settings.color_mode = 'RGB'; scene.render.image_settings.color_depth = '8'
    scene.render.use_border = False; scene.render.use_crop_to_border = False; scene.render.film_transparent = False
    scene.cycles.use_denoising = True


def execute_blender(config_path, out, threads, *, gpu_uuid=None, probe_only=False):
    import bpy
    cfg = json.loads(env.bounded(config_path, True).read_text())
    preview, contract, shot, evidence = validate_config(cfg)
    out = output_path(out, fresh=False)
    if threads != 2 or (probe_only and gpu_uuid is not None) or (not probe_only and gpu_uuid != GPU_UUID):
        raise ValueError('Explicit CPU2 preflight or pinned physical GPU5 required')
    bpy.ops.wm.open_mainfile(filepath=str(env.bounded(preview['sources']['integrated_asset']['path'], True)), load_ui=False, use_scripts=False)
    scene = bpy.context.scene; bpy.context.view_layer.update()
    integrated.verify(scene, contract)
    binding = photo_binding(scene, cfg['sources']['photo'])
    before = renderer.state_snapshot(scene)
    animated_seed_before = scene.cycles.use_animated_seed
    records = []
    try:
        try:
            with renderer.temporary_render_state(scene):
                _configure_rgb(scene, preview)
                env.apply_capture_pose(scene, shot['pose'], contract['environment'])
                integrated.verify(scene, contract)
                targets = renderer.target_parts(scene)
                visibility = renderer.inspection_visibility_guard(scene, targets, contract['canonical8']['corners_3d_m'], [1])
                target_evidence = {str(i): {'parts': sorted(o.name for o in target['parts']),
                    'core_world': [list(row) for row in target['core'].matrix_world],
                    'corners_px': renderer.project_canonical(scene, target['core'], contract['canonical8']['corners_3d_m'])[0]}
                    for i, target in targets.items()}
                device = denoiser = None
                if not probe_only:
                    device = renderer.synth._render_device(scene, {**preview['render'], 'worker_gpu_uuid': gpu_uuid})
                    denoiser = renderer.synth._set_denoiser(scene, preview['render'])
                baseline = comparison_state(scene)
                def render_variant(name, expected, mutation):
                    scene.render.filepath = str(out / (name + '.png'))
                    if comparison_state(scene) != expected:
                        raise ValueError('Comparison differs beyond the one specified skin colour input')
                    if not probe_only:
                        bpy.ops.render.render(write_still=True)
                        image = bpy.data.images.load(str(out / (name + '.png')), check_existing=False)
                        try:
                            if list(image.size) != [3248, 2464]:
                                raise ValueError('Diagnostic RGB is not native resolution')
                        finally:
                            bpy.data.images.remove(image)
                    if comparison_state(scene) != expected:
                        raise ValueError('Render changed the frozen paired scene state')
                    records.append({'variant': name, 'rendered': not probe_only, 'mutation': mutation,
                        'image': str((out / (name + '.png')).relative_to(ROOT)) if not probe_only else None,
                        'image_sha256': env.sha(out / (name + '.png')) if not probe_only else None,
                        'target_geometry_and_canonical_projection': target_evidence,
                        'pair_state_sha256': env.digest(expected)})
                render_variant('original_photo', baseline, None)
                integrated.verify(scene, contract)
                with constant_photo_input(scene, contract, cfg['sources']['photo'], evidence['linear_rgb_mean_float64']) as (mutation, expected_material):
                    expected = copy.deepcopy(baseline)
                    expected['fixed_scene']['dynamic_G1_geometry_and_materials']['global_source_anatomy']['materials'][binding[0].name] = expected_material
                    render_variant('constant_linear_mean', expected, mutation)
                if comparison_state(scene) != baseline:
                    raise RuntimeError('Pair baseline did not restore after colour ablation')
        finally:
            scene.cycles.use_animated_seed = animated_seed_before
            bpy.context.view_layer.update()
            if renderer.state_snapshot(scene) != before:
                raise RuntimeError('Temporary render state was not restored')
            integrated.verify(scene, contract)
        validate_config(cfg)
        report = {'status': 'PASS_CPU_PREFLIGHT_NO_RENDER' if probe_only else 'SKIN_ABLATION_RENDERED_PENDING_VISUAL_REVIEW',
                  'dataset_generation_allowed': False, 'rendered': not probe_only, 'mask_render_calls': 0,
                  'source_config_sha256': env.sha(config_path), 'sources': cfg['sources'],
                  'baseline_preview_sources': preview['sources'], 'mean_evidence': evidence,
                  'shot': shot, 'K': contract['environment']['K_prior'], 'seed': preview['render']['seed'],
                  'resolution_wh': [3248, 2464], 'RGB_samples': 64, 'requested_denoiser': 'OPTIX',
                  'actual_device': device, 'actual_denoiser': denoiser, 'inspection_visibility': visibility,
                  'canonical8': contract['canonical8'], 'records': records, 'restoration_verified': True,
                  'limits': ['Diagnostic ablation, not final skin/albedo/BRDF or production acceptance',
                             'Same seed and geometry/light/camera; only photo colour input differs',
                             'Float64 linear mean is stored in Blender float32 socket with recorded rounding',
                             'No masks recomputed; original ordered CAD and target geometry/projections are identical']}
        write_new(out / ('preflight.json' if probe_only else 'diagnostic_report.json'), report)
        return report
    except BaseException as exc:
        failure = out / 'failure.json'
        if not failure.exists():
            write_new(failure, {'status': 'FAILED_SKIN_DIAGNOSTIC', 'dataset_generation_allowed': False,
                'error': repr(exc), 'traceback': traceback.format_exc(), 'completed_variants': records})
        raise


def launch(config_path, out, threads, *, gpu_uuid=None, probe_only=False):
    cfg = json.loads(env.bounded(config_path, True).read_text())
    preview, _, _, _ = validate_config(cfg, recompute_mean=True)
    out = output_path(out)
    if threads != 2 or (probe_only and gpu_uuid is not None) or (not probe_only and gpu_uuid != GPU_UUID):
        raise ValueError('CPU2 and pinned physical GPU5 only; CPU preflight must not request GPU')
    child_env = renderer.worker_environment(None if probe_only else gpu_uuid)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        child_env[key] = str(threads)
    command = [str(env.bounded(preview['sources']['blender']['path'], True)), '--factory-startup', '--disable-autoexec', '-noaudio', '--background', '--threads', str(threads), '--python-exit-code', '1', '--python', str(ROOT / 'scripts/action4_skin_diagnostic.py'), '--', '--inside-blender', '--config', str(env.bounded(config_path, True)), '--output', str(out), '--threads', str(threads)]
    if probe_only: command.append('--probe-only')
    else: command.extend(['--gpu-uuid', gpu_uuid])
    out.mkdir(parents=True)
    write_new(out / 'launch.json', {'command': command, 'probe_only': probe_only, 'gpu_uuid': gpu_uuid,
                                  'config_sha256': env.sha(config_path), 'mean_recomputed': True})
    with (out / 'blender.log').open('xb') as log:
        process = subprocess.run(command, cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError('Skin diagnostic failed; preserved ' + str(out / 'blender.log'))
    return json.loads((out / ('preflight.json' if probe_only else 'diagnostic_report.json')).read_text())


def main():
    args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--threads', required=True, type=int); parser.add_argument('--gpu-uuid')
    parser.add_argument('--probe-only', action='store_true'); parser.add_argument('--inside-blender', action='store_true', help=argparse.SUPPRESS)
    a = parser.parse_args(args)
    if a.inside_blender and 'bpy' not in sys.modules:
        raise ValueError('Internal entry requires Blender')
    result = (execute_blender if a.inside_blender else launch)(a.config, a.output, a.threads, gpu_uuid=a.gpu_uuid, probe_only=a.probe_only)
    print(json.dumps({'status': result['status'], 'rendered': result['rendered'], 'dataset_generation_allowed': False}))
