"""Recipe-driven Action 4 synthesis, immutable shards, and explicit acceptance.

The host-side planner has no rendering dependencies. Blender is initialized only
by an explicit render command. Generated candidates are never training-accepted
merely because rendering or schema validation succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

SCHEMA = "action4.synthesis.v2"
ROOT = Path(__file__).resolve().parents[2]


def _read(path):
    return json.loads(_path(path, exists=True).read_text(encoding="utf-8"))


def _hash(path):
    return hashlib.sha256(_path(path, exists=True).read_bytes()).hexdigest()


def _path(path, *, exists=False):
    p = Path(path)
    p = (ROOT / p if not p.is_absolute() else p).resolve(strict=False)
    if not p.is_relative_to(ROOT):
        raise ValueError("Synthesis paths must resolve inside this project")
    return p.resolve(strict=True) if exists else p


def _write(path, value):
    path = _path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _jsonl(path):
    return [json.loads(line) for line in _path(path, exists=True).read_text().splitlines() if line.strip()]



def _denoiser_name(settings):
    name = settings.get("denoiser", "OPENIMAGEDENOISE")
    if name not in ("OPENIMAGEDENOISE", "OPTIX"):
        raise ValueError("denoiser must be OPENIMAGEDENOISE or OPTIX")
    if name == "OPTIX" and settings.get("device") not in ("CUDA", "OPTIX"):
        raise ValueError("OPTIX denoising requires an explicit CUDA or OPTIX GPU renderer")
    return name


def _set_denoiser(scene, settings):
    """Use Blender's dynamic enum setter; unsupported requests never fall back."""
    requested = _denoiser_name(settings)
    try:
        # Blender 3.6 exposes an empty static enum_items for this dynamic enum.
        # Assignment validates the actual runtime list instead of assuming support.
        scene.cycles.denoiser = requested
    except (TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError(f"Requested RGB denoiser {requested} is unavailable") from error
    actual = scene.cycles.denoiser
    if actual != requested:
        raise RuntimeError(f"RGB denoiser assignment differs: requested={requested}, actual={actual}")
    scene.cycles.use_denoising = True
    return actual


def instance_weights_for_split(recipe, split):
    """Explicit split override; historical recipes retain their global weights."""
    return recipe.get("instance_weights_by_split", {}).get(split, recipe["instance_weights"])


def validate_recipe(recipe,*,_validation_index=None):
    if recipe.get("schema_version") != SCHEMA:
        raise ValueError("Expected action4.synthesis.v2 recipe")
    families = recipe["families"]
    names = [f["name"] for f in families]
    if len(set(names)) != len(names) or len(names) < 3:
        raise ValueError("At least three uniquely named scene families are required")
    if "hard_negative" not in names:
        raise ValueError("The hard_negative family must consume the zero-instance quota")
    if len(recipe["instance_weights"]) != 3 or not all(float(x) > 0 for x in recipe["instance_weights"]):
        raise ValueError("Specify positive weights for 0, 1 and 2 instances")
    split_weights = recipe.get("instance_weights_by_split", {})
    if not isinstance(split_weights, dict) or set(split_weights) - {"train", "eval", "test"}:
        raise ValueError("Only train/eval/test instance weight overrides are supported")
    for values in split_weights.values():
        if (not isinstance(values, list) or len(values) != 3
                or any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in values)
                or not any(w > 0 for w in values)):
            raise ValueError("Per-split 0/1/2 weights must be finite nonnegative with positive total")
    for family in families:
        allowed=family.get('eligible_target_counts')
        if allowed is not None:
            expected={0} if family['name']=='hard_negative' else {1,2}
            if not isinstance(allowed,list) or not allowed or any(type(v) is not int for v in allowed) or len(set(allowed))!=len(allowed) or not set(allowed)<=expected:
                raise ValueError('Explicit supported family target-count eligibility required')
        if family["weight"] <= 0 or not family["evidence_frames"]:
            raise ValueError("Every scene needs a positive weight and source-frame evidence")
        if recipe.get("asset_backend") != "modular_composed" and len(family["anchor_rotations_xyz_deg"]) != 2:
            raise ValueError("Every scene needs linked left/right wrist anchor rotations")
    if len(recipe["resolution_wh"]) != 2 or min(recipe["resolution_wh"]) < 128:
        raise ValueError("Invalid rendering resolution")
    from data.action4_amodal import validate_settings
    validate_settings(recipe["render"])
    _denoiser_name(recipe["render"])
    if recipe["render"]["samples"] < 1 or recipe["render"]["mask_samples"] != 1:
        raise ValueError("Invalid bounded render sample settings")
    if recipe["render"].get("device") not in ("CPU", "CUDA", "OPTIX"):
        raise ValueError("Select an explicit CPU, CUDA or OPTIX render device")
    if recipe["render"]["device"] != "CPU" and not (recipe["render"].get("gpu_ids") or recipe["render"].get("gpu_uuids")):
        raise ValueError("GPU rendering requires explicit Blender device IDs")
    if recipe["render"].get("gpu_uuids"):
        from data.action4_devices import ALLOWED_GPU_UUIDS
        if any(u not in ALLOWED_GPU_UUIDS for u in recipe["render"]["gpu_uuids"]):
            raise ValueError("Recipe contains an unauthorized GPU UUID")
    if recipe["quality_gate"]["minimum_reviewed_candidates"] < 128:
        raise ValueError("Formal acceptance must retain the established 128-candidate review")
    if recipe["quality_gate"].get("minimum_visible_fraction") != .1 or recipe["quality_gate"].get("minimum_visible_fullinput_area_px") != 16:
        raise ValueError("The established low-support limits are 10% visibility and 16 full-input pixels")
    if recipe.get("asset_backend") == "modular_composed":
        from data import action4_modular_synthesis_backend as modular
        return modular.validate_recipe(recipe,_index=_validation_index)
    if recipe.get("asset_backend", "procedural_r3") not in ("procedural_r3", "blend_r4"):
        raise ValueError("Unknown target asset backend")
    if recipe.get("asset_backend") == "blend_r4":
        _path(recipe["foundation_blend"], exists=True)
    _path(recipe["asset_dir"], exists=True)
    _path(recipe["hand_asset"], exists=True)
    if recipe.get("foundation_source_manifest"):
        provenance=_path(recipe["foundation_source_manifest"],exists=True)
        if recipe.get("foundation_source_manifest_sha256") and _hash(provenance)!=recipe["foundation_source_manifest_sha256"]:
            raise ValueError("Foundation source provenance changed")
    allowed_rig={"forearm_width_gain","forearm_proximal_reach_m","forearm_proximal_splay_m",
                 "camera_head_rotation_xyz_deg","hand_offset_m","hand_pitch_deg","palm_roll_deg","mirror_hand","finger_pose","forearm_reach_power"}
    for family in families:
        for key,lo,hi,default in (("capture_focal_mm",18,32,26),("capture_roll_deg",-25,25,0),("table_grid_frequency_per_m",15,60,40)):
            value=float(family.get(key,default))
            if not math.isfinite(value) or not lo<=value<=hi:raise ValueError("Invalid bounded "+key)
        if family.get("garment_state","laid_open") not in ("laid_open","left_sleeve_fold","right_sleeve_fold","folded_lift"):
            raise ValueError("Unsupported garment state")
        contact_mode=family.get("contact_mode")
        if contact_mode not in (None,"grasped_hem","held_left_cuff","held_right_cuff"):
            raise ValueError("Unsupported cloth contact mode")
        if family.get("environment")!="table" and (family.get("garment_state","laid_open")!="laid_open" or contact_mode is not None):
            raise ValueError("Cloth fold/contact requires a table garment environment")
        side_options=family.get("rig_options_by_side",[])
        if side_options and (not isinstance(side_options,list) or len(side_options)!=2):
            raise ValueError("Two side-specific rig priors required")
        for options in [family.get("rig_options",{}),*side_options]:
            if not isinstance(options,dict) or set(options)-allowed_rig:raise ValueError("Unknown rig appearance control")
            for key,limit in (("camera_head_rotation_xyz_deg",120),("hand_offset_m",.02)):
                if key in options and (len(options[key])!=3 or any(not math.isfinite(float(v)) or abs(float(v))>limit for v in options[key])):
                    raise ValueError("Invalid bounded "+key)
            if "mirror_hand" in options and type(options["mirror_hand"]) is not bool:raise ValueError("Hand mirror must be boolean")
            if options.get("finger_pose","legacy") not in ("legacy","relaxed","closed","contact"):raise ValueError("Unsupported finger pose")
    return recipe


def _asset_identities(recipe,*,_validation_index=None):
    if recipe.get("asset_backend") == "modular_composed":
        from data import action4_modular_synthesis_backend as modular
        return modular.asset_identities(recipe,_index=_validation_index)
    asset = _path(recipe["asset_dir"], exists=True)
    manifest = _read(asset / "asset_manifest.json")
    paths = [asset / "asset_manifest.json", asset / manifest["repaired_stl"], asset / manifest["mesh_data"], _path(recipe["hand_asset"], exists=True)]
    paths.extend(sorted(asset.glob("*.png")))
    paths.append(ROOT / "src/data/action4_amodal.py")
    if recipe.get("foundation_source_manifest"):
        paths.append(_path(recipe["foundation_source_manifest"],exists=True))
    if recipe.get("asset_backend") == "blend_r4":
        paths.append(_path(recipe["foundation_blend"], exists=True))
    helper_enabled = recipe.get("asset_backend") == "blend_r4" or recipe.get("material_evidence") or recipe.get("wrist_skin_union")
    helper_enabled = helper_enabled or any(f.get("garment_state","laid_open") != "laid_open" or
        any(o.get("camera_head_rotation_xyz_deg") for o in [f.get("rig_options",{}), *f.get("rig_options_by_side",[])])
        for f in recipe["families"])
    if helper_enabled:
        paths.append(ROOT / "src/data/action4_blender_foundation.py")
    if recipe.get("material_evidence"):
        for key in ("photo_manifest", "denim_manifest"):
            path = _path(recipe["material_evidence"][key], exists=True)
            paths.append(path)
            evidence = _read(path)
            specs = evidence.get("files", evidence.get("patches", {}))
            for spec in specs.values():
                texture = _path(spec["path"], exists=True)
                if _hash(texture) != spec["sha256"]:
                    raise ValueError("Texture provenance hash mismatch")
                paths.append(texture)
    return {str(p.relative_to(ROOT)): _hash(p) for p in paths}


def validate_target_presence(presence, family=None):
    if not isinstance(presence, list) or len(presence) != 2 or any(type(v) is not bool for v in presence):
        raise ValueError("target_presence must contain two booleans [left, right]")
    if family is not None and ((family == "hard_negative") != (sum(presence) == 0)):
        raise ValueError("Zero targets must use the hard_negative family")
    return presence


def target_indices(entry):
    count = entry["target_count"]
    if type(count) is not int or count not in (0, 1, 2):
        raise ValueError("target_count must be 0, 1 or 2")
    presence = validate_target_presence(entry.get("target_presence", [count > 0, count > 1]), entry.get("scene_family"))
    if sum(presence) != count:
        raise ValueError("target_presence disagrees with target_count")
    return [i + 1 if present else 0 for i, present in enumerate(presence)]


def validate_request_recipe(value, families):
    """Strict shared feedback contract: unknown or unsupported controls fail."""
    required = {"scene_family", "view", "scale", "occluder", "occluder_location", "allowed_perturbations", "reference_frame_ids"}
    if not required.issubset(value) or set(value) - (required | {"target_presence"}) or value["scene_family"] not in families:
        raise ValueError("Unknown scene family or request recipe fields")
    if "target_presence" in value:
        validate_target_presence(value["target_presence"], value["scene_family"])
    if not isinstance(value["view"], dict) or set(value["view"]) != {"wrist_delta_xyz_deg"}:
        raise ValueError("view must specify only wrist_delta_xyz_deg")
    rotation = value["view"]["wrist_delta_xyz_deg"]
    if len(rotation) != 3 or not all(math.isfinite(float(x)) and -8 <= x <= 8 for x in rotation):
        raise ValueError("Requested wrist rotation delta must stay within ±8 degrees")
    if not .9 <= value["scale"] <= 1.1:
        raise ValueError("Requested projected scale multiplier must be in [0.9,1.1]")
    if value["occluder"] != "scene_default" or value["occluder_location"] != "linked_geometry":
        raise ValueError("Only scene_default occluders at linked_geometry are implemented; arbitrary image covers are rejected")
    perturb = value["allowed_perturbations"]
    if set(perturb) != {"rotation_jitter_deg", "position_jitter_m", "light_intensity_gain", "exposure_delta_ev"}:
        raise ValueError("Unsupported perturbation fields")
    if not 0 <= perturb["rotation_jitter_deg"] <= 8:
        raise ValueError("Rotation jitter must be [0,8] degrees")
    if len(perturb["position_jitter_m"]) != 3 or not all(0 <= x <= .004 for x in perturb["position_jitter_m"]):
        raise ValueError("Position jitter must be three values in [0,0.004] m")
    if not .9 <= perturb["light_intensity_gain"] <= 1.1 or not -.15 <= perturb["exposure_delta_ev"] <= .15:
        raise ValueError("Light or exposure change exceeds the observed-scene envelope")
    if not value["reference_frame_ids"] or not all(isinstance(x, int) and 0 <= x <= 2849 for x in value["reference_frame_ids"]):
        raise ValueError("Valid source video reference frame IDs are required")
    return value


def create_s1_plan(recipe_path, requests_path, output_path, seed=20261020, max_count=2000):
    """Consume reviewed feedback JSONL with exact event quotas and train-only split."""
    recipe_path, requests_path = _path(recipe_path, exists=True), _path(requests_path, exists=True)
    recipe = validate_recipe(_read(recipe_path))
    requests = _jsonl(requests_path)
    if not requests:
        raise ValueError("No reviewed synthesis requests")
    families = {f["name"] for f in recipe["families"]}
    seen, entries, request_counts = set(), [], {}
    for request in requests:
        rid = request["request_id"]
        if rid in seen or not isinstance(rid, str) or not rid:
            raise ValueError("S1 request IDs must be unique nonempty strings")
        seen.add(rid)
        if request.get("status") != "reviewed_request_pending_render" or request.get("split") != "train":
            raise ValueError("Only reviewed requests assigned to train may generate S1")
        if request.get("ground_truth") != "recompute_from_renderer_geometry" or request.get("prediction_coordinates_used_as_training_labels") is not False:
            raise ValueError("Feedback predictions cannot be used as synthetic pose labels")
        if not request.get("source_event_id"):
            raise ValueError("Every request must identify its reviewed failure event")
        quota = request["quota"]
        if not isinstance(quota, int) or isinstance(quota, bool) or quota < 1:
            raise ValueError("A request needs a positive integer quota")
        controls = validate_request_recipe(request["recipe"], families)
        start, stop = request["source_frame_start"], request["source_frame_end"]
        if not 0 <= start <= stop <= 2849 or any(not start <= f <= stop for f in controls["reference_frame_ids"]):
            raise ValueError("Reference frames must lie within the reviewed event")
        request_counts[rid] = quota
        if sum(request_counts.values()) > max_count:
            raise ValueError("Requested S1 quotas exceed the explicit generation budget")
        for variant in range(quota):
            index = len(entries)
            group = hashlib.sha256((request["source_event_id"] + ":" + str(variant // 4)).encode()).hexdigest()[:16]
            entries.append({"sample_id": f"s1_s{seed}_{index:07d}", "index": index, "seed": seed + index*104729,
                            "scene_family": controls["scene_family"],
                            "target_presence": controls.get("target_presence", [controls["scene_family"] != "hard_negative"] * 2),
                            "target_count": sum(controls.get("target_presence", [controls["scene_family"] != "hard_negative"] * 2)),
                            "split": "train", "group_id": f"s1_{group}", "background_source_id": f"s1_procedural_{group}",
                            "background_seed": seed + int(group[:8],16), "request_id":rid,
                            "source_event_id":request["source_event_id"], "request_recipe": controls})
    result = {"schema_version":SCHEMA,"status":"render_candidates_pending_acceptance","dataset_round":"S1",
              "count":len(entries),"seed":seed,"recipe_path":str(recipe_path.relative_to(ROOT)),"recipe_sha256":_hash(recipe_path),
              "synthesis_code_sha256":_hash(__file__),"render_base_sha256":_hash(ROOT/"src/data/action4_render.py"),
              "asset_identities":_asset_identities(recipe),"recipe":recipe,"split_counts":{"train":len(entries),"eval":0,"test":0},
              "requests_path":str(requests_path.relative_to(ROOT)),"requests_sha256":_hash(requests_path),
              "request_counts":request_counts,"requests":requests,"entries":entries,
              "sampling_notes":"Exact reviewed event quotas, train only. Scene/view/scale/lighting controls are executed; generated GT is recomputed from geometry. No automatic visual acceptance."}
    _write(output_path,result)
    return result


def allocate_entries(recipe,count=10000,seed=20261019):
    """Pure original quota/group/RNG allocation; no assets or parameter release."""
    if count < 8 or count % 4:
        raise ValueError("Use at least 8 candidates and a multiple of four grouped views")
    rng = random.Random(seed)
    families = recipe["families"]
    weights = [f["weight"] for f in families]
    groups = count // 4
    train_groups = max(1, int(groups * .8))
    eval_groups = max(1, int(groups * .1)) if groups >= 3 else 0
    test_groups = groups - train_groups - eval_groups
    if groups >= 3 and test_groups == 0:
        train_groups -= 1
        test_groups = 1
    splits = ["train"] * train_groups + ["eval"] * eval_groups + ["test"] * test_groups
    rng.shuffle(splits)
    negatives = next(i for i, f in enumerate(families) if f["name"] == "hard_negative")
    positive_families = [i for i, f in enumerate(families) if f["name"] != "hard_negative"]
    assigned_instances = {}
    quota_by_split = {}
    preview_exception = count < 128
    if not preview_exception:
        for split in ("train", "eval", "test"):
            indices = [i for i in range(count) if splits[i // 4] == split]
            count_weights = instance_weights_for_split(recipe, split)
            raw = [len(indices)*float(w)/sum(count_weights) for w in count_weights]
            quotas = [math.floor(x) for x in raw]
            for k in sorted(range(3), key=lambda k: raw[k]-quotas[k], reverse=True)[:len(indices)-sum(quotas)]:
                quotas[k] += 1
            initial_positive = {i for i in indices if i < len(families) and i != negatives}
            reserved_two = initial_positive if quotas[2] else set()
            reserved_one = initial_positive if not quotas[2] and quotas[1] else set()
            # Never force a hard negative into a split whose zero quota is0.
            reserved_zero = {negatives} if negatives in indices and quotas[0] else set()
            if len(reserved_two) > quotas[2] or len(reserved_one) > quotas[1] or len(reserved_zero) > quotas[0]:
                raise ValueError("Exact per-split instance quotas cannot support the initial family coverage at this count")
            available = [i for i in indices if i not in reserved_two | reserved_one | reserved_zero]
            rng.shuffle(available)
            n_extra_zero = quotas[0]-len(reserved_zero)
            zero = reserved_zero | set(available[:n_extra_zero])
            rest = available[n_extra_zero:]
            one = reserved_one | set(rest[:quotas[1] - len(reserved_one)])
            for i in indices:
                assigned_instances[i] = 0 if i in zero else 1 if i in one else 2
            quota_by_split[split] = {str(k): quotas[k] for k in range(3)}
    entries = []
    for group in range(groups):
        split = splits[group]
        group_id = f"s{seed}_group_{group:06d}"
        for variant in range(4):
            index = group * 4 + variant
            # Initial family coverage respects each split's allowed presence.
            # Overall hard-negative coverage comes from actual zero quotas.
            n = assigned_instances.get(index)
            if n is None:
                count_weights = instance_weights_for_split(recipe, split)
                n = rng.choices([0, 1, 2], count_weights)[0]
                if index < len(families):
                    if index == negatives and count_weights[0] > 0: n = 0
                    elif index != negatives and count_weights[2] > 0: n = 2
                    elif index != negatives and count_weights[1] > 0: n = 1
            eligible_positive = [i for i in positive_families if n in families[i].get('eligible_target_counts',[1,2])]
            if n > 0 and not eligible_positive:
                raise ValueError('Frozen presence quota has no eligible positive family; quotas are unchanged')
            initial = index if index < len(families) and index != negatives else None
            if n > 0 and initial is not None and initial not in eligible_positive:
                raise ValueError('Initial family coverage incompatible with exact frozen presence quota')
            family_index = negatives if n == 0 else (initial if initial is not None else rng.choices(eligible_positive, [weights[i] for i in eligible_positive])[0])
            family = families[family_index]
            entries.append({"sample_id": f"s{seed}_{index:07d}", "index": index, "seed": seed + index * 104729,
                            "scene_family": family["name"], "target_count": n, "group_id": group_id,
                            "background_source_id": "procedural_background_" + group_id,
                            "background_seed": seed + group * 65537, "split": split})
    for split in ("train", "eval", "test"):
        singles = [e for e in entries if e["split"] == split and e["target_count"] == 1]
        offset = random.Random(str(seed) + ":" + split + ":presence").randrange(2)
        for position, entry in enumerate(singles):
            right = bool((position + offset) % 2)
            entry["target_presence"] = [not right, right]
    for entry in entries:
        if "target_presence" not in entry:
            entry["target_presence"] = [entry["target_count"] == 2] * 2
        target_indices(entry)
    return entries,quota_by_split,preview_exception


def entry_allocation_controls(recipe):
    """Only fields read by the original allocator; excludes release/source cycles."""
    families=[]
    for family in recipe['families']:
        row={k:family[k] for k in ('name','weight')}
        if 'eligible_target_counts' in family:row['eligible_target_counts']=family['eligible_target_counts']
        families.append(row)
    return {'families':families,'instance_weights':recipe['instance_weights'],'instance_weights_by_split':recipe.get('instance_weights_by_split',{})}


def build_entry_authority(recipe,count=10000,seed=20261019):
    """First-stage immutable identity artifact, deliberately not a render plan.
    It can be built before bounded parameters/releases; all rendering still uses
    create_plan's full recipe/release validation and an exact authority check.
    """
    controls=entry_allocation_controls(recipe)
    names=[f['name'] for f in controls['families']]
    if len(names)!=len(set(names)) or len(names)<3 or 'hard_negative' not in names:raise ValueError('Unique original family names with hard negative required')
    for family in controls['families']:
        if type(family['weight']) not in (int,float) or not math.isfinite(family['weight']) or family['weight']<=0:raise ValueError('Finite positive family weights required')
        eligible=family.get('eligible_target_counts',[0] if family['name']=='hard_negative' else [1,2])
        if not isinstance(eligible,list) or not eligible or any(type(n) is not int for n in eligible) or len(eligible)!=len(set(eligible)) or not set(eligible)<=({0} if family['name']=='hard_negative' else {1,2}):raise ValueError('Supported eligibility only')
    for split in ('train','eval','test'):
        weights=instance_weights_for_split(controls,split)
        if len(weights)!=3 or any(type(w) not in (int,float) or not math.isfinite(w) or w<0 for w in weights) or not any(w>0 for w in weights):raise ValueError('Finite nonnegative presence weights')
    entries,quotas,preview=allocate_entries(controls,count,seed)
    if {e['scene_family'] for e in entries}!=set(names):raise ValueError('Entry authority must preserve overall family coverage')
    return {'schema':'action4.modular_entry_authority.v1','status':'FROZEN_ENTRY_IDENTITIES_ONLY_NOT_RENDER_PLAN','count':count,'seed':seed,'allocation_controls':controls,'entries':entries,'intended_instance_quotas_by_split':quotas,'instance_counts_by_split':{s:{str(k):sum(e['split']==s and e['target_count']==k for e in entries) for k in range(3)} for s in ('train','eval','test')},'split_counts':{s:sum(e['split']==s for e in entries) for s in ('train','eval','test')},'preview_coverage_exception':preview,'dataset_generation_allowed':False,'training_accepted':False}


def create_plan(recipe_path, output_path, count=10000, seed=20261019, *, entry_authority_ref=None):
    """Freeze exact split quotas before sampling; related views never cross splits."""
    recipe_path = _path(recipe_path, exists=True)
    # Bind the serialized recipe and its SHA to the same read, not a later file.
    recipe_bytes=recipe_path.read_bytes()
    recipe=json.loads(recipe_bytes)
    recipe_sha256=hashlib.sha256(recipe_bytes).hexdigest()
    validation_index=None;draft=None
    if recipe.get("asset_backend") == "modular_composed":
        from data import action4_modular_synthesis_backend as modular
        validation_index=modular._PlanIndex()
        # Private to this call. No allocation identity changes at full binding.
        draft={'recipe':recipe,'count':count,'seed':seed}
        validation_index.bind_plan(draft)
        recipe=validate_recipe(recipe,_validation_index=validation_index)
    else:
        recipe=validate_recipe(recipe)
    entries,quota_by_split,preview_exception=allocate_entries(recipe,count,seed)
    if draft is not None:draft['entries']=entries
    families=recipe['families']
    if entry_authority_ref is not None:
        if validation_index is None:
            from data import action4_modular_parameters as parameters
            authority=json.loads(parameters.verified_bytes(entry_authority_ref))
            allocated=build_entry_authority(recipe,count,seed)
        else:
            authority=validation_index.read_ref(entry_authority_ref)
            allocated=validation_index.plan_authority(draft)
        if authority!=allocated:raise ValueError('Final recipe allocation differs from frozen entry authority')
    result = {"schema_version": SCHEMA, "status": "render_candidates_pending_acceptance", "count": count,
              "seed": seed, "recipe_path": str(recipe_path.relative_to(ROOT)), "recipe_sha256": recipe_sha256,
              "synthesis_code_sha256": _hash(__file__),
              "render_base_sha256": _hash(ROOT / "src/data/action4_render.py"),
              "asset_identities": _asset_identities(recipe,_validation_index=validation_index),
              "recipe": recipe, "split_counts": {s: sum(e["split"] == s for e in entries) for s in ("train", "eval", "test")},
              "instance_counts_by_split": {s: {str(k):sum(e["split"]==s and e["target_count"]==k for e in entries) for k in range(3)} for s in ("train","eval","test")},
              "intended_instance_quotas_by_split":quota_by_split,
              "family_counts":{f["name"]:sum(e["scene_family"]==f["name"] for e in entries) for f in families},
              "preview_coverage_exception":preview_exception,
              "sampling_notes": "Four related views share a split. At counts>=128, exact per-split 0/1/2 quotas are allocated before scenes; hard negatives consume the zero quota. Smaller previews prioritize family coverage and report their actual counts. Scene weights are design priors, not observed video frequencies.",
              "entries": entries}
    if entry_authority_ref is not None:result['entry_authority_ref']=entry_authority_ref
    if validation_index is not None:
        # The sole draft->full transition is local, with exactly the same
        # recipe/count/seed/entries. No generic index rebinding rule is relaxed.
        if validation_index.plan is not draft or any(result[k]!=draft[k] for k in ('recipe','count','seed','entries')):
            raise ValueError('Plan build allocation identity changed before final binding')
        validation_index.bind_plan(result)
        modular.validate_plan(result,_index=validation_index)
        validation_index.seal()
    if json.loads(recipe_bytes)!=recipe:
        raise ValueError('Plan validator mutated the parsed recipe bytes')
    # Source identities describe the actual validated bytes. No claim that all
    # files stay unchanged forever: renderer/Host retain their full closing.
    # Serialization must succeed before the exclusive output file is created.
    json.dumps(result,ensure_ascii=False,allow_nan=False)
    _write(output_path, result)
    return result


def select_shard_entries(plan, shard_index=0, num_shards=1, limit=None, entry_start=0, entry_count=None):
    if type(num_shards) is not int or num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid shard index/count")
    if type(entry_start) is not int or entry_start < 0:
        raise ValueError("entry_start must be nonnegative")
    if limit is not None and entry_count is not None:
        raise ValueError("Use either legacy limit or entry_count")
    count = entry_count if entry_count is not None else limit
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError("Entry count must be positive")
    entries = [e for e in plan["entries"] if e["index"] % num_shards == shard_index]
    chosen = entries[entry_start:None if count is None else entry_start+count]
    if not chosen:
        raise ValueError("Empty shard range")
    return chosen


def _run_output_dir(data_dir):
    """Runtime evidence belongs under outputs; RGB/masks/labels stay in data."""
    data_dir = _path(data_dir)
    relative = data_dir.relative_to(ROOT / "data")
    return _path(ROOT / "outputs/action4_synthesis_runs" / relative)


def render_shard(plan_path, output_dir, blender, shard_index=0, num_shards=1, limit=None, threads=2,
                 *, gpu_uuid=None, entry_start=0, entry_count=None):
    plan_path, output_dir = _path(plan_path, exists=True), _path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Use a new shard output directory; failed and completed shards are immutable")
    if not 0 <= shard_index < num_shards or not 1 <= threads <= 64:
        raise ValueError("Invalid shard or explicit CPU thread budget")
    plan = _read(plan_path)
    validate_recipe(plan["recipe"])
    if plan.get("synthesis_code_sha256") != _hash(__file__) or plan.get("render_base_sha256") != _hash(ROOT / "src/data/action4_render.py"):
        raise ValueError("Renderer changed after plan freeze; create a new plan/version")
    if plan.get("asset_identities") != _asset_identities(plan["recipe"]):
        raise ValueError("Scene or target assets changed after plan freeze")
    chosen = select_shard_entries(plan,shard_index,num_shards,limit,entry_start,entry_count)
    from data.action4_devices import worker_environment
    settings = plan["recipe"]["render"]
    if settings["device"] == "CPU" and gpu_uuid is not None:
        raise ValueError("CPU plan cannot silently switch to GPU")
    if settings["device"] != "CPU" and gpu_uuid not in settings.get("gpu_uuids", []):
        raise ValueError("GPU worker needs one UUID from the frozen plan allowlist")
    env = worker_environment(gpu_uuid)
    blender = _path(blender, exists=True)
    output_dir.mkdir(parents=True)
    run_dir = _run_output_dir(output_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    command = [str(blender), "--factory-startup", "--disable-autoexec", "-noaudio", "--background", "--threads", str(threads), "--python-exit-code", "1", "--python",
               str(ROOT / "scripts/action4_synthesis.py"), "--", "_render-shard", "--plan", str(plan_path),
               "--output", str(output_dir), "--shard-index", str(shard_index), "--num-shards", str(num_shards)]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    command.extend(["--entry-start",str(entry_start)])
    if entry_count is not None:command.extend(["--entry-count",str(entry_count)])
    if gpu_uuid is not None:command.extend(["--gpu-uuid",gpu_uuid])
    _write(run_dir / "launch.json", {"command": command,"gpu_uuid":gpu_uuid,"entry_start":entry_start,"entry_count":entry_count,
                                        "expected_sample_ids": [e["sample_id"] for e in chosen],
                                        "plan_sha256": _hash(plan_path)})
    with (run_dir / "blender.log").open("xb") as log:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        raise RuntimeError(f"Blender exited {process.returncode}; inspect {run_dir / 'blender.log'}")
    report = validate_shard(output_dir)
    if report["sample_ids"] != [e["sample_id"] for e in chosen]:
        raise ValueError("Completed sample IDs differ from the frozen shard request")
    return report


def _validate_shard_records(output_dir,records=None,validation_session=None):
    """Independently verify PNG bytes, mask accounting, and OpenCV reprojection."""
    import numpy as np
    from PIL import Image
    from data.action4_adapter import DataAdapter
    output_dir = _path(output_dir, exists=True)
    records = _jsonl(output_dir / "manifest.jsonl") if records is None else records
    if not records:
        raise ValueError("Empty render output")
    adapter = DataAdapter(ROOT)
    seen, errors, visible_pixels = set(), [], []
    for record in records:
        adapter.validate_sample(record)
        if record.get("asset_backend") == "modular_composed":
            from data import action4_modular_synthesis_backend as modular
            modular.validate_host_record(record,validation_session=validation_session)
        if _hash(_path(record["image"], exists=True)) != record["rgb_sha256"]:
            raise ValueError("RGB bytes differ from the rendered sample identity")
        if record["sample_id"] in seen:
            raise ValueError("Duplicate rendered sample ID")
        seen.add(record["sample_id"])
        camera = record["camera"]
        corners = np.array(record["corners_3d_mm"], dtype=float)
        for instance in record["instances"]:
            T = np.diag([1., -1., -1., 1.]) @ np.linalg.inv(np.array(camera["camera_to_world_m"])) @ np.array(instance["object_to_world_m"])
            T[:3, 3] *= 1000
            projected, _ = adapter.project_corners(corners, np.array(camera["K"]), T)
            error = float(np.max(np.abs(projected - np.array(instance["corners_px"]))))
            if error > .001:
                raise ValueError(f"Projection convention mismatch: {error}")
            errors.append(error)
            for field in ("mask_visible", "mask_amodal_expanded"):
                p = _path(instance[field], exists=True)
                expected_hash = instance.get(field + "_sha256")
                if expected_hash is not None and _hash(p) != expected_hash:
                    raise ValueError("Mask bytes differ from the rendered sample identity")
                with Image.open(p) as im:
                    im.load()
                    array = np.array(im) > 127
                if field == "mask_visible":
                    visible = array
                else:
                    full = array
            h, w = record["height"], record["width"]
            amodal = full[h:2*h, w:2*w]
            av, ai, af = int(visible.sum()), int(amodal.sum()), int(full.sum())
            if (av, ai, af) != (instance["area_visible_px"], instance["area_inframe_amodal_px"], instance["area_full_amodal_px"]):
                raise ValueError("Stored mask areas do not match image bytes")
            if not 0 < ai <= af or av > ai + max(4, .005*ai):
                raise ValueError("Invalid or substantially nonnested masks")
            visible_pixels.append(av)
    result = {"status": "geometry_verified_visual_acceptance_pending", "sample_ids": [r["sample_id"] for r in records],
              "images": len(records), "instances": len(errors), "corners": len(errors)*8,
              "max_projection_abs_error_px": max(errors, default=0.),
              "minimum_visible_area_px": min(visible_pixels, default=None),
              "training_accepted": False, "manifest_sha256": _hash(output_dir / "manifest.jsonl")}
    destination = _run_output_dir(output_dir) / "geometry_validation.json"
    if validation_session is not None:
        validation_session.defer_report(destination,result)
        return result
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _read(destination) != result:
            raise ValueError("Existing validation report no longer matches bytes")
    else:
        _write(destination, result)
    return result


def _assemble_validated(plan_path, shard_dirs, output_dir, acceptance_path=None,validation_session=None):
    """Assemble complete immutable shards; explicit review is required for training."""
    plan = _read(_path(plan_path, exists=True)) if validation_session is None else validation_session.plan
    plan_sha256 = _hash(_path(plan_path, exists=True)) if validation_session is None else validation_session.plan_ref["sha256"]
    output_dir = _path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Assembly output already exists")
    by_id = {}
    for directory in shard_dirs:
        directory = _path(directory, exists=True)
        current_receipt = _run_output_dir(directory) / "launch.json"
        legacy_receipt = directory / "launch.json"
        if current_receipt.exists() or current_receipt.is_symlink():
            # Prefer the writer's canonical mapping; never hide malformed or
            # dangling current evidence by falling back to an older receipt.
            launch = _read(current_receipt)
            if (legacy_receipt.exists() or legacy_receipt.is_symlink()) and _read(legacy_receipt) != launch:
                raise ValueError("Conflicting canonical and legacy launch receipts")
        else:
            launch = _read(legacy_receipt)
        if launch.get("plan_sha256") != plan_sha256:
            raise ValueError("Cannot combine shards launched from a different frozen plan")
        validate_shard(directory,validation_session=validation_session)
        for record in _jsonl(_path(directory, exists=True) / "manifest.jsonl"):
            key = record["sample_id"]
            if key in by_id:
                raise ValueError("Duplicate sample across shards")
            by_id[key] = record
    if set(by_id) != {e["sample_id"] for e in plan["entries"]}:
        raise ValueError("A formal assembly requires every planned sample exactly once")
    modular_rows = None
    if plan["recipe"].get("asset_backend") == "modular_composed":
        from data import action4_modular_synthesis_backend as modular
        if validation_session is None:
            modular.validate_plan(plan)
            modular_rows = modular._release(plan["recipe"])[3]
    accepted = False
    generation_release = None
    if acceptance_path:
        acceptance = (_read(_path(acceptance_path, exists=True)) if validation_session is None else validation_session.read_tracked_json(acceptance_path))
        evidence = acceptance.get("reviewed_samples", [])
        if acceptance.get("decision") != "accepted" or acceptance.get("plan_sha256") != _hash(_path(plan_path, exists=True)):
            raise ValueError("Acceptance must explicitly bind this plan")
        if plan["recipe"]["quality_gate"].get("native_resolution_pending", True):
            raise ValueError("The preview recipe still requires native-resolution detail acceptance")
        if plan["recipe"]["resolution_wh"] != [3248, 2464]:
            raise ValueError("Accepted synthesis requires native resolution [3248,2464]; changing a review flag cannot promote preview images")
        if any((record["width"], record["height"]) != (3248, 2464) for record in by_id.values()):
            raise ValueError("Every accepted sample must match native resolution 3248x2464")
        required_checks = ("anatomy_and_pose", "mount_and_cable_attachment", "cloth_contact", "lighting_and_material", "native_roi_detail", "collision_and_masks")
        if not all(acceptance.get("criteria", {}).get(key) is True for key in required_checks):
            raise ValueError("Formal acceptance requires every declared visual/geometry criterion")
        if len({e["sample_id"] for e in evidence}) < plan["recipe"]["quality_gate"]["minimum_reviewed_candidates"]:
            raise ValueError("Insufficient explicit candidate visual review")
        for item in evidence:
            record = by_id.get(item["sample_id"])
            if not record or not item.get("reviewer") or not item.get("observations") or item.get("decision") != "accepted":
                raise ValueError("Each reviewed sample needs a reviewer, observations and an accepted decision")
            if item["rgb_sha256"] != _hash(_path(record["image"], exists=True)):
                raise ValueError("Visual review image identity is stale")
        reviewed_families = {by_id[item["sample_id"]]["scene_family"] for item in evidence}
        if reviewed_families != {entry["scene_family"] for entry in plan["entries"]}:
            raise ValueError("Visual acceptance must cover every scene family represented in this plan")
        if any("mask_visible_sha256" not in i or "mask_amodal_expanded_sha256" not in i for r in by_id.values() for i in r["instances"]):
            raise ValueError("Formal acceptance requires mask byte identities for every target")
        if plan["recipe"].get("asset_backend") == "modular_composed":
            from data import action4_modular_synthesis_backend as modular
            generation_release = (modular.validate_generation_release(plan, {"path": str(_path(plan_path).relative_to(ROOT)), "sha256": plan_sha256}, list(by_id.values()), acceptance) if validation_session is None else modular.validate_generation_release_batched(plan,validation_session.plan_ref,list(by_id.values()),acceptance,validation_session))
        accepted = True
    records = [by_id[e["sample_id"]] for e in plan["entries"]]
    for e, r in zip(plan["entries"], records):
        if plan["recipe"].get("asset_backend") == "modular_composed":
            from data import action4_modular_synthesis_backend as modular
            if validation_session is None:modular.validate_record(r, e, plan_sha256, plan, modular_rows)
            else:validation_session.validate_record(r,e)
        if r["split"] != e["split"] or r["group_id"] != e["group_id"]:
            raise ValueError("Rendered grouping differs from frozen plan")
        if r["synthesis_code_sha256"] != plan["synthesis_code_sha256"] or r["render_base_sha256"] != plan["render_base_sha256"]:
            raise ValueError("Cannot combine shards rendered with different source code")
        if len(r["instances"]) != e["target_count"]:
            raise ValueError("Rendered target count differs from its allocated quota; repair within this sample's split")
    if validation_session is not None:validation_session.close()
    for r in records:
        r["status"] = "training_accepted" if accepted else "render_candidate_not_training_accepted"
        if plan["recipe"].get("asset_backend") == "modular_composed":
            r["training_accepted"] = accepted
            r["dataset_generation_allowed"] = accepted
            r["modular_backend"]["generation_release_ref"] = generation_release
    output_dir.mkdir(parents=True)
    for split in ("train", "eval", "test"):
        with (output_dir / (split + ".jsonl")).open("x") as file:
            for record in records:
                if record["split"] == split:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {"training_accepted": accepted, "count": len(records), "split_counts": plan["split_counts"],
               "plan_sha256": plan_sha256, "acceptance": str(acceptance_path) if accepted else None}
    if plan["recipe"].get("asset_backend") == "modular_composed":
        summary["generation_release_ref"] = generation_release
    _write(output_dir / "dataset_manifest.json", summary)
    return summary



def _shard_launch_identity(directory,records,plan_ref=None):
    current=_run_output_dir(directory)/'launch.json';legacy=directory/'launch.json';refs=[]
    def read_bound(path):
        raw=path.read_bytes();refs.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest()});return json.loads(raw)
    if current.exists() or current.is_symlink():
        launch=read_bound(current)
        if (legacy.exists() or legacy.is_symlink()) and read_bound(legacy)!=launch:raise ValueError('Conflicting canonical and legacy launch receipts')
    else:launch=read_bound(legacy)
    if plan_ref is None:
        candidates=[r['pose_controls_proof']['plan_ref'] for r in records if 'pose_controls_proof' in r]
        if candidates:
            plan_ref=candidates[0]
            if any(r!=plan_ref for r in candidates):raise ValueError('Mixed plan references in shard')
        else:
            command=launch.get('command',[])
            if command.count('--plan')!=1:raise ValueError('Shard launch must identify its exact formal plan')
            path=_path(command[command.index('--plan')+1],exists=True);plan_ref={'path':str(path.relative_to(ROOT)),'sha256':launch['plan_sha256']}
    if launch.get('plan_sha256')!=plan_ref['sha256']:raise ValueError('Shard launch plan SHA differs from session')
    command=launch.get('command',[])
    if command.count('--plan')==1 and str(_path(command[command.index('--plan')+1],exists=True).relative_to(ROOT))!=plan_ref['path']:raise ValueError('Shard launch plan path differs from session')
    if command.count('--plan')>1:raise ValueError('Ambiguous shard plan command')
    if 'expected_sample_ids' in launch and launch['expected_sample_ids']!=[r['sample_id'] for r in records]:raise ValueError('Manifest differs from launched sample IDs')
    return plan_ref,refs


def validate_shard(output_dir,*,validation_session=None,plan_ref=None):
    # The whole public operation participates in the session failure domain,
    # including directory resolution, manifest decoding and backend classification.
    try:
        if validation_session is not None:validation_session.require_plan(plan_ref or validation_session.plan_ref)
        return _validate_shard_impl(output_dir,validation_session=validation_session,plan_ref=plan_ref)
    except BaseException as exc:
        if validation_session is not None:validation_session.poison(exc)
        raise


def _validate_shard_impl(output_dir,*,validation_session=None,plan_ref=None):
    directory=_path(output_dir,exists=True);manifest=directory/'manifest.jsonl';raw=manifest.read_bytes();records=[json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if not records:raise ValueError('Empty render output')
    modular_flags=[r.get('asset_backend')=='modular_composed' for r in records]
    if not any(modular_flags):
        if validation_session is not None:validation_session.poison('Nonmodular shard in modular session');raise ValueError('Wrong backend for host session')
        return _validate_shard_records(directory,records)
    if not all(modular_flags):raise ValueError('Mixed backend records in modular shard')
    from data import action4_modular_synthesis_backend as modular
    def execute(session):
        try:
            expected=plan_ref or session.plan_ref;session.require_plan(expected)
            reference,launch_refs=_shard_launch_identity(directory,records,expected);session.require_plan(reference)
            session.track_reference({'path':str(manifest.relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest()})
            for reference in launch_refs:session.track_reference(reference)
            return _validate_shard_records(directory,records,session)
        except BaseException as exc:session.poison(exc);raise
    if validation_session is not None:return execute(validation_session)
    reference,_=_shard_launch_identity(directory,records,plan_ref)
    with modular.HostValidationSession(reference) as session:result=execute(session)
    return result


def assemble(plan_path,shard_dirs,output_dir,acceptance_path=None):
    path=_path(plan_path,exists=True);plan=_read(path)
    if plan['recipe'].get('asset_backend')!='modular_composed':return _assemble_validated(plan_path,shard_dirs,output_dir,acceptance_path)
    from data import action4_modular_synthesis_backend as modular
    reference={'path':str(path.relative_to(ROOT)),'sha256':_hash(path)}
    with modular.HostValidationSession(reference) as session:
        return _assemble_validated(plan_path,shard_dirs,output_dir,acceptance_path,session)


def _anatomical_hand(asset_path, name, material, parent, mirror=False, curl=.65, finger_pose="legacy"):
    """Append the selected CC0 anatomical mesh and use a bounded finger curl."""
    from data import action4_render as base
    from mathutils import Matrix
    if finger_pose not in ("legacy","relaxed","closed","contact"):
        raise ValueError("Unsupported finite finger pose")
    bpy = base._bpy()
    with bpy.data.libraries.load(str(asset_path), link=False) as (source, target):
        if "Hand  - Realistic" not in source.objects:
            raise ValueError("Required CC0 anatomical hand missing")
        target.objects = ["Hand  - Realistic"]
    obj = target.objects[0]
    bpy.context.collection.objects.link(obj)
    obj.name = name
    obj.location = (0, 0, 0)
    obj.rotation_euler = (0, 0, 0)
    obj.scale = (1, 1, 1)
    # Bake the licensed sculpt before changing coordinates. Leaving the source
    # multires displacement attached to a rotated/deformed base warps fingertips.
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    baked = bpy.data.meshes.new_from_object(evaluated)
    obj.modifiers.clear()
    obj.data = baked
    # The distributed hand points down. Rotate it to fingers +Z, then place its
    # wrist behind the camera/mount. This is a generic morphology, not subject GT.
    for v in obj.data.vertices:
        x, y, z = v.co
        x = (-1 if mirror else 1) * x * .78
        y = -y * .78 - .060
        z = -z * .78 - .062
        if z > .018:
            distance = z - .018
            curvature = max(curl, .01) / .10
            if finger_pose != "legacy":
                # Digit lengths measured from this existing CC0 mesh in the
                # straight-hand probe; normalize bend rather than shorten fingers.
                anatomical_x = (-1 if mirror else 1) * x
                centres = (-.0356,-.0099,.0128,.0377)
                lengths = (.050,.071,.080,.068)
                nearest=min(range(4),key=lambda k:abs(anatomical_x-centres[k]))
                length=lengths[nearest]
                total_angle={"relaxed":2.8,"closed":3.25,"contact":2.9}[finger_pose]
                is_thumb=anatomical_x>.052 or (anatomical_x>.045 and z<.04)
                # Each separated distal digit retains one curvature across its
                # cross-section; interpolating into the thumb distorted tips.
                curvature=.1 if is_thumb else total_angle/length
            angle = curvature * distance
            # Bend the centerline AND rotate its cross section. Translating
            # the centerline alone collapsed fingertip depth near a 90-degree
            # bend, producing the flat ends rejected in previews v005/v006.
            # This is a geometric deformation of generic anatomy, not inferred
            # subject joint angles or a mechanically validated hand rig.
            cross_section_y = y + .060
            y = -.060 + (1 - math.cos(angle)) / curvature + cross_section_y * math.cos(angle)
            z = .018 + math.sin(angle) / curvature - cross_section_y * math.sin(angle)
            x *= 1 - .10 * min(distance / .05, 1)
        v.co = (x, y, z)
    # Mirroring reverses winding; recalculate to keep lighting physically valid.
    if mirror:
        import bmesh
        bm = bmesh.new(); bm.from_mesh(obj.data)
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        bm.to_mesh(obj.data); bm.free()
    obj.parent = parent
    obj.data.materials.clear(); obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    smoothing = obj.modifiers.new("posed_finger_surface", "SUBSURF")
    smoothing.levels = 1
    smoothing.render_levels = 1
    obj["source_asset"] = "Blender Studio Human Base Meshes v1.0.0 / Hand  - Realistic"
    obj["author"] = "Dan Ulrich"; obj["license"] = "CC0"
    obj["pose_status"] = "Generic bounded curl; not measured subject articulation"
    return obj


def build_assets(source_asset_dir, hand_bundle, output_asset_dir, output_hand_dir):
    """Package traceable mesh/material and selected CC0 anatomy in new locations."""
    from data import action4_render as base
    bpy = base._bpy()
    source = _path(source_asset_dir, exists=True)
    bundle = _path(hand_bundle, exists=True)
    out, hand_out = _path(output_asset_dir), _path(output_hand_dir)
    if out.exists() or hand_out.exists():
        raise FileExistsError("Asset versions are immutable; select new output directories")
    original = _read(source / "asset_manifest.json")
    if _hash(source / original["repaired_stl"]) != original["repaired_sha256"]:
        raise ValueError("Canonical STL differs from its provenance manifest")
    out.mkdir(parents=True); hand_out.mkdir(parents=True)
    for name in (original["repaired_stl"], original["mesh_data"]):
        shutil.copy2(source / name, out / name)
    for p in source.glob("*.png"):
        shutil.copy2(p, out / p.name)
    for p in source.glob("appearance_evidence*.json"):
        shutil.copy2(p, out / p.name)
    manifest = {**original, "asset_version": "action4_asset_r3_v001",
                "status": "geometry_verified_appearance_review_pending",
                "parent_asset_manifest_sha256": _hash(source / "asset_manifest.json"),
                "parent_asset_dir": str(source.relative_to(ROOT)),
                "r3_geometry_change": "none; no evidence supports further topology or dimension changes",
                "r3_appearance_change": "editable node material and packed source-image screen/top labels; no physical reflectance measurement"}
    _write(out / "asset_manifest.json", manifest)
    base._reset_scene(812, 616, 32)
    core = base._core(out)
    base._camera((.10, .20, .10), (0, 0, 0), .11)
    base._light((.05, .12, .18), .5, .2)
    bpy.data.texts.new("asset_manifest.json").write(json.dumps(manifest,ensure_ascii=False,indent=2))
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "Osmo_Action_4_appearance_r3_v001.blend"),check_existing=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    with bpy.data.libraries.load(str(bundle),link=False) as (src,dst):
        dst.objects = ["Hand  - Realistic"]
    hand = dst.objects[0]
    if hand.asset_data is None or hand.asset_data.license != "CC0":
        raise ValueError("Selected anatomical asset does not carry its expected CC0 metadata")
    bpy.context.collection.objects.link(hand)
    hand.location = (0,0,0)
    provenance = {"name": "Hand  - Realistic", "author": hand.asset_data.author,
                  "license": hand.asset_data.license, "source_blend_sha256": _hash(bundle),
                  "source_bundle": str(bundle.relative_to(ROOT)),
                  "official_url": "https://download.blender.org/demo/asset-bundles/human-base-meshes/human-base-meshes-bundle-v1.0.0.zip",
                  "source_archive_sha256": "46a912c0524072ac3b78c35d5d2471df7b8df102394a050ca8cd7184e3393648",
                  "geometry_status": "unmodified selected anatomical mesh and original multires sculpt; scene posing occurs at render time",
                  "limitations": "generic anatomical mesh; no subject-specific shape, pose, skin or garment reconstruction"}
    bpy.data.texts.new("PROVENANCE.json").write(json.dumps(provenance,ensure_ascii=False,indent=2))
    target = hand_out / "hand_realistic_cc0.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(target),check_existing=False)
    provenance["derived_blend_sha256"] = _hash(target)
    _write(hand_out / "asset_manifest.json", provenance)
    return {"asset_dir":str(out),"hand_asset":str(target),"geometry_changed":False,"appearance_accepted":False}


def _skin(name, color):
    from data import action4_render as base
    mat = base._material(name, color, .58)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Subsurface"].default_value = .08
    bsdf.inputs["Subsurface Radius"].default_value = (.002, .0008, .0004)
    noise = nodes.new("ShaderNodeTexNoise"); noise.inputs["Scale"].default_value = 700
    noise.inputs["Detail"].default_value = 2
    bump = nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = .12
    bump.inputs["Distance"].default_value = .00012
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _forearm(name, material, parent, width_gain=1.0, proximal_reach_m=0.0, proximal_splay_m=0.0, reach_power=2):
    """Tapered elliptical arm with wrist transition, not isolated primitive fingers."""
    from data import action4_render as base
    bpy = base._bpy()
    profiles = [(-.48, .047, .040), (-.38, .045, .036), (-.28, .040, .033),
                (-.18, .034, .029), (-.115, .029, .023), (-.084, .027, .022)]
    verts, faces, n = [], [], 40
    for z, rx, ry in profiles:
        for i in range(n):
            a = 2*math.pi*i/n
            weight = max(0., min(1., (-z-.14)/.34))
            verts.append((width_gain*rx*math.cos(a) + proximal_splay_m*weight**reach_power,
                          -.060 + width_gain*ry*math.sin(a) + proximal_reach_m*weight**reach_power, z))
    for j in range(len(profiles)-1):
        for i in range(n):
            k=j*n+i; q=j*n+(i+1)%n
            faces.append((k, q, q+n, k+n))
    faces.extend([tuple(reversed(range(n))), tuple(range((len(profiles)-1)*n, len(profiles)*n))])
    mesh = bpy.data.meshes.new(name); mesh.from_pydata(verts, [], faces); mesh.update()
    obj = bpy.data.objects.new(name, mesh); bpy.context.collection.objects.link(obj)
    obj.parent = parent; mesh.materials.append(material)
    for poly in mesh.polygons: poly.use_smooth = True
    modifier = obj.modifiers.new("anatomical_profile_smoothing", "SUBSURF")
    modifier.levels = 2; modifier.render_levels = 2
    return obj


def _rounded_box(name, center, dimensions, material, parent=None, bevel=.002):
    from data import action4_render as base
    obj = base._cube(name, center, dimensions, material, parent)
    mod = obj.modifiers.new("manufactured_edge_radius", "BEVEL")
    mod.width = bevel; mod.segments = 3
    obj.data.use_auto_smooth = True
    obj.data.auto_smooth_angle = math.radians(30)
    obj.modifiers.new("weighted_surface_normals", "WEIGHTED_NORMAL")
    return obj


def _wrist_rig(recipe, name, position, rotation, side, target_index, curl=.65, rig_options=None):
    from data import action4_render as base
    bpy = base._bpy()
    rig = bpy.data.objects.new(name, None); bpy.context.collection.objects.link(rig)
    backend = recipe.get("asset_backend", "procedural_r3")
    if target_index and backend == "blend_r4":
        from data.action4_blender_foundation import load_r4_core
        core = load_r4_core(_path(recipe["asset_dir"]), _path(recipe["foundation_blend"], exists=True), name + "_core", target_index)
    else:
        core = base._core(_path(recipe["asset_dir"]), name + "_core", target_index) if target_index else None
    if core: core.parent = rig
    skin = _skin(name + "_skin", recipe["skin_linear_rgb"])
    if recipe.get("material_evidence"):
        from data.action4_blender_foundation import apply_photo_material
        photo = _read(_path(recipe["material_evidence"]["photo_manifest"], exists=True))
        apply_photo_material(skin, photo["patches"]["skin_forearm"]["path"], .06, strength=.65)
    rig_options = rig_options or {}
    mirror_hand=rig_options.get("mirror_hand",side > 0)
    if type(mirror_hand) is not bool:raise ValueError("Hand mirror prior must be boolean")
    hand = _anatomical_hand(_path(recipe["hand_asset"]), name+"_anatomical_hand", skin, rig, mirror_hand, curl, rig_options.get("finger_pose","legacy"))
    width_gain = float(rig_options.get("forearm_width_gain", 1.0))
    reach = float(rig_options.get("forearm_proximal_reach_m", 0.0))
    splay = float(rig_options.get("forearm_proximal_splay_m", 0.0))
    if not .85 <= width_gain <= 1.3 or not 0 <= reach <= .35 or not -.20 <= splay <= .20:
        raise ValueError("Forearm appearance prior outside bounded range")
    hand_offset = rig_options.get("hand_offset_m", [0,0,0])
    if len(hand_offset)!=3 or any(not math.isfinite(float(v)) or abs(float(v))>.02 for v in hand_offset):
        raise ValueError("Hand placement offset outside bounded appearance prior")
    palm_roll=float(rig_options.get("palm_roll_deg",0))
    if palm_roll not in (0.,180.): raise ValueError("Only two observed palm-facing variants are supported")
    hand_pitch=float(rig_options.get("hand_pitch_deg",0))
    if not -25<=hand_pitch<=25: raise ValueError("Hand pitch prior outside bounded range")
    from mathutils import Euler,Vector
    hand_rotation=Euler((math.radians(hand_pitch),0,math.radians(palm_roll)))
    pivot=Vector((0,-.060,-.100))
    hand.rotation_euler=hand_rotation
    hand.location=Vector(hand_offset)+pivot-hand_rotation.to_matrix()@pivot
    hand["finite_finger_pose"]=rig_options.get("finger_pose","legacy")
    hand["palm_roll_deg"]=palm_roll
    reach_power=rig_options.get("forearm_reach_power",2)
    if reach_power not in (1,2):raise ValueError("Forearm reach power must be1or2")
    forearm = _forearm(name+"_forearm", skin, rig, width_gain, reach, splay,reach_power)
    if recipe.get("wrist_skin_union", False):
        from data.action4_blender_foundation import join_wrist_skin
        join_wrist_skin(hand, forearm)
    black = base._material(name+"_mount_black", (.008, .010, .012), .62)
    gray = base._material(name+"_strap_woven_gray", (.15, .18, .18), .87)
    if recipe.get("material_evidence"):
        apply_photo_material(gray, photo["patches"]["wristband_weave"]["path"], .022, strength=1.0)
    green = base._material(name+"_strap_fluorescent_edge", (.24, .40, .025), .80)
    pale = base._material(name+"_wrist_puck_pale_gray", (.55, .58, .57), .65)
    # A band is an elliptical surface, leaving the forearm continuous through it.
    for z in (-.079, -.049):
        points = [(.032*math.cos(i*math.pi/16), -.060+.027*math.sin(i*math.pi/16), z) for i in range(33)]
        base._tube(name+"_band_green_edge", points, .0012, green, rig)
    verts, faces = [], []
    for z in (-.078, -.050):
        verts.extend([(.032*math.cos(i*math.pi/16), -.060+.027*math.sin(i*math.pi/16), z) for i in range(32)])
    for i in range(32): faces.append((i,(i+1)%32,(i+1)%32+32,i+32))
    mesh=bpy.data.meshes.new(name+"_band_mesh"); mesh.from_pydata(verts,[],faces); mesh.update()
    band=bpy.data.objects.new(name+"_band",mesh); bpy.context.collection.objects.link(band); band.parent=rig; mesh.materials.append(gray)
    solid=band.modifiers.new("band_thickness", "SOLIDIFY"); solid.thickness=.0012
    base._ellipsoid(name+"_white_wrist_module", (0, -.091, -.063), (.021, .009, .016), pale, rig)
    _rounded_box(name+"_white_module_base", (0, -.089, -.065), (.041,.015,.031), pale, rig,.006)
    _rounded_box(name+"_mount_base", (0,-.027,-.061), (.051,.022,.032), black, rig)
    base._ellipsoid(name+"_mount_ball", (0,-.010,-.041), (.016,.016,.016), black, rig)
    # The short ball joint is supported by the cuff, as in f0; there is no long
    # free-standing stem between the wrist and camera.
    base._tube(name+"_mount_link", [(0,-.020,-.057),(0,-.010,-.041),(0,-.002,-.030)],.009,black,rig)
    _rounded_box(name+"_quick_release", (0,0,-.027), (.043,.026,.009),black,rig,.0015)
    if core:
        _rounded_box(name+"_cable_connector",(.039,.003,-.004),(.008,.012,.023),black,rig,.001)
        base._tube(name+"_attached_cable",[(.040,.003,-.009),(.046,.012,-.050),(.041,-.037,-.139),(.025,-.047,-.28),(.05,-.062,-.45)],.0017,black,rig)
    if rig_options.get("camera_head_rotation_xyz_deg"):
        from data.action4_blender_foundation import apply_camera_head_pose
        apply_camera_head_pose(rig, core, name, rig_options["camera_head_rotation_xyz_deg"])
    rig.location=position; rig.rotation_euler=tuple(math.radians(x) for x in rotation)
    bpy.context.view_layer.update()
    return {"rig":rig,"core":core,"parts":[core,*core.children] if core else [],"hand":hand,"forearm":forearm}


def _denim_material(name, evidence=None):
    from data import action4_render as base
    mat=base._material(name,(.055,.095,.14),.86)
    nodes,links=mat.node_tree.nodes,mat.node_tree.links
    coords=nodes.new("ShaderNodeTexCoord")
    noise=nodes.new("ShaderNodeTexNoise"); noise.inputs["Scale"].default_value=38; noise.inputs["Detail"].default_value=3
    links.new(coords.outputs["Object"],noise.inputs["Vector"])
    ramp=nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position=.18; ramp.color_ramp.elements[0].color=(.031,.052,.078,1)
    ramp.color_ramp.elements[1].position=.85; ramp.color_ramp.elements[1].color=(.145,.19,.23,1)
    links.new(noise.outputs["Fac"],ramp.inputs[0]); links.new(ramp.outputs[0],nodes.get("Principled BSDF").inputs["Base Color"])
    wave=nodes.new("ShaderNodeTexWave"); wave.wave_type="BANDS"; wave.bands_direction="DIAGONAL"; wave.inputs["Scale"].default_value=1350
    links.new(coords.outputs["Object"],wave.inputs["Vector"])
    bump=nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value=.18; bump.inputs["Distance"].default_value=.0003
    links.new(wave.outputs["Color"],bump.inputs["Height"]); links.new(bump.outputs["Normal"],nodes.get("Principled BSDF").inputs["Normal"])
    if evidence:
        from data.action4_blender_foundation import apply_denim_material
        apply_denim_material(mat, evidence)
    return mat


def _garment_panel(name, outline, material, height, phase):
    """A disconnected analytic garment panel with folds and boundary curves."""
    from data import action4_render as base
    import mathutils.geometry
    from mathutils import Vector
    bpy=base._bpy()
    # Tessellate a panel outline, then subdivide so folds change its geometry.
    mesh=bpy.data.meshes.new(name+"_mesh")
    mesh.from_pydata([(x,y,height) for x,y in outline],[],[tuple(range(len(outline)))]); mesh.update()
    obj=bpy.data.objects.new(name,mesh); bpy.context.collection.objects.link(obj); mesh.materials.append(material)
    import bmesh
    bm=bmesh.new(); bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm,faces=list(bm.faces))
    bmesh.ops.subdivide_edges(bm,edges=list(bm.edges),cuts=10,use_grid_fill=True)
    for v in bm.verts:
        x,y=v.co.x,v.co.y
        v.co.z+=.005*math.sin(30*x+phase)*math.cos(24*y)+.003*math.sin(60*y+phase)
    bm.to_mesh(mesh); bm.free()
    uv = mesh.uv_layers.new(name="metric_xy_0p1m")
    for loop in mesh.loops:
        vertex = mesh.vertices[loop.vertex_index].co
        uv.data[loop.index].uv = (vertex.x / .1, vertex.y / .1)
    for face in mesh.polygons:face.use_smooth=True
    solid=obj.modifiers.new("denim_thickness", "SOLIDIFY"); solid.thickness=.0012
    # Offset dark stitching follows the same panel fold equation at its boundary.
    seam_mat=base._material(name+"_stitch",(.20,.22,.21),.95)
    points=[]
    for (x0,y0),(x1,y1) in zip(outline,outline[1:]+outline[:1]):
        for t in range(8):
            q=t/8; x=x0+(x1-x0)*q; y=y0+(y1-y0)*q
            z=height+.0017+.005*math.sin(30*x+phase)*math.cos(24*y)+.003*math.sin(60*y+phase)
            points.append((x,y,z))
    points.append(points[0]); base._tube(name+"_boundary_stitch",points,.00045,seam_mat)
    return obj



def _connected_garment(recipe, phase):
    """Continuous front/sleeve surface with overlapping pocket/collar details."""
    from data import action4_render as base
    denim=_denim_material("connected_jacket_denim",recipe.get("material_evidence",{}).get("denim_manifest"))
    outline=[(-.06,.40),(-.20,.34),(-.45,.12),(-.32,.04),(-.15,.20),
             (-.16,-.10),(.16,-.10),(.15,.20),(.32,.04),(.45,.12),
             (.20,.34),(.06,.40),(.025,.33),(-.025,.33)]
    panels=[_garment_panel("jacket_connected_front_sleeves",outline,denim,.006,phase)]
    for side in (-1,1):
        pocket=[(.05*side,.17),(.125*side,.17),(.125*side,.25),(.05*side,.25)]
        collar=[(.015*side,.33),(.065*side,.27),(.115*side,.355),(.055*side,.40)]
        if side<0:pocket.reverse();collar.reverse()
        panels.append(_garment_panel(f"chest_pocket_{side}",pocket,denim,.010,phase))
        panels.append(_garment_panel(f"collar_{side}",collar,denim,.012,phase))
    metal=base._material("connected_jacket_buttons",(.30,.32,.31),.38,.65)
    for i in range(6):
        base._ellipsoid("jacket_metal_button",(.010,-.075+i*.065,.016),(.007,.007,.0016),metal)
    return panels


def _pin_grasped_hem(garments, rigs, region="hem"):
    """Pin a continuous hem to actual posed skin surfaces, preserving the body."""
    from data import action4_render as base
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    bpy=base._bpy();contacts=[];deps=bpy.context.evaluated_depsgraph_get()
    for rig in rigs:
        hand=rig["hand"];evaluated=hand.evaluated_get(deps);mesh=evaluated.to_mesh()
        try:
            local=[v.co.copy() for v in mesh.vertices if v.co.z>.018 and v.co.y>-.025]
            if not local:raise ValueError("No finite fingertip skin region")
            world=[hand.matrix_world@p for p in local]
            # Cloth is supported at the lower posed finger surface, rather than
            # the nearest surface to a centroid inside the curled-finger void.
            support=sorted(world,key=lambda p:p.z)[:max(1,len(world)//10)]
            center=sum(support,Vector())/len(support)
            tree=BVHTree.FromObject(hand,deps)
            hit,normal,_,distance=tree.find_nearest(hand.matrix_world.inverted()@center)
            if hit is None:raise ValueError("Fingertip surface query failed")
            surface=hand.matrix_world@hit
            # Thin hem lies just below the selected real skin point.
            contacts.append(surface-Vector((0,0,.0008)))
        finally:evaluated.to_mesh_clear()
    contacts.sort(key=lambda p:p.x);left,right=contacts
    if region not in ("hem","left_sleeve","right_sleeve"):raise ValueError("Unsupported local cloth contact region")
    crease_amplitude=.009 if region!="hem" or any(g.get("garment_state") for g in garments) else 0.
    def deform(point, rest=None):
        x,y,z=point
        rx,ry,rz = rest if rest is not None else point
        if region != "hem":
            signed_x = -rx if region=="left_sleeve" else rx
            edge_x=.32+1.625*(ry-.04)
            # Orthogonal coordinates relative to the original distal sleeve edge.
            xx=-rx if region=="left_sleeve" else rx
            t=((xx-.32)*.13+(ry-.04)*.08)/(.13*.13+.08*.08)
            depth=max(0.,(-(xx-.32)*.08+(ry-.04)*.13)/math.sqrt(.13*.13+.08*.08))
            gate=max(0.,min(1.,(xx-.17)/.13))
            w=math.exp(-(depth/.16)**2)*gate*gate*(3-2*gate)
            if region=="right_sleeve":t=1-t
            pin=left.lerp(right,t)
            supported=pin+Vector((0,depth*.65,-depth*.30))
            supported.z+=.003*math.sin(3*math.pi*t)*min(1.,depth/.05)
            return Vector(point)*(1-w)+supported*w
        w=max(0.,min(1.,(.18-ry)/.28))**2
        t=max(0.,min(1.,(rx+.16)/.32))
        pin=left.lerp(right,t)
        crease=crease_amplitude*math.sin(4*math.pi*t)*w*w
        return Vector((x+w*(pin.x-rx),y+w*(pin.y-ry),z+w*(pin.z-rz)+crease))
    for garment in garments:
        rest=garment.get("contact_rest_coordinates")
        for vertex in garment.data.vertices:
            prior=rest[3*vertex.index:3*vertex.index+3] if rest is not None else None
            vertex.co=deform(vertex.co,prior)
    for obj in bpy.context.scene.objects:
        if obj.type=="CURVE" and "boundary_stitch" in obj.name:
            rest=obj.get("contact_rest_coordinates");offset=0
            for spline in obj.data.splines:
                for point in spline.bezier_points:
                    prior=rest[offset:offset+3] if rest is not None else None
                    point.co=deform(point.co,prior);offset+=3
        elif obj.name.startswith("jacket_metal_button"):
            obj.location=deform(obj.location,obj.get("contact_rest_location"))
    bpy.context.view_layer.update()
    distances=[]
    for rig in rigs:
        skin=BVHTree.FromObject(rig["hand"],bpy.context.evaluated_depsgraph_get())
        inv=rig["hand"].matrix_world.inverted()
        values=[]
        for garment in garments:
            if garment.name!="jacket_connected_front_sleeves":continue
            evaluated=garment.evaluated_get(bpy.context.evaluated_depsgraph_get())
            mesh=evaluated.to_mesh()
            try:
                for vertex in mesh.vertices:
                    world=garment.matrix_world@vertex.co
                    hit,*_=skin.find_nearest(inv@world)
                    if hit is not None:values.append((world-rig["hand"].matrix_world@hit).length)
            finally:evaluated.to_mesh_clear()
        distances.append(min(values))
    if any(x>.006 for x in distances):raise ValueError("Hem did not reach both hands within 6mm")
    return {"method":"existing cloth contact region pinned to evaluated fingertip skin; analytic local drape","contact_region":region,
            "contact_points_world_m":[list(left),list(right)],"minimum_hand_cloth_distance_m":distances,"distance_basis":"evaluated cloth vertices to evaluated skin triangle surfaces",
            "claim_scope":"geometric local contact; no measured human articulation or cloth equilibrium"}

def _environment(recipe, family, phase):
    from data import action4_render as base
    bpy=base._bpy()
    wall=base._material("neutral_gray_wall",(.23,.245,.255),.98)
    if family["environment"]=="wall":
        _rounded_box("wall",(0,-.58,.35),(2.5,.012,2.3),wall)
        return []
    table_xy=family.get("table_center_xy_m",[0,.17])
    table_size=family.get("table_size_xy_m",[1.12,.80])
    frequency=float(family.get("table_grid_frequency_per_m",40))
    if len(table_xy)!=2 or len(table_size)!=2 or not 15<=frequency<=60 or any(not .5<=float(v)<=1.5 for v in table_size):
        raise ValueError("Invalid bounded table appearance prior")
    table_material=base._grid_material()
    for node in table_material.node_tree.nodes:
        if node.type=="MATH" and node.operation=="MULTIPLY":node.inputs[1].default_value=frequency
    _rounded_box("table_surface",(table_xy[0],table_xy[1],-.010),(table_size[0],table_size[1],.016),table_material,bevel=.001)
    _rounded_box("back_wall",(0,.74,.65),(2.5,.012,1.5),wall)
    _rounded_box("floor",(0,0,-.20),(3.,3.,.01),base._material("floor_neutral_gray_brown",(.13,.12,.105),.9))
    dark=base._material("foreground_black_clothes",(.004,.005,.007),.95)
    foreground_y=float(family.get("foreground_cloth_center_y_m",-.52))
    foreground_z=float(family.get("foreground_cloth_z_m",.00))
    if not -.85<=foreground_y<=-.35 or not -.15<=foreground_z<=.05:raise ValueError("Foreground cloth prior outside bounds")
    base._cloth("near_dark_clothing",(0,foreground_y,foreground_z),(.95,.45),dark,phase)
    if recipe.get("connected_garment",False):
        return _connected_garment(recipe,phase)
    denim=_denim_material("jacket_denim_from_observed_palette", recipe.get("material_evidence", {}).get("denim_manifest"))
    panels=[]
    for side in (-1,1):
        body=[(.012*side,-.065),(.17*side,-.065),(.19*side,.28),(.11*side,.39),(.015*side,.32)]
        if side<0:body.reverse()
        panels.append(_garment_panel(f"jacket_body_{side}",body,denim,.006,phase))
        sleeve=[(.15*side,.24),(.32*side,.07),(.45*side,.15),(.23*side,.37)]
        if side<0:sleeve.reverse()
        panels.append(_garment_panel(f"jacket_sleeve_{side}",sleeve,denim,.008,phase+.5))
        pocket=[(.05*side,.17),(.125*side,.17),(.125*side,.25),(.05*side,.25)]
        if side<0:pocket.reverse()
        panels.append(_garment_panel(f"chest_pocket_{side}",pocket,denim,.014,phase))
        collar=[(.015*side,.33),(.065*side,.27),(.115*side,.355),(.055*side,.40)]
        if side<0:collar.reverse()
        panels.append(_garment_panel(f"collar_{side}",collar,denim,.025,phase))
    metal=base._material("brushed_metal_buttons",(.30,.32,.31),.38,.65)
    for i in range(6):
        y=-.043+i*.062
        base._ellipsoid("jacket_metal_button",(.010,y,.016),(.007,.007,.0016),metal)
    return panels


def _lift_garment_edge(garments, rigs):
    """Lift the near hem toward the two posed fingertips with a smooth cloth field.

    This constructs pinned contact geometry rather than a floating rectangle.
    It is an analytic drape approximation; no claim of physical cloth equilibrium.
    """
    from data import action4_render as base
    from mathutils import Vector
    bpy = base._bpy()
    contacts = []
    for rig in rigs:
        hand = rig["hand"]
        points = [hand.matrix_world @ vertex.co for vertex in hand.data.vertices if vertex.co.z > .018 and vertex.co.y > -.025]
        if not points:
            raise ValueError("No posed fingertip contact region for garment lift")
        centre = sum(points, Vector((0, 0, 0))) / len(points)
        contacts.append(centre)
    contacts.sort(key=lambda p: p.x)
    left, right = contacts
    def deform(point):
        x,y,z = point
        weight = max(0., min(1., (.18-y)/.245)) ** 2
        t = max(0., min(1., (x+.17)/.34))
        contact = left.lerp(right, t)
        return (x+weight*(contact.x-x), y+weight*(contact.y+.065), z+weight*(contact.z-.006))
    for garment in garments:
        for vertex in garment.data.vertices:
            vertex.co = deform(vertex.co)
    for obj in bpy.context.scene.objects:
        if obj.type == "CURVE" and "boundary_stitch" in obj.name:
            for spline in obj.data.splines:
                for point in spline.bezier_points:
                    point.co = deform(point.co)
        elif obj.name.startswith("jacket_metal_button"):
            obj.location = deform(obj.location)
    bpy.context.view_layer.update()
    return {"method":"smooth near-hem displacement pinned to posed fingertip regions",
            "contact_points_world_m":[list(left),list(right)],
            "limitation":"analytic drape; does not solve textile tension or friction"}


def _collision_pairs(rigs, garments=()):
    """Check target/hand, forearm, garment and target surface intersections."""
    from data import action4_render as base
    from mathutils.bvhtree import BVHTree
    bpy=base._bpy(); deps=bpy.context.evaluated_depsgraph_get()
    def tree(obj):
        ev=obj.evaluated_get(deps); mesh=ev.to_mesh()
        vertices=[ev.matrix_world@v.co for v in mesh.vertices]
        polygons=[list(p.vertices) for p in mesh.polygons]
        result=BVHTree.FromPolygons(vertices,polygons)
        ev.to_mesh_clear(); return result
    cores=[r["core"] for r in rigs if r["core"]]
    obstacles=[r[field] for r in rigs for field in ("hand","forearm")]+list(garments)
    checks=[]
    for i,core in enumerate(cores):
        a=tree(core)
        for other in cores[i+1:]+obstacles:
            overlaps=a.overlap(tree(other))
            if overlaps:checks.append({"a":core.name,"b":other.name,"intersecting_triangle_pairs":len(overlaps)})
    return checks


def _render_device(scene, settings):
    """Require selected GPU IDs; unavailable GPU rendering never falls back to CPU."""
    from data import action4_render as base
    bpy = base._bpy()
    kind = settings["device"]
    if kind == "CPU":
        scene.cycles.device = "CPU"
        return {"backend":"CPU","devices":[]}
    preferences = bpy.context.preferences.addons["cycles"].preferences
    try:
        preferences.compute_device_type = kind
        preferences.get_devices()
    except (TypeError, RuntimeError) as error:
        raise RuntimeError(f"Requested Cycles {kind} backend is unavailable") from error
    available = {d.id: d for d in preferences.devices if d.type == kind}
    uuid = settings.get("worker_gpu_uuid")
    if uuid is not None:
        from data.action4_devices import gpu_identity
        identity = gpu_identity(uuid)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid or len(available) != 1:
            raise RuntimeError("Cycles did not expose exactly the assigned single GPU")
        domain, bus, slot = identity["pci_bus_id"].lower().split(":")
        pci = f"{int(domain,16):04x}:{bus}:{slot.split('.')[0]}"
        selected = [identifier for identifier in available if pci in identifier.lower()]
        if len(selected) != 1:
            raise RuntimeError("Cycles PCI identity differs from the assigned physical GPU")
    else:
        raise RuntimeError("GPU rendering requires an explicit single-worker UUID")
    if not selected or any(identifier not in available for identifier in selected):
        raise RuntimeError(f"Requested {kind} GPU IDs are unavailable; available IDs: {sorted(available)}")
    for device in preferences.devices:
        device.use = device.id in selected and device.type == kind
    scene.cycles.device = "GPU"
    return {"backend":kind,"physical_uuid":uuid,"devices":[{"id":identifier,"name":available[identifier].name} for identifier in selected]}


def search_occlusion(recipe_path, output_path, max_trials=27):
    """Bounded geometric search for near-double occlusion without RGB rendering."""
    from data import action4_render as base
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    from bpy_extras.object_utils import world_to_camera_view
    import numpy as np
    recipe = validate_recipe(_read(_path(recipe_path, exists=True)))
    if not 1 <= max_trials <= 125:
        raise ValueError("Set an explicit search budget between 1 and 125")
    bpy = base._bpy()
    width, height = recipe["resolution_wh"]
    scene = base._reset_scene(width, height, 1)
    family = next(f for f in recipe["families"] if f["name"] == "near_double")
    camera = base._camera(family["camera_location_m"], family["camera_target_m"])
    rigs = [_wrist_rig(recipe, f"search_wrist_{j}", family["anchor_positions_m"][j], family["anchor_rotations_xyz_deg"][j], -1 if j == 0 else 1, j+1, family["finger_curl_rad"]) for j in range(2)]
    corners = np.array(_read(_path(recipe["asset_dir"])/"asset_manifest.json")["corners_3d_mm"])*.001
    first = rigs[0]["rig"].location.copy()
    toward = (camera.location-first).normalized()
    right = camera.matrix_world.to_3x3() @ Vector((1,0,0))
    up = camera.matrix_world.to_3x3() @ Vector((0,1,0))
    trials = []
    selected = None
    for separation in (.065, .085, .105):
        for lateral in (.018, .027, .036):
            for vertical in (-.012, 0., .012):
                if len(trials) >= max_trials:
                    break
                position = first+toward*separation+right*lateral+up*vertical
                rigs[1]["rig"].location = position
                bpy.context.view_layer.update()
                collision = _collision_pairs(rigs)
                trial = {"trial_index":len(trials), "second_position_m":list(position),
                         "separation_toward_capture_m":separation, "screen_lateral_m":lateral,
                         "screen_vertical_m":vertical, "surface_intersections":collision}
                if collision:
                    trial["status"] = "rejected_surface_intersection"
                    trials.append(trial)
                    continue
                core = rigs[0]["core"]
                deps = bpy.context.evaluated_depsgraph_get()
                evaluated = core.evaluated_get(deps)
                mesh = evaluated.to_mesh()
                tree = BVHTree.FromPolygons([evaluated.matrix_world@v.co for v in mesh.vertices], [list(p.vertices) for p in mesh.polygons])
                evaluated.to_mesh_clear()
                points = [world_to_camera_view(scene,camera,core.matrix_world@Vector(c)) for c in corners]
                if any(p.z <= 0 for p in points):
                    trial["status"]="rejected_negative_depth";trials.append(trial);continue
                lo = np.min([[p.x,p.y] for p in points],axis=0);hi=np.max([[p.x,p.y] for p in points],axis=0)
                if np.any(lo<0) or np.any(hi>1):
                    trial["status"]="rejected_outside_capture";trials.append(trial);continue
                amodal = other = visible = 0
                for u in np.linspace(lo[0],hi[0],24):
                    for v in np.linspace(lo[1],hi[1],24):
                        ray=camera.matrix_world.to_3x3()@Vector(((u-.5)*32/26,(v-.5)*32/26*height/width,-1))
                        ray.normalize()
                        hit,normal,face,distance=tree.ray_cast(camera.location,ray)
                        if hit is None:continue
                        amodal+=1
                        is_hit,location,normal,face,obj,matrix=scene.ray_cast(deps,camera.location,ray)
                        if is_hit and obj==core:visible+=1
                        elif is_hit and obj==rigs[1]["core"]:other+=1
                trial.update(ray_grid=[24,24],amodal_ray_hits=amodal,visible_ray_hits=visible,
                             other_camera_occlusion_ray_hits=other,
                             other_camera_occlusion_fraction=other/amodal if amodal else None)
                fraction = other/amodal if amodal else 0
                trial["status"]="geometric_candidate" if .15<=fraction<=.55 and visible>0 else "rejected_occlusion_range"
                trials.append(trial)
                if trial["status"]=="geometric_candidate" and selected is None:selected=trial
            if len(trials)>=max_trials:break
        if len(trials)>=max_trials:break
    result={"status":"search_completed_not_visual_acceptance", "trials":trials, "selected":selected,
            "first_position_m":list(first), "first_rotation_xyz_deg":family["anchor_rotations_xyz_deg"][0],
            "second_rotation_xyz_deg":family["anchor_rotations_xyz_deg"][1],
            "criterion":"15–55% sampled rays occluded by the other camera itself, positive capture depth, in-frame cuboid, no checked surface intersections",
            "limitations":["24x24 ray sampling is a search estimate, not final pixel mask accounting","Generic pose/material realism remains subject to visual review","BVH surface intersection does not detect complete containment"],
            "recipe_sha256":_hash(_path(recipe_path,exists=True)), "synthesis_code_sha256":_hash(__file__)}
    _write(output_path,result)
    return result


def _render_one(plan, entry, output_dir):
    if plan["recipe"].get("asset_backend") == "modular_composed":
        from data import action4_modular_synthesis_backend as modular
        return modular.render_entry(plan, entry, output_dir)
    from data import action4_render as base
    import numpy as np
    from mathutils import Vector
    from bpy_extras.object_utils import world_to_camera_view
    bpy=base._bpy(); recipe=plan["recipe"]
    family=next(f for f in recipe["families"] if f["name"]==entry["scene_family"])
    width,height=recipe["resolution_wh"]
    rng=np.random.default_rng(entry["seed"])
    scene=base._reset_scene(width,height,recipe["render"]["samples"])
    device_info=_render_device(scene,recipe["render"])
    denoiser=_set_denoiser(scene,recipe["render"])
    scene.cycles.seed=entry["seed"]
    scene.cycles.max_bounces=6; scene.cycles.transparent_max_bounces=4
    scene.view_settings.look="Medium High Contrast"
    controls=entry.get("request_recipe")
    if controls:
        validate_request_recipe(controls,{f["name"] for f in recipe["families"]})
    perturb=controls["allowed_perturbations"] if controls else None
    exposure=family["exposure_ev"]+(perturb["exposure_delta_ev"] if perturb else float(rng.uniform(-.15,.15)))
    scene.view_settings.exposure=exposure
    phase=random.Random(entry["background_seed"]).uniform(-math.pi,math.pi)
    garments=_environment(recipe,family,phase)
    garment_state=family.get("garment_state","laid_open")
    if garments and garment_state != "laid_open":
        from data.action4_blender_foundation import fold_existing_garment
        fold_existing_garment(garments,garment_state)
    camera_location=family["camera_location_m"]
    if controls:
        camera_location=[t+(p-t)/controls["scale"] for p,t in zip(camera_location,family["camera_target_m"])]
    camera=base._camera(camera_location,family["camera_target_m"])
    focal_length=float(family.get("capture_focal_mm",26))
    if not 18<=focal_length<=32:raise ValueError("Capture focal prior outside bounded range")
    camera.data.lens=focal_length
    focal_length=float(camera.data.lens)
    capture_roll = float(family.get("capture_roll_deg", 0.0))
    if not -25 <= capture_roll <= 25:
        raise ValueError("Capture roll appearance prior outside bounded range")
    camera.rotation_euler.rotate_axis("Z", math.radians(capture_roll))
    bpy.context.view_layer.update()
    light_energy=family["area_light_energy_w"]*(perturb["light_intensity_gain"] if perturb else float(rng.uniform(.9,1.1)))
    base._light(family["light_location_m"],light_energy,family["light_size_m"])
    rigs=[]; placement=[]
    for j,side in enumerate((-1,1)):
        position=list(family["anchor_positions_m"][j])
        position_jitter=perturb["position_jitter_m"] if perturb else recipe["position_jitter_m"]
        rotation_jitter=perturb["rotation_jitter_deg"] if perturb else recipe["rotation_jitter_deg"]
        delta=controls["view"]["wrist_delta_xyz_deg"] if controls else [0,0,0]
        position=[p+float(rng.uniform(-d,d)) for p,d in zip(position,position_jitter)]
        rotation=[r+d+float(rng.uniform(-rotation_jitter,rotation_jitter)) for r,d in zip(family["anchor_rotations_xyz_deg"][j],delta)]
        target_index=target_indices(entry)[j]
        rig_options = dict(family.get("rig_options", {}))
        if family.get("rig_options_by_side"):
            if len(family["rig_options_by_side"]) != 2: raise ValueError("Two side-specific rig priors required")
            rig_options.update(family["rig_options_by_side"][j])
        rig=_wrist_rig(recipe,"left_wrist" if side<0 else "right_wrist",position,rotation,side,target_index,family["finger_curl_rad"],rig_options)
        rigs.append(rig); placement.append({"side":side,"position_m":position,"rotation_xyz_degrees":rotation,"has_target":bool(target_index),"rig_options":rig_options})
    if family.get("contact_mode") in ("grasped_hem","held_left_cuff","held_right_cuff"):
        region={"grasped_hem":"hem","held_left_cuff":"left_sleeve","held_right_cuff":"right_sleeve"}[family["contact_mode"]]
        garment_lift=_pin_grasped_hem(garments,rigs,region)
    else:
        garment_lift=_lift_garment_edge(garments,rigs) if family["name"]=="cloth_lift" else None
    intersections=_collision_pairs(rigs,garments)
    if intersections:
        raise ValueError("Rejected geometry intersections: "+json.dumps(intersections))
    bpy.context.view_layer.update()
    out=output_dir; sid=entry["sample_id"]
    nodes=base._mask_nodes(out/"masks")
    for j,node in enumerate(nodes,1):node.file_slots[0].path=f"{sid}_i{j}_visible_"
    scene.render.filepath=str(out/"rgb"/(sid+".png")); bpy.ops.render.render(write_still=True)
    # Save the editable scene before any temporary amodal changes.
    if entry["index"]<len(recipe["families"]):
        bpy.data.texts.new("synthesis_recipe.json").write(json.dumps(recipe,ensure_ascii=False,indent=2))
        bpy.data.texts.new("source_sample.json").write(json.dumps(entry,ensure_ascii=False,indent=2))
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(out/"scenes"/(sid+".blend")),check_existing=False)
    asset=_read(_path(recipe["asset_dir"])/"asset_manifest.json")
    corners=np.array(asset["corners_3d_mm"])*.001
    camera_to_world=[list(row) for row in camera.matrix_world]
    hidden={obj.name:obj.hide_render for obj in scene.objects}
    instances=[]; excluded=[]
    for rig in rigs:
        core=rig["core"]
        if core is None:continue
        j=core.pass_index
        projected=[]; corner_visible=[]
        for corner in corners:
            point=core.matrix_world@Vector(corner)
            v=world_to_camera_view(scene,camera,point)
            if v.z<=0:raise ValueError("Corner behind capture plane")
            projected.append([float(v.x*width-.5),float((1-v.y)*height-.5)])
            direction=point-camera.location;distance=direction.length
            hit,location,*_=scene.ray_cast(bpy.context.evaluated_depsgraph_get(),camera.location,direction.normalized(),distance=distance+1e-6)
            corner_visible.append(bool((not hit or (location-camera.location).length>=distance-1e-5) and 0<=v.x<=1 and 0<=v.y<=1))
        projected=np.array(projected)
        if (projected[:,0]<-width-.5).any() or (projected[:,0]>2*width-.5).any() or (projected[:,1]<-height-.5).any() or (projected[:,1]>2*height-.5).any():
            raise ValueError("Expanded canvas does not contain target")
        for obj in scene.objects:
            if obj.type not in ("CAMERA","LIGHT"):obj.hide_render=obj not in rig["parts"]
        for k,node in enumerate(nodes,1):node.file_slots[0].path=f"{sid}_i{k}_amodal_for_{j}_"
        scene.render.resolution_x=3*width;scene.render.resolution_y=3*height;camera.data.sensor_width=96
        previous_samples, previous_denoising = scene.cycles.samples, scene.cycles.use_denoising
        scene.cycles.samples=1;scene.cycles.use_denoising=False
        vp=out/"masks"/f"{sid}_i{j}_visible_0001.png"
        ap=out/"masks"/f"{sid}_i{j}_amodal_for_{j}_0001.png"
        amodal_info = {"mode": "full"}
        try:
            if recipe["render"].get("amodal_mode", "full") == "crop":
                from data.action4_amodal import render_crop
                full, amodal_info = render_crop(scene, camera, core, rig["parts"], corners, nodes, j, ap,
                                               _run_output_dir(out) / "amodal_crop" / f"{sid}_i{j}")
            else:
                bpy.ops.render.render(write_still=False)
                full=base._mask_array(ap)
        finally:
            # Restore before rejection/continue or the following instance's projection.
            scene.render.resolution_x=width;scene.render.resolution_y=height;camera.data.sensor_width=32
            scene.cycles.samples=previous_samples;scene.cycles.use_denoising=previous_denoising
        visible=base._mask_array(vp)
        amodal=full[height:2*height,width:2*width]
        av,ai,af=int(visible.sum()),int(amodal.sum()),int(full.sum())
        if af==0:raise ValueError("Empty amodal mask")
        if ai==0:
            excluded.append({"instance_id":j,"reason":"outside_image","area_full_amodal_px":af});continue
        nonnested=int((visible&~amodal).sum())
        if nonnested>max(4,.005*ai):raise ValueError("Mask boundary mismatch exceeds fixed tolerance")
        from data.action4_adapter import DataAdapter
        affine,_=DataAdapter.letterbox_affine((width,height),(1024,768))
        visible_input_area=av*float(affine[0,0]*affine[1,1])
        low_support=visible_input_area<16 or av/ai<.1
        inst={"instance_id":j,"object_id":1,"corners_px":projected.tolist(),"bbox_full_xyxy":[float(projected[:,0].min()),float(projected[:,1].min()),float(projected[:,0].max()),float(projected[:,1].max())],
              "corner_visible":corner_visible,"object_to_world_m":[list(row) for row in core.matrix_world],
              "mask_visible":str(vp.relative_to(ROOT)),"mask_amodal_expanded":str(ap.relative_to(ROOT)),
              "mask_visible_sha256":_hash(vp),"mask_amodal_expanded_sha256":_hash(ap),
              "amodal_expanded_origin_px":[-width,-height],"amodal_expanded_shape_hw":[3*height,3*width],
              "amodal_render":amodal_info,
              "area_visible_px":av,"area_inframe_amodal_px":ai,"area_full_amodal_px":af,
              "occlusion_inframe":float(max(0,1-av/ai)),"occlusion_inframe_raw":float(1-av/ai),"truncation":float(1-ai/af),
              "mask_nonnested_boundary_pixels":nonnested,
              "visible_area_at_fullinput_px":visible_input_area,
              "ignore":bool(low_support),"ignore_reason":"low_visible_support" if low_support else None}
        yy,xx=np.nonzero(visible)
        if len(xx):inst["bbox_visible_xyxy"]=[float(xx.min()-.5),float(yy.min()-.5),float(xx.max()+.5),float(yy.max()+.5)]
        instances.append(inst)
        scene.render.resolution_x=width;scene.render.resolution_y=height;camera.data.sensor_width=32
    for obj in scene.objects:obj.hide_render=hidden[obj.name]
    record={"schema_version":"action4.v1",**entry,"image":str((out/"rgb"/(sid+".png")).relative_to(ROOT)),
            "width":width,"height":height,"instances":instances,"excluded_instances":excluded,
            "status":"render_candidate_not_training_accepted","target_count":len(instances),"planned_target_count":entry["target_count"],
            "asset_version":asset["asset_version"],"asset_backend":recipe.get("asset_backend", "procedural_r3"),
            "foundation_sha256":_hash(_path(recipe["foundation_blend"])) if recipe.get("asset_backend")=="blend_r4" else None,
            "source_stl_sha256":asset["source_sha256"],"corners_3d_mm":asset["corners_3d_mm"],
            "camera":{"K":[[focal_length/32*width,0,(width-1)/2],[0,focal_length/32*width,(height-1)/2],[0,0,1]],"camera_to_world_m":camera_to_world,
                      "calibration_status":"synthetic prior; real capture calibration unknown","pixel_convention":"integer_centers"},
            "render_settings":{"samples":recipe["render"]["samples"],"mask_samples":1,"exposure_ev":exposure,"light_energy_w":light_energy,"cpu_threads":int(scene.render.threads),"device":device_info,"denoiser":denoiser,"amodal_mode":recipe["render"].get("amodal_mode","full"),"amodal_helper_sha256":_hash(ROOT / "src/data/action4_amodal.py")},
            "evidence_frames":controls["reference_frame_ids"] if controls else family["evidence_frames"],"scene_parameters":{"wrists":placement,"background_phase":phase,"anatomy":"CC0 generic mesh, not source subject reconstruction","cloth_model":"continuous front/sleeve analytic cloth with overlapping pockets/collar; not physical cloth simulation" if recipe.get("connected_garment") else "disconnected analytic garment panels with seam curves; not a sewn garment or cloth dynamics simulation","garment_lift":garment_lift,"garment_state":garment_state,"capture_roll_deg":capture_roll,"capture_focal_mm":focal_length,"anchor_status":"visually informed design priors, not measured real 6D poses","executed_feedback_recipe":controls},
            "geometry_collision_check":{"method":"evaluated triangle BVH surface intersections: target/hand, target/forearm, target/garment, target/target","intersections":intersections,"limitation":"does not detect complete containment or establish soft-tissue contact pressure or cloth physical equilibrium"},
            "synthesis_code_sha256":_hash(__file__),"render_base_sha256":_hash(Path(base.__file__)),"rgb_sha256":_hash(out/"rgb"/(sid+".png"))}
    return record


def _render_inside_blender(plan_path, output_dir, shard_index, num_shards, limit=None, *, gpu_uuid=None, entry_start=0, entry_count=None):
    plan=_read(_path(plan_path,exists=True))
    if plan['recipe'].get('asset_backend')!='modular_composed':
        return _render_inside_blender_legacy(plan_path,output_dir,shard_index,num_shards,limit,gpu_uuid=gpu_uuid,entry_start=entry_start,entry_count=entry_count)
    from data import action4_modular_synthesis_backend as modular
    plan_ref={'path':str(_path(plan_path,exists=True).relative_to(ROOT)),'sha256':_hash(plan_path)}
    entries=select_shard_entries(plan,shard_index,num_shards,limit,entry_start,entry_count)
    out=_path(output_dir,exists=True);run_dir=_run_output_dir(out)
    pending=out/'manifest.worker_pending.jsonl';manifest_path=out/'manifest.jsonl';completion_path=run_dir/'render_completion.json'
    if any(p.exists() or p.is_symlink() for p in (pending,manifest_path,completion_path)):raise FileExistsError('Worker attempt artifacts already exist')
    batch=None;publication=None;manifest_proof=None;manifest_digest=hashlib.sha256()
    try:
        with modular.RenderBatch(plan_ref,entries,expected_plan=plan,gpu_uuid=gpu_uuid) as batch:
            for name in ('rgb','masks','scenes'):(out/name).mkdir(exist_ok=False)
            run_dir.mkdir(parents=True,exist_ok=True);shutil.copy2(__file__,run_dir/'synthesis_source_snapshot.py')
            with pending.open('x',encoding='utf-8') as manifest:
                for entry in entries:
                    record=batch.render_entry(entry,out)
                    line=json.dumps(record,ensure_ascii=False)+chr(10)
                    if manifest.write(line)!=len(line):raise OSError('Incomplete pending worker manifest write')
                    manifest.flush();manifest_digest.update(line.encode('utf-8'))
                    print('SYNTHESIS_SAMPLE_RENDERED_PENDING_BATCH_CLOSE',entry['sample_id'],len(record['instances']),flush=True)
        # Source closure and every entry restoration precede public manifest.
        receipt=batch.receipt
        raw,proof=modular._ReportPublication._read_regular(pending)
        if proof['sha256']!=manifest_digest.hexdigest():raise ValueError('Pending manifest differs from validated worker records')
        import os
        os.link(pending,manifest_path,follow_symlinks=False);manifest_proof=proof
        complete={'sample_ids':[e['sample_id'] for e in entries],'status':'rendered_pending_independent_validation','worker_batch_closing':receipt,'manifest_ref':{'path':str(manifest_path.relative_to(ROOT)),'sha256':proof['sha256']},'training_accepted':False}
        canonical=json.dumps(complete,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        publication=modular._ReportPublication([(completion_path,canonical)],receipt['run_id']);publication.publish()
        batch.receipt
        if not modular._ReportPublication._matches(manifest_path,manifest_proof):raise ValueError('Worker manifest changed before completion')
        cleanup=modular._ReportPublication._safe_remove(pending,proof)
        if not cleanup['removed']:raise RuntimeError('Owned pending manifest cleanup incomplete: '+repr(cleanup))
    except BaseException as exc:
        if batch is not None:batch.poison(exc)
        failure=publication.abort(exc) if publication is not None else None
        rollback=modular._ReportPublication._safe_remove(manifest_path,manifest_proof) if manifest_proof is not None else None
        if run_dir.exists():
            import traceback
            _write(run_dir/'worker_batch_failure.json',{'status':'WORKER_BATCH_FAILED_NO_COMPLETION','error':repr(exc),'traceback':traceback.format_exc(),'publication_failure':failure,'manifest_rollback':rollback,'pending_manifest_retained':pending.exists(),'training_accepted':False})
        raise


def _render_inside_blender_legacy(plan_path, output_dir, shard_index, num_shards, limit=None, *, gpu_uuid=None, entry_start=0, entry_count=None):
    plan=_read(_path(plan_path,exists=True)); out=_path(output_dir,exists=True)
    if plan.get("synthesis_code_sha256") != _hash(__file__) or plan.get("render_base_sha256") != _hash(ROOT / "src/data/action4_render.py"):
        raise ValueError("Renderer changed after plan freeze")
    if plan.get("asset_identities") != _asset_identities(plan["recipe"]):
        raise ValueError("Scene or target assets changed after plan freeze")
    if _hash(_path(plan["recipe_path"],exists=True))!=plan["recipe_sha256"]:
        raise ValueError("Recipe file changed after plan freeze")
    plan["_runtime_plan_sha256"] = _hash(plan_path)
    plan["_runtime_plan_ref"] = {"path":str(_path(plan_path,exists=True).relative_to(ROOT)),"sha256":plan["_runtime_plan_sha256"]}
    entries=select_shard_entries(plan,shard_index,num_shards,limit,entry_start,entry_count)
    if plan["recipe"]["render"]["device"] != "CPU":
        if gpu_uuid not in plan["recipe"]["render"].get("gpu_uuids", []):
            raise ValueError("Worker UUID is outside frozen plan allowlist")
        plan["recipe"]["render"]["worker_gpu_uuid"] = gpu_uuid
    for name in ("rgb","masks","scenes"):(out/name).mkdir(exist_ok=False)
    run_dir = _run_output_dir(out)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__,run_dir/"synthesis_source_snapshot.py")
    with (out/"manifest.jsonl").open("x",encoding="utf-8") as manifest:
        for entry in entries:
            record=_render_one(plan,entry,out)
            manifest.write(json.dumps(record,ensure_ascii=False)+"\n");manifest.flush()
            print("SYNTHESIS_SAMPLE_COMPLETE",entry["sample_id"],len(record["instances"]),flush=True)
    _write(run_dir/"render_completion.json",{"sample_ids":[e["sample_id"] for e in entries],"status":"rendered_pending_independent_validation"})


def main():
    args=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    plan=sub.add_parser("plan");plan.add_argument("--recipe",required=True);plan.add_argument("--output",required=True);plan.add_argument("--count",type=int,default=10000);plan.add_argument("--seed",type=int,default=20261019)
    s1=sub.add_parser("plan-s1");s1.add_argument("--recipe",required=True);s1.add_argument("--requests",required=True);s1.add_argument("--output",required=True);s1.add_argument("--seed",type=int,default=20261020);s1.add_argument("--max-count",type=int,default=2000)
    assets=sub.add_parser("build-assets");assets.add_argument("--source-asset-dir",required=True);assets.add_argument("--hand-bundle",required=True);assets.add_argument("--output-asset-dir",required=True);assets.add_argument("--output-hand-dir",required=True)
    search=sub.add_parser("search-occlusion");search.add_argument("--recipe",required=True);search.add_argument("--output",required=True);search.add_argument("--max-trials",type=int,default=27)
    for name in ("render-shard","_render-shard"):
        p=sub.add_parser(name);p.add_argument("--plan",required=True);p.add_argument("--output",required=True);p.add_argument("--shard-index",type=int,default=0);p.add_argument("--num-shards",type=int,default=1);p.add_argument("--limit",type=int)
        p.add_argument("--gpu-uuid");p.add_argument("--entry-start",type=int,default=0);p.add_argument("--entry-count",type=int)
        if name=="render-shard":p.add_argument("--blender",required=True);p.add_argument("--threads",type=int,default=2)
    validate=sub.add_parser("validate");validate.add_argument("--output",required=True)
    join=sub.add_parser("assemble");join.add_argument("--plan",required=True);join.add_argument("--shards",nargs="+",required=True);join.add_argument("--output",required=True);join.add_argument("--acceptance")
    a=parser.parse_args(args)
    if a.command=="plan":result=create_plan(a.recipe,a.output,a.count,a.seed)
    elif a.command=="plan-s1":result=create_s1_plan(a.recipe,a.requests,a.output,a.seed,a.max_count)
    elif a.command=="build-assets":result=build_assets(a.source_asset_dir,a.hand_bundle,a.output_asset_dir,a.output_hand_dir)
    elif a.command=="search-occlusion":result=search_occlusion(a.recipe,a.output,a.max_trials)
    elif a.command=="render-shard":result=render_shard(a.plan,a.output,a.blender,a.shard_index,a.num_shards,a.limit,a.threads,gpu_uuid=a.gpu_uuid,entry_start=a.entry_start,entry_count=a.entry_count)
    elif a.command=="_render-shard":return _render_inside_blender(a.plan,a.output,a.shard_index,a.num_shards,a.limit,gpu_uuid=a.gpu_uuid,entry_start=a.entry_start,entry_count=a.entry_count)
    elif a.command=="validate":result=validate_shard(a.output)
    else:result=assemble(a.plan,a.shards,a.output,a.acceptance)
    print(json.dumps({k:v for k,v in result.items() if k not in ("entries","recipe")},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
