"""One RGB frame → raw model prediction → native red cuboid → exact-PTS MP4."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from data.action4_paths import checked_root

import numpy as np

from data import DataAdapter
from model.training.action4_runtime import (canonical_hash, environment_identity, input_path,
                                           load_checkpoint, output_path, select_device, sha256_file,
                                           source_identity, write_json_new, validate_run_id)
from .action4_runner import build_model, inference_contract, predict_images

CUBOID_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7))


def clip_line(first, second, width: int, height: int):
    """Liang-Barsky clipping affects drawing only, never raw corner coordinates."""
    x0, y0 = map(float, first)
    x1, y1 = map(float, second)
    if not np.isfinite([x0, y0, x1, y1]).all():
        raise ValueError("Cannot draw non-finite corners")
    dx, dy, lower, upper = x1 - x0, y1 - y0, 0., 1.
    for p, q in ((-dx, x0), (dx, width - 1 - x0), (-dy, y0), (dy, height - 1 - y0)):
        if p == 0:
            if q < 0:
                return None
            continue
        ratio = q / p
        if p < 0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return None
    return ((x0 + lower * dx, y0 + lower * dy), (x0 + upper * dx, y0 + upper * dy))


def draw_cuboids(rgb, instances: list[dict], *, line_width: int = 6, debug_numbers: bool = False,
                 diagnostic_watermark: bool = False):
    from PIL import ImageDraw, ImageFont
    if type(line_width) is not int or line_width < 1:
        raise ValueError("Line width must be a positive native-pixel integer")
    result = rgb.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    for instance in instances:
        corners = np.asarray(instance["corners_px"], dtype=float)
        if corners.shape != (8, 2):
            raise ValueError("Eight fixed ordered corners required")
        for first, second in CUBOID_EDGES:
            clipped = clip_line(corners[first], corners[second], result.width, result.height)
            if clipped is not None:
                draw.line(clipped, fill=(255, 0, 0), width=line_width)
        if debug_numbers:
            for index, point in enumerate(corners):
                if 0 <= point[0] < result.width and 0 <= point[1] < result.height:
                    draw.text(tuple(point), str(index), fill=(255, 255, 0))
    if diagnostic_watermark:
        font = ImageFont.load_default(size=max(14, result.width // 80))
        text = "COMPONENT DIAGNOSTIC — NOT AN ACCEPTED MODEL RESULT"
        bounds = draw.textbbox((8, 8), text, font=font)
        draw.rectangle((0, 0, result.width, bounds[3] + 8), fill=(0, 0, 0))
        draw.text((8, 8), text, fill=(255, 255, 255), font=font)
    return result


def validate_video_qualification(checkpoint_state: dict, config: dict, *, allow_component: bool = False) -> tuple[bool, str]:
    """Qualification and output role are independent of visualization appearance."""
    component = (checkpoint_state.get("claim_scope") != "accepted_data_model_training"
                 or checkpoint_state.get("stage") not in ("B0", "C1")
                 or checkpoint_state.get("epoch_boundary") is not True
                 or type(checkpoint_state.get("global_step")) is not int
                 or checkpoint_state["global_step"] <= 0)
    if component and not allow_component:
        raise ValueError("Video needs qualified B0/C1 complete-epoch weights; diagnostic mode requires --allow-component")
    annotation_stage = config.get("video", {}).get("annotation_stage", "checks")
    if annotation_stage not in ("checks", "final_inference"):
        raise ValueError("Video annotation_stage must be checks or final_inference; it never supplies training labels")
    if annotation_stage == "final_inference" and component:
        raise ValueError("Component weights cannot write final_inference annotations")
    return component, annotation_stage


def run_video(root: str | Path, config: dict, checkpoint: str | Path, frame_manifest: str | Path,
              *, run_id: str, output_dir: str | Path | None = None, allow_component: bool = False,
              debug_numbers: bool = False) -> dict:
    import torch
    from PIL import Image
    root = checked_root(root)
    validate_run_id(run_id)
    checkpoint_path, manifest_path = input_path(root, checkpoint), input_path(root, frame_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source, frames = manifest["source"], manifest["frames"]
    count = source["frame_count"]
    if not manifest.get("complete") or [f["frame_id"] for f in frames] != list(range(count)):
        raise ValueError("Video requires the complete ordered native PNG frame manifest")
    if len(source["frames"]) != count or any(any(frame.get(key) != source["frames"][i].get(key)
                                                   for key in ("frame_id", "pts", "duration_pts"))
                                           for i, frame in enumerate(frames)):
        raise ValueError("Source frame timing differs from the cache")
    checkpoint_state = load_checkpoint(checkpoint_path)
    component, annotation_stage = validate_video_qualification(checkpoint_state, config, allow_component=allow_component)
    output = output_path(root, output_dir if output_dir is not None else Path("outputs/action4") / run_id)
    if not output.is_relative_to(output_path(root, "outputs")):
        raise ValueError("Video predictions and drawn frames must be stored under outputs, never data")
    if output.exists():
        raise FileExistsError("Video output must be a fresh directory; no mixed checkpoint resume")
    drawn = output_path(root, output / "drawn_frames")
    output.mkdir(parents=True)
    drawn.mkdir()
    device = select_device(config.get("runtime", {}).get("device", "auto"))
    model = build_model(config, checkpoint=checkpoint_path, device=device)
    model_hash = sha256_file(checkpoint_path)
    config_hash = canonical_hash(inference_contract(config))
    identity = {"schema_version": "action4.video_run.v1", "role": "development_video",
                "claim_scope": "component_diagnostic" if component else "model_raw_predictions_pending_visual_review",
                "source_sha256": source.get("source_sha256"), "frame_manifest_sha256": sha256_file(manifest_path),
                "checkpoint_sha256": model_hash, "inference_config_sha256": config_hash,
                "config": config, "code": source_identity(root), "environment": environment_identity(torch),
                "frame_count": count, "annotation_dir": str(drawn.relative_to(root)), "annotation_stage": annotation_stage, "smoothing": None, "association": None,
                "review_status": "unreviewed", "coordinate_clamping": "drawing clip only"}
    write_json_new(output / "run_manifest.json", identity)
    with (output / "predictions_raw.jsonl").open("x", encoding="utf-8") as prediction_file, \
            (output / "review_coverage.csv").open("x", encoding="utf-8", newline="") as review_file:
        writer = csv.DictWriter(review_file, fieldnames=["frame_id", "pts", "time_base", "status", "reviewer", "notes"])
        writer.writeheader()
        for frame in frames:
            image_path = input_path(root, frame["image"])
            image_hash = sha256_file(image_path)
            if image_hash != frame["image_sha256"]:
                raise ValueError(f"Cached PNG changed: frame {frame['frame_id']}")
            with Image.open(image_path) as checked:
                checked.verify()
            with Image.open(image_path) as original:
                original.load()
                rgb = original.convert("RGB")
                if rgb.size != (source["width"], source["height"]):
                    raise ValueError("Native frame size changed")
                tensor = torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).contiguous()
                instances = predict_images(model, [tensor], config, device)[0]
                record = {"frame_id": frame["frame_id"], "pts": frame["pts"], "duration_pts": frame["duration_pts"],
                          "time_base": source["time_base"], "source_image": frame["image"],
                          "source_image_sha256": image_hash, "checkpoint_sha256": model_hash,
                          "inference_config_sha256": config_hash, "instances": instances,
                          "claim_scope": identity["claim_scope"]}
                prediction_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                rendered = draw_cuboids(rgb, instances, line_width=config.get("video", {}).get("line_width", 6),
                                        debug_numbers=debug_numbers, diagnostic_watermark=component)
                rendered.save(drawn / f"frame_{frame['frame_id']:08d}.png")
            writer.writerow({"frame_id": frame["frame_id"], "pts": frame["pts"], "time_base": source["time_base"],
                             "status": "pending_review", "reviewer": "", "notes": ""})
    verification = DataAdapter(root).encode_video(manifest, drawn, output / "full_red_raw.mp4",
                                                  fps=source["observed_cfr_fps"],
                                                  crf=config.get("video", {}).get("crf", 16))
    result = {"status": "complete_io_pending_accuracy_review", "output_dir": str(output),
              "frame_count": count, "annotation_dir": str(drawn.relative_to(root)), "annotation_stage": annotation_stage, "claim_scope": identity["claim_scope"], "timeline_verification": verification}
    write_json_new(output / "completion.json", result)
    return result
