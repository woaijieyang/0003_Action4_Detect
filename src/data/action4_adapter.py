"""Action 4 data IO and geometry, independent of training and evaluation.

NumPy is loaded only by geometry functions. FFmpeg/FFprobe are invoked only by
explicit video methods. Renderers and models are deliberately outside this module.
Pixel coordinates identify pixel centres; boxes accepted by ``crop_affine``
identify edges in that same coordinate system (the full image starts at -0.5).
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

from .action4_paths import checked_root, contained_path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "action4.v1"


def _np():
    try:
        import numpy
    except ImportError as exc:
        raise RuntimeError("Geometry requires the project's optional 'data' extra") from exc
    return numpy


def _json_write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(args: Sequence[str]) -> str:
    completed = subprocess.run(args, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"{args[0]} exited {completed.returncode}: {completed.stderr[-4000:]}")
    if completed.stderr.strip():
        raise RuntimeError(f"{args[0]} reported errors: {completed.stderr[-4000:]}")
    return completed.stdout


def _validated_png_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("PNG validation requires the project's optional 'data' extra") from exc
    try:
        with Image.open(path) as decoded:
            if decoded.format != "PNG":
                raise ValueError(f"Expected a PNG: {path}")
            size = decoded.size
            decoded.verify()
        with Image.open(path) as decoded:
            decoded.load()
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"PNG failed complete decoding: {path}: {exc}") from exc
    return size


class DataAdapter:
    """Read and write project-local assets only within ``root``.

    Existing runs are never overwritten. Resolving a symlink outside the root
    is rejected for writes, even when its lexical path appears to be inside.
    """

    def __init__(self, root: str | Path, *, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe"):
        self.root = checked_root(root)
        if not self.root.is_dir():
            raise ValueError("root must be an existing project directory")
        self.ffmpeg, self.ffprobe = ffmpeg, ffprobe

    def _source(self, path: str | Path) -> Path:
        value = contained_path(self.root, path, exists=True)
        if not value.is_file():
            raise ValueError(f"Source is not a regular file: {value}")
        return value

    def _destination(self, path: str | Path) -> Path:
        value = Path(path).expanduser()
        value = value if value.is_absolute() else self.root / value
        value = value.resolve(strict=False)
        if value == self.root or not value.is_relative_to(self.root):
            raise ValueError("Derived output must be inside the project root")
        if value.exists():
            raise FileExistsError(f"Use a new run/output path: {value}")
        return value

    def _new_dir(self, path: str | Path) -> Path:
        value = self._destination(path)
        value.parent.mkdir(parents=True, exist_ok=True)
        value.mkdir()
        return value

    def save_json(self, value: Any, path: str | Path) -> Path:
        target = self._destination(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        _json_write_new(target, value)
        return target

    def probe(self, source_video: str | Path, *, hash_source: bool = True,
              threads: int | None = None) -> dict[str, Any]:
        """Decode metadata for every video frame, retaining integer PTS + time base.

        ``frame_id`` is zero-based decoder presentation order, matching FFmpeg's
        ``select`` filter ``n``. A missing true PTS is not silently fabricated.
        ``best_effort_timestamp`` is recorded separately for diagnosis.
        Optional threads bounds both FFprobe invocations; None preserves legacy defaults.
        """
        if threads is not None and (type(threads) is not int or not 1 <= threads <= 8):
            raise ValueError("threads must be None or an integer from 1 to 8")
        thread_options = ["-threads", str(threads)] if threads is not None else []
        source = self._source(source_video)
        info = json.loads(_run([
            self.ffprobe, "-v", "error", *thread_options,
            "-show_streams", "-show_format", "-of", "json", str(source),
        ]))
        streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
        if not streams:
            raise ValueError("No video stream")
        stream = streams[0]
        raw_frames = json.loads(_run([
            self.ffprobe, "-v", "error", *thread_options, "-select_streams", "v:0", "-show_frames",
            "-show_entries", "frame=pts,best_effort_timestamp,duration,pkt_duration",
            "-of", "json", str(source),
        ])).get("frames", [])
        if not raw_frames:
            raise ValueError("No decoded frames")
        time_base = Fraction(stream["time_base"])
        frames = []
        for index, frame in enumerate(raw_frames):
            pts = int(frame["pts"]) if "pts" in frame else None
            duration = frame.get("duration", frame.get("pkt_duration"))
            frames.append({
                "frame_id": index, "pts": pts,
                "best_effort_timestamp": (int(frame["best_effort_timestamp"])
                                          if "best_effort_timestamp" in frame else None),
                "duration_pts": int(duration) if duration is not None else None,
                "timestamp_seconds": str(pts * time_base) if pts is not None else None,
            })
        pts_values = [frame["pts"] for frame in frames]
        complete_pts = all(value is not None for value in pts_values)
        differences = ([b - a for a, b in zip(pts_values, pts_values[1:])]
                       if complete_pts else [])
        increasing = bool(complete_pts and all(value > 0 for value in differences))
        constant_step = (differences[0] if increasing and differences
                         and len(set(differences)) == 1 else None)
        observed_fps = str(1 / (constant_step * time_base)) if constant_step else None
        return {
            "schema_version": SCHEMA_VERSION, "kind": "video_probe",
            "adapter_sha256": _sha256(Path(__file__)), "python_version": sys.version,
            "source_video": str(source), "source_bytes": source.stat().st_size,
            "source_sha256": _sha256(source) if hash_source else None,
            "video_stream_index": stream["index"], "codec": stream.get("codec_name"),
            "width": int(stream["width"]), "height": int(stream["height"]),
            "pixel_format": stream.get("pix_fmt"), "time_base": str(time_base),
            "avg_frame_rate": stream.get("avg_frame_rate"),
            "r_frame_rate": stream.get("r_frame_rate"), "frame_count": len(frames),
            "pts_complete": complete_pts, "pts_strictly_increasing": increasing,
            "constant_pts_step": constant_step, "observed_cfr_fps": observed_fps,
            "audio_stream_indices": [s["index"] for s in info.get("streams", [])
                                     if s.get("codec_type") == "audio"],
            "rotation_metadata": stream.get("side_data_list", []),
            "frames": frames,
        }

    def extract(
        self, source_video: str | Path, output_dir: str | Path,
        *, frame_ids: Iterable[int] | None = None, all_frames: bool = False,
        probe: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract chosen original coded pixels as PNG, without rotation/scaling.

        Exactly one of ``frame_ids`` and ``all_frames=True`` is required. Frame
        names retain source IDs; the manifest preserves original PTS. Full-cache
        extraction is an explicit, potentially large operation.
        """
        if (frame_ids is None) == (not all_frames):
            raise ValueError("Specify exactly one of frame_ids or all_frames=True")
        source = self._source(source_video)
        metadata = probe or self.probe(source)
        if Path(metadata["source_video"]).resolve() != source:
            raise ValueError("Probe belongs to another source")
        if (metadata["source_bytes"] != source.stat().st_size
                or metadata.get("source_sha256") and metadata["source_sha256"] != _sha256(source)):
            raise ValueError("Source changed after probing")
        count = metadata["frame_count"]
        selected = list(range(count)) if all_frames else list(frame_ids)
        if not selected or any(type(i) is not int or i < 0 or i >= count for i in selected):
            raise ValueError("frame_ids must be nonempty zero-based integers inside the source")
        if len(set(selected)) != len(selected):
            raise ValueError("Duplicate frame IDs")
        selected.sort()
        target = self._new_dir(output_dir)
        frames_dir = target / "frames"
        frames_dir.mkdir()
        selection = "+".join(f"eq(n\\,{i})" for i in selected)
        command = [self.ffmpeg, "-v", "error", "-nostdin", "-n", "-xerror", "-noautorotate", "-i", str(source),
                   "-map", "0:v:0"]
        if not all_frames:
            command.extend(["-vf", f"select={selection}"])
        command.extend(["-fps_mode", "passthrough", "-start_number", "0", str(frames_dir / "selected_%08d.png")])
        _run(command)
        actual = sorted(frames_dir.glob("selected_*.png"))
        if len(actual) != len(selected):
            raise RuntimeError(f"Extracted {len(actual)} frames, expected {len(selected)}; preserve failed run at {target}")
        records = []
        for rank, frame_id in enumerate(selected):
            path = frames_dir / f"frame_{frame_id:08d}.png"
            actual[rank].rename(path)
            if _validated_png_size(path) != (metadata["width"], metadata["height"]):
                raise RuntimeError(f"Unexpected extracted size: {path}")
            records.append({**metadata["frames"][frame_id], "image": str(path.relative_to(self.root)),
                            "image_sha256": _sha256(path)})
        manifest = {
            "schema_version": SCHEMA_VERSION, "kind": "video_frames",
            "adapter_sha256": _sha256(Path(__file__)),
            "source": metadata, "complete": selected == list(range(count)),
            "extraction": {"autorotate": False, "resize": None, "format": "png", "command": command},
            "frames": records,
        }
        _json_write_new(target / "frames_manifest.json", manifest)
        return manifest

    def profile(self, source_video: str | Path, output_dir: str | Path, *, samples: int = 16) -> dict[str, Any]:
        """Create a timing profile and deterministic chronological inspection frames.

        This is a mechanical sampling helper, not a semantic/occlusion analysis.
        A reviewer must inspect frames and record observations separately.
        """
        if type(samples) is not int or samples < 2:
            raise ValueError("samples must be an integer >= 2")
        metadata = self.probe(source_video)
        count = min(samples, metadata["frame_count"])
        selected = ([0] if count == 1 else
                    sorted({round(i * (metadata["frame_count"] - 1) / (count - 1)) for i in range(count)}))
        return self.extract(source_video, output_dir, frame_ids=selected, probe=metadata)

    def render_pilot(self, asset_dir: str | Path, output_dir: str | Path, *, blender: str | Path = "blender",
                     count: int = 32, seed: int = 20261018, threads: int = 2) -> dict[str, Any]:
        """Delegate a fresh bounded diagnostic pilot to the existing Blender backend.

        Blender runs as a separate process; this method never imports bpy. Asset
        preparation is a separate explicit operation. Successful schema/image
        validation does not accept the pilot's appearance or training suitability.
        All logs and failed outputs are retained in the newly created output run.
        """
        if (type(count) is not int or count != 32 or type(seed) is not int or seed < 0
                or type(threads) is not int or threads < 1):
            raise ValueError("The current backend recipe requires count=32; threads must be positive and seed nonnegative integers")
        executable = shutil.which(str(Path(blender).expanduser()))
        if executable is None:
            raise FileNotFoundError(f"Blender executable not found: {blender}")
        asset_path = contained_path(self.root, asset_dir, exists=True)
        if not asset_path.is_dir():
            raise ValueError("asset_dir must be an existing prepared asset directory")
        asset_manifest = self._source(asset_path / "asset_manifest.json")
        mesh = self._source(asset_path / "mesh_v001.npz")
        asset = json.loads(asset_manifest.read_text(encoding="utf-8"))
        if asset.get("mesh_data_sha256") and _sha256(mesh) != asset["mesh_data_sha256"]:
            raise ValueError("Prepared asset mesh disagrees with asset manifest hash")
        backend = Path(__file__).with_name("action4_render.py").resolve(strict=True)
        target = self._new_dir(output_dir)
        command = [executable, "--background", "--factory-startup", "--disable-autoexec", "-noaudio", "--threads", str(threads),
                   "--python-exit-code", "1", "--python", str(backend), "--", "pilot",
                   "--asset-dir", str(asset_path), "--output-dir", str(target), "--project-root", str(self.root),
                   "--count", str(count), "--seed", str(seed), "--start", "0", "--stop", str(count)]
        request = {"schema_version": SCHEMA_VERSION, "status": "launching", "scope": "diagnostic_pilot",
                   "adapter_sha256": _sha256(Path(__file__)), "backend_sha256": _sha256(backend),
                   "asset_manifest_sha256": _sha256(asset_manifest), "mesh_sha256": _sha256(mesh),
                   "count": count, "seed": seed, "threads": threads, "command": command}
        _json_write_new(target / "adapter_render_request.json", request)
        try:
            with (target / "blender_stdout.log").open("x", encoding="utf-8") as stdout, \
                    (target / "blender_stderr.log").open("x", encoding="utf-8") as stderr:
                completed = subprocess.run(command, check=False, stdout=stdout, stderr=stderr, text=True)
            if completed.returncode:
                raise RuntimeError(f"Blender exited {completed.returncode}; inspect logs in {target}")
            manifest_path = target / "manifest.jsonl"
            samples = self.read_samples(manifest_path)
            expected = {target.name + "_" + str(index).zfill(6) for index in range(count)}
            if len(samples) != count or {sample["sample_id"] for sample in samples} != expected:
                raise ValueError("Rendered sample IDs/count do not match the explicit pilot request")
            if any(not (self.root / sample["image"]).resolve().is_relative_to(target) for sample in samples):
                raise ValueError("Pilot manifest references images outside its new run")
            report = {"schema_version": SCHEMA_VERSION, "status": "data_contract_validated_visual_review_pending",
                      "scope": "diagnostic_pilot", "sample_count": len(samples), "seed": seed,
                      "manifest": str(manifest_path.relative_to(self.root)), "manifest_sha256": _sha256(manifest_path),
                      "adapter_sha256": request["adapter_sha256"], "backend_sha256": request["backend_sha256"],
                      "training_accepted": False}
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            _json_write_new(target / "adapter_render_result.json", {
                "schema_version": SCHEMA_VERSION, "status": "failed", "scope": "diagnostic_pilot",
                "error": str(exc), "adapter_sha256": request["adapter_sha256"],
                "backend_sha256": request["backend_sha256"], "training_accepted": False,
            })
            raise
        _json_write_new(target / "adapter_render_result.json", report)
        return report

    @staticmethod
    def fixed_corners(bounds_min: Sequence[float], bounds_max: Sequence[float]):
        """Fixed order: bottom 0..3, top 4..7; x changes first, then y."""
        np = _np()
        lo, hi = np.asarray(bounds_min, dtype=float), np.asarray(bounds_max, dtype=float)
        if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or not (hi > lo).all():
            raise ValueError("Bounds must be finite, ordered 3-vectors")
        return np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
                         [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                         [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                         [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])

    @staticmethod
    def project_corners(corners_obj, intrinsics, transform_cam_obj, *, min_depth: float = 1e-6):
        """Project 8 corners using rectified pinhole K and OpenCV +Z forward.

        Coordinates and transform translation must use the same physical units.
        No capture calibration, distortion or hidden-corner truth is inferred.
        """
        np = _np()
        corners, k, transform = (np.asarray(value, dtype=float) for value in
                                 (corners_obj, intrinsics, transform_cam_obj))
        if corners.shape != (8, 3) or k.shape != (3, 3) or transform.shape != (4, 4):
            raise ValueError("Expected corners (8,3), K (3,3), T_cam_obj (4,4)")
        if not all(np.isfinite(value).all() for value in (corners, k, transform)):
            raise ValueError("Non-finite geometry")
        if not np.allclose(transform[3], [0, 0, 0, 1]) or not np.allclose(k[2], [0, 0, 1]):
            raise ValueError("Invalid homogeneous transform or pinhole K")
        if k[0, 0] <= 0 or k[1, 1] <= 0 or min_depth <= 0:
            raise ValueError("Focal lengths and min_depth must be positive")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not math.isclose(float(np.linalg.det(rotation)), 1, abs_tol=1e-5):
            raise ValueError("T_cam_obj rotation must be a proper rigid rotation")
        camera = corners @ rotation.T + transform[:3, 3]
        if (camera[:, 2] <= min_depth).any():
            raise ValueError("A corner is behind/on the camera plane")
        projected = camera @ k.T
        return projected[:, :2] / projected[:, 2:3], camera[:, 2].copy()

    @staticmethod
    def letterbox_affine(source_size: Sequence[int], output_size: Sequence[int]):
        """Return exact centre-coordinate affine + integer resize/padding metadata."""
        np = _np()
        width, height = source_size
        out_w, out_h = output_size
        if any(type(v) is not int or v <= 0 for v in (width, height, out_w, out_h)):
            raise ValueError("Sizes must be positive integer (width,height) pairs")
        nominal = min(out_w / width, out_h / height)
        resize_w, resize_h = min(out_w, max(1, round(width * nominal))), min(out_h, max(1, round(height * nominal)))
        left, top = (out_w - resize_w) // 2, (out_h - resize_h) // 2
        sx, sy = resize_w / width, resize_h / height
        matrix = np.array([[sx, 0, left + (sx - 1) / 2], [0, sy, top + (sy - 1) / 2], [0, 0, 1]], dtype=float)
        return matrix, {"resize_size": [resize_w, resize_h],
                        "padding_ltrb": [left, top, out_w - resize_w - left, out_h - resize_h - top]}

    @staticmethod
    def crop_affine(box_edges_xyxy: Sequence[float], output_size: Sequence[int]):
        """Map a virtual crop to output pixel centres; out-of-image boxes are valid."""
        np = _np()
        left, top, right, bottom = map(float, box_edges_xyxy)
        width, height = output_size
        if not np.isfinite([left, top, right, bottom]).all() or right <= left or bottom <= top:
            raise ValueError("Invalid crop edges")
        if any(type(v) is not int or v <= 0 for v in (width, height)):
            raise ValueError("Output size must be positive integers")
        sx, sy = width / (right - left), height / (bottom - top)
        return np.array([[sx, 0, -left * sx - 0.5], [0, sy, -top * sy - 0.5], [0, 0, 1]])

    @staticmethod
    def transform_points(points, affine, *, inverse: bool = False):
        np = _np()
        xy, matrix = np.asarray(points, dtype=float), np.asarray(affine, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or matrix.shape != (3, 3):
            raise ValueError("Expected points (N,2) and affine (3,3)")
        if not np.isfinite(xy).all() or not np.isfinite(matrix).all() or not np.allclose(matrix[2], [0, 0, 1]):
            raise ValueError("Non-finite values or non-affine matrix")
        if abs(float(np.linalg.det(matrix))) < 1e-12:
            raise ValueError("Singular affine")
        matrix = np.linalg.inv(matrix) if inverse else matrix
        return (np.c_[xy, np.ones(len(xy))] @ matrix.T)[:, :2]

    def validate_sample(self, sample: dict[str, Any], *, check_image: bool = True) -> None:
        """Validate renderer/adapter sample semantics without requiring a framework."""
        np = _np()
        if sample.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported sample schema_version")
        for name in ("sample_id", "image", "group_id", "background_source_id", "scene_family"):
            if not isinstance(sample.get(name), str) or not sample[name]:
                raise ValueError(f"Missing nonempty string {name}")
        if any(type(sample.get(name)) is not int or sample[name] <= 0 for name in ("width", "height")):
            raise ValueError("Positive integer image dimensions required")
        if Path(sample["image"]).is_absolute():
            raise ValueError("Sample image must be relative to project root")
        image = (self.root / sample["image"]).resolve(strict=False)
        if not image.is_relative_to(self.root):
            raise ValueError("Sample image escapes project root")
        if check_image and (not image.is_file() or _validated_png_size(image) != (sample["width"], sample["height"])):
            raise ValueError("Sample PNG is missing or its dimensions disagree")
        instances = sample.get("instances")
        if not isinstance(instances, list) or len(instances) > 2:
            raise ValueError("instances must contain 0, 1, or 2 targets")
        seen = set()
        for instance in instances:
            identity = instance.get("instance_id")
            if not isinstance(identity, (int, str)) or isinstance(identity, bool) or identity in seen:
                raise ValueError("Unique instance_id required")
            seen.add(identity)
            if instance.get("object_id") != 1:
                raise ValueError("Action 4 object_id must be 1")
            corners = np.asarray(instance.get("corners_px"), dtype=float)
            bbox = np.asarray(instance.get("bbox_full_xyxy"), dtype=float)
            visible = instance.get("corner_visible")
            if corners.shape != (8, 2) or bbox.shape != (4,) or not np.isfinite(corners).all() or not np.isfinite(bbox).all():
                raise ValueError("Finite 8x2 corners and full xyxy box required")
            if not isinstance(visible, list) or len(visible) != 8 or any(type(v) is not bool for v in visible):
                raise ValueError("corner_visible must contain eight booleans")
            expected = np.r_[corners.min(axis=0), corners.max(axis=0)]
            if not np.allclose(bbox, expected, atol=1e-3, rtol=0):
                raise ValueError("Full box must enclose exactly the projected fixed eight corners")
            visible_box = instance.get("bbox_visible_xyxy")
            if visible_box is not None:
                visible_box = np.asarray(visible_box, dtype=float)
                if (visible_box.shape != (4,) or not np.isfinite(visible_box).all()
                        or visible_box[2] <= visible_box[0] or visible_box[3] <= visible_box[1]
                        or visible_box[0] < -0.5 or visible_box[1] < -0.5
                        or visible_box[2] > sample["width"] - 0.5
                        or visible_box[3] > sample["height"] - 0.5):
                    raise ValueError("Visible bbox must be an ordered in-image pixel-edge box")
            for name in ("occlusion_inframe", "truncation"):
                if (name == "occlusion_inframe" and instance.get(name) is None
                        and instance.get("occlusion_inframe_valid") is False):
                    continue
                if name in instance and (not isinstance(instance[name], (int, float))
                                         or not math.isfinite(instance[name]) or not 0 <= instance[name] <= 1):
                    raise ValueError(f"{name} must be in [0,1]; undefined occlusion requires null and occlusion_inframe_valid=false")
        if "split" in sample and sample["split"] not in ("train", "eval", "test"):
            raise ValueError("Unknown split")

    def read_samples(self, manifest: str | Path, *, check_images: bool = True) -> list[dict[str, Any]]:
        samples = []
        with self._source(manifest).open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                sample = json.loads(line)
                try:
                    self.validate_sample(sample, check_image=check_images)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Manifest line {line_no}: {exc}") from exc
                samples.append(sample)
        if not samples or len({s["sample_id"] for s in samples}) != len(samples):
            raise ValueError("Manifest must be nonempty with unique sample IDs")
        return samples

    @staticmethod
    def grouped_split(samples: Sequence[dict[str, Any]], *, seed: int = 20261018,
                      ratios: Sequence[float] = (0.8, 0.1, 0.1)) -> list[dict[str, Any]]:
        """Split connected groups, including transitive shared-background links.

        Input ordering never changes assignment. Ratios are targets, not exact
        guarantees when components have unequal sizes. Insufficient independent
        components raises an error instead of leaking the same source across splits.
        """
        if type(seed) is not int or len(ratios) != 3 or any(not math.isfinite(v) or v <= 0 for v in ratios) or not math.isclose(sum(ratios), 1, abs_tol=1e-9):
            raise ValueError("Use an integer seed and three positive ratios summing to 1")
        ids = [sample["sample_id"] for sample in samples]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Unique, nonempty sample IDs required")
        if any("split" in sample for sample in samples):
            raise ValueError("Refusing to repartition assigned data; preserve frozen eval/test splits")
        parent = {identity: identity for identity in ids}

        def find(identity):
            while parent[identity] != identity:
                parent[identity] = parent[parent[identity]]
                identity = parent[identity]
            return identity

        owners = {}
        for sample in sorted(samples, key=lambda item: item["sample_id"]):
            for key in ("group_id", "background_source_id"):
                value = sample.get(key)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"Nonempty {key} required for splitting")
                tag = key, value
                if tag in owners:
                    a, b = find(sample["sample_id"]), find(owners[tag])
                    parent[max(a, b)] = min(a, b)
                else:
                    owners[tag] = sample["sample_id"]
        components = {}
        for identity in ids:
            components.setdefault(find(identity), []).append(identity)
        if len(components) < 3:
            raise ValueError("Fewer than three independent components after group/background closure")
        groups = [sorted(group) for group in components.values()]
        groups.sort(key=lambda group: (-len(group), hashlib.sha256(f"{seed}:{group[0]}".encode()).hexdigest()))
        names = ("train", "eval", "test")
        counts = dict.fromkeys(names, 0)
        targets = dict(zip(names, [len(ids) * ratio for ratio in ratios]))
        assignment = {}
        for index, group in enumerate(groups):
            empty = [name for name in names if counts[name] == 0]
            choices = empty if len(groups) - index == len(empty) else names
            chosen = max(choices, key=lambda name: (targets[name] - counts[name], -names.index(name)))
            for identity in group:
                assignment[identity] = chosen
            counts[chosen] += len(group)
        result = [{**sample, "split": assignment[sample["sample_id"]]} for sample in sorted(samples, key=lambda sample: sample["sample_id"])]
        DataAdapter.validate_splits(result)
        return result

    @staticmethod
    def validate_splits(samples: Sequence[dict[str, Any]]) -> dict[str, int]:
        counts = dict.fromkeys(("train", "eval", "test"), 0)
        owner, identities = {}, set()
        for sample in samples:
            split = sample.get("split")
            if split not in counts or sample["sample_id"] in identities:
                raise ValueError("Missing/invalid split or duplicate sample ID")
            identities.add(sample["sample_id"])
            counts[split] += 1
            for key in ("group_id", "background_source_id"):
                tag = key, sample[key]
                if tag in owner and owner[tag] != split:
                    raise ValueError(f"Cross-split leakage through {key}: {sample[key]}")
                owner[tag] = split
        if not all(counts.values()):
            raise ValueError("Every split must contain at least one sample")
        return counts

    def write_split(self, samples: Sequence[dict[str, Any]], output_dir: str | Path, *, seed: int = 20261018,
                    ratios: Sequence[float] = (0.8, 0.1, 0.1)) -> dict[str, Any]:
        split_samples = self.grouped_split(samples, seed=seed, ratios=ratios)
        target = self._new_dir(output_dir)
        for name in ("train", "eval", "test"):
            with (target / f"{name}.jsonl").open("x", encoding="utf-8") as handle:
                for sample in split_samples:
                    if sample["split"] == name:
                        handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        report = {"schema_version": SCHEMA_VERSION, "seed": seed, "requested_ratios": list(ratios),
                  "adapter_sha256": _sha256(Path(__file__)),
                  "counts": self.validate_splits(split_samples),
                  "grouping_keys": ["group_id", "background_source_id"],
                  "manifest_sha256": {name: _sha256(target / f"{name}.jsonl") for name in ("train", "eval", "test")}}
        _json_write_new(target / "split_report.json", report)
        return report

    def encode_video(self, frame_manifest: str | Path | dict[str, Any], drawn_frame_dir: str | Path,
                     output_path: str | Path, *, fps: str | None = None, crf: int = 16,
                     threads: int = 4) -> dict[str, Any]:
        """Encode and re-decode complete native-size, zero-origin CFR PNG frames.

        Filenames must be frame_%08d.png with all source IDs exactly once.
        The manifest defines source timing, not drawn pixels: actual encoder input
        hashes are recorded separately. VFR/audio and implicit resizing are rejected.
        Outputs are new MP4 + verification JSON; failed generated files are retained.
        CLI restricts results to outputs/, while this API permits project-local paths.
        """
        if type(threads) is not int or not 1 <= threads <= 8:
            raise ValueError("threads must be an integer from 1 to 8")
        manifest_path = None
        if isinstance(frame_manifest, dict):
            manifest = json.loads(json.dumps(frame_manifest, allow_nan=False))
            manifest_sha = hashlib.sha256(json.dumps(manifest, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            manifest_hash_kind = "canonical_json"
        else:
            manifest_path = self._source(frame_manifest)
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
            manifest_hash_kind = "file_bytes"
        source = manifest["source"]
        count = source["frame_count"]
        if type(count) is not int or count <= 0:
            raise ValueError("frame_count must be a positive integer")
        expected_ids = list(range(count))
        records = manifest["frames"]
        if (manifest.get("complete") is not True or len(records) != count
                or any(type(f.get("frame_id")) is not int for f in records)
                or [f["frame_id"] for f in records] != expected_ids):
            raise ValueError("Encoding requires all source frame IDs in order, exactly once")
        if len(source["frames"]) != count or any(
            frame.get(key) != source["frames"][index].get(key)
            for index, frame in enumerate(records)
            for key in ("frame_id", "pts", "duration_pts")
        ):
            raise ValueError("Frame manifest timing disagrees with source probe")
        if not source["pts_complete"] or not source["pts_strictly_increasing"]:
            raise ValueError("VFR/missing PTS is not supported by this CFR encoder")
        if source["frames"][0]["pts"] != 0 or source.get("audio_stream_indices"):
            raise ValueError("This encoder currently requires zero-origin video without audio")
        try:
            time_base = Fraction(source["time_base"])
            requested_fps = Fraction(fps) if fps is not None else None
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            raise ValueError("FPS and time_base must be valid positive rational numbers") from exc
        step = source["constant_pts_step"]
        if time_base <= 0 or type(step) is not int or step <= 0:
            raise ValueError("Positive time_base and integer constant_pts_step required")
        if time_base.numerator != 1:
            raise ValueError("Exact MP4 time_base preservation requires a unit numerator")
        if any(type(f.get("pts")) is not int or type(f.get("duration_pts")) is not int
               or f["pts"] != i * step or f["duration_pts"] != step
               for i, f in enumerate(source["frames"])):
            raise ValueError("Every source PTS and duration must match the declared CFR sequence")
        expected_fps = 1 / (time_base * step)
        if requested_fps is not None and requested_fps != expected_fps:
            raise ValueError("Requested FPS differs from exact source PTS timing")
        if type(crf) is not int or not 0 <= crf <= 51:
            raise ValueError("crf must be an integer from 0 to 51")
        directory = contained_path(self.root, drawn_frame_dir, exists=True)
        if not directory.is_dir():
            raise ValueError("drawn_frame_dir must be a directory")
        expected_names = {f"frame_{i:08d}.png" for i in expected_ids}
        actual_names = {p.name for p in directory.iterdir() if p.suffix.lower() == ".png"}
        if actual_names != expected_names:
            raise ValueError(f"Frame set mismatch: {len(expected_names - actual_names)} missing, {len(actual_names - expected_names)} extra")
        width, height = source["width"], source["height"]
        if any(type(v) is not int or v <= 0 for v in (width, height)):
            raise ValueError("Native dimensions must be positive integers")
        if width % 2 or height % 2:
            raise ValueError("yuv420p encoding requires even native dimensions; resizing is not implicit")
        inputs = []
        resolved_inputs = set()
        for index in expected_ids:
            path = self._source(directory / f"frame_{index:08d}.png")
            if path in resolved_inputs:
                raise ValueError("Duplicate resolved PNG input")
            resolved_inputs.add(path)
            if _validated_png_size(path) != (width, height):
                raise ValueError(f"Frame size mismatch: {path.name}")
            inputs.append({"frame_id": index, "image": str(path.relative_to(self.root)),
                           "sha256": _sha256(path)})
        target = self._destination(output_path)
        if target.suffix.lower() != ".mp4":
            raise ValueError("Current verified encoder emits .mp4 only")
        report_path = self._destination(str(target) + ".verification.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [self.ffmpeg, "-v", "error", "-nostdin", "-n", "-xerror",
                   "-threads", str(threads), "-framerate", str(expected_fps),
                   "-start_number", "0", "-i", str(directory / "frame_%08d.png"), "-frames:v", str(count),
                   "-map", "0:v:0", "-an", "-c:v", "libx264", "-threads", str(threads),
                   "-filter_threads", "1", "-preset", "medium", "-crf", str(crf),
                   "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-video_track_timescale",
                   str(time_base.denominator), str(target)]
        _run(command)
        encoded = self.probe(target, threads=threads)
        encoded_base = Fraction(encoded["time_base"])
        expected_pts = [f["pts"] * time_base for f in source["frames"]]
        actual_pts = [f["pts"] * encoded_base if f["pts"] is not None else None for f in encoded["frames"]]
        expected_durations = [f["duration_pts"] * time_base for f in source["frames"]]
        actual_durations = [f["duration_pts"] * encoded_base if f["duration_pts"] is not None else None
                            for f in encoded["frames"]]
        changed_inputs = []
        for item in inputs:
            path = directory / f"frame_{item['frame_id']:08d}.png"
            try:
                current = self._source(path)
                unchanged = (str(current.relative_to(self.root)) == item["image"]
                             and _sha256(current) == item["sha256"])
            except (OSError, ValueError):
                unchanged = False
            if not unchanged:
                changed_inputs.append(item["frame_id"])
        manifest_unchanged = True
        if manifest_path is not None:
            try:
                current_manifest = self._source(manifest_path)
                manifest_unchanged = (current_manifest == manifest_path
                                      and _sha256(current_manifest) == manifest_sha)
            except (OSError, ValueError):
                manifest_unchanged = False
        frame_set_unchanged = {p.name for p in directory.iterdir()
                               if p.suffix.lower() == ".png"} == expected_names
        inputs_unchanged = not changed_inputs and manifest_unchanged and frame_set_unchanged
        fps_equal = encoded.get("observed_cfr_fps") == str(expected_fps) if count > 1 else (
            Fraction(encoded["avg_frame_rate"]) == expected_fps)
        verified = (encoded["frame_count"] == count and (encoded["width"], encoded["height"]) == (width, height)
                    and actual_pts == expected_pts and actual_durations == expected_durations
                    and encoded_base == time_base and fps_equal and not encoded["audio_stream_indices"]
                    and inputs_unchanged)
        report = {"schema_version": SCHEMA_VERSION, "kind": "video_encode_verification",
                  "verified": verified, "command": command,
                  "output": str(target), "verification_report": str(report_path),
                  "adapter_sha256": _sha256(Path(__file__)),
                  "frame_manifest": str(manifest_path) if manifest_path else None,
                  "frame_manifest_sha256": manifest_sha, "frame_manifest_hash_kind": manifest_hash_kind,
                  "source_sha256": source.get("source_sha256"), "encoded_sha256": encoded["source_sha256"],
                  "input_frames": inputs, "input_frames_unchanged": inputs_unchanged,
                  "changed_frame_ids": changed_inputs, "frame_manifest_unchanged": manifest_unchanged,
                  "frame_set_unchanged": frame_set_unchanged,
                  "frame_count": encoded["frame_count"], "width": encoded["width"], "height": encoded["height"],
                  "fps": str(expected_fps), "time_base": encoded["time_base"],
                  "all_pts_equal": actual_pts == expected_pts, "all_durations_equal": actual_durations == expected_durations,
                  "time_base_equal": encoded_base == time_base, "fps_equal": fps_equal,
                  "no_audio": not encoded["audio_stream_indices"]}
        _json_write_new(report_path, report)
        if not verified:
            raise RuntimeError(f"Encoded video failed timeline/size/input verification: {report_path}")
        return report
