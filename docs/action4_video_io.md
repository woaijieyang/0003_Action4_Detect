# Action4 PNG-to-video interface

This guide documents the existing DataAdapter.encode_video API and action4_data.py encode CLI.
It supplements the approved technical plan; it does not introduce a second encoder or a different training plan.

## Stored inputs and results

- Raw frames: data/action4_real/cache_v001/frames/frame_00000000.png through frame_00002849.png.
- Raw timing/provenance: data/action4_real/cache_v001/frames_manifest.json.
- Reference demo frames: data/action4_reference/demo_standard_cache_v001; keep them separate from target data.
- Model-drawn frames and all MP4/check reports belong under outputs/. Never overwrite raw PNGs or draw over data/ in place.
- Data contains raw/annotated datasets and assets. An overlay made for inspection is a run result, not a training annotation.
- The two existing full extraction reports are outputs/action4_extract_raw_v001/verification.json and outputs/action4_extract_demo_v001/verification.json. They are independent evidence from encoder interface tests.

## CLI

From the authorized server project directory:

~~~bash
cd /data/zy/vln/a-test &&
test "$(pwd -P)" = /data/zy/vln/a-test &&
source runtime/bootstrap/project_env.sh &&
PYTHONPATH=src python scripts/action4_data.py encode \
  --manifest data/action4_real/cache_v001/frames_manifest.json \
  --frames-dir outputs/action4/VIDEO_RUN/drawn_frames \
  --output outputs/action4/VIDEO_RUN/full_red_reencoded.mp4 \
  --fps 2997/100 --crf 16 --threads 4
~~~

Replace VIDEO_RUN with the real existing run directory. This command is an interface example, not a claim that a qualified full video already exists.
The input directory must already contain exactly all 2,850 PNGs. The MP4 and its .verification.json must not already exist.
For a raw reconstruction check, --frames-dir may instead point to the raw frames directory; choose a new outputs/ check directory and identify the result as raw reconstruction, not model prediction.

Successful CLI stdout is one JSON report, including absolute output and verification_report paths. Input errors exit 2.
--root, when used, is a global argument before encode. Both the selected root's outputs/ subtree and the actual repository outputs/ subtree constrain CLI result paths, including symlinks.
--threads is an integer 1–8 (default 4) applied to FFmpeg input/output codecs and both post-encode FFprobe invocations, including full-frame decoding; no GPU is used. --crf is 0–51 (default 16). FPS is optional as an exact assertion; timing is always derived from source PTS. Use 2997/100, not 30 or 30000/1001.

## Python API

~~~python
from data.action4_adapter import DataAdapter

report = DataAdapter("/data/zy/vln/a-test").encode_video(
    "data/action4_real/cache_v001/frames_manifest.json",
    "outputs/action4/VIDEO_RUN/drawn_frames",
    "outputs/action4/VIDEO_RUN/full_red_reencoded.mp4",
    fps="2997/100",
    crf=16,
    threads=4,
)
assert report["verified"]
~~~

frame_manifest accepts a project-contained JSON path or a complete in-memory manifest dictionary.
For compatibility with existing project-local callers/test fixtures, the generic API permits new outputs anywhere within its checked root; production callers must use outputs/. The CLI additionally enforces the actual outputs/ rule.
Input/output/manifest paths must remain inside root after symlink resolution. The original source-video path in metadata is provenance, not an instruction to access an external file.
The API does not generate corners or GT, repair model predictions, select checkpoints, or certify object localization. Formal B0/C1 checkpoint qualification remains the responsibility of the existing model-video entrypoint.

## Enforced sequence contract

Before running FFmpeg:

1. A complete manifest contains every integer source frame ID from zero in order exactly once; no partial manifest promoted to complete.
2. Every record's PTS and duration agrees with source metadata; actual source PTS must equal frame_id × constant step, including the last duration.
3. Only zero-origin CFR without audio is supported. Time base must be positive with unit numerator so MP4 preserves it exactly.
4. PNG names must be exactly frame_%08d.png for the declared IDs. Missing/extra PNGs (including alternate-case .PNG), duplicate resolved paths, corrupt PNGs and non-PNG content fail.
5. Every PNG retains the declared native width/height. H.264 yuv420p requires positive even dimensions. The interface never resizes or crops to make an input pass.
6. Existing MP4 or verification sidecar is rejected; output remains within the checked root.

For the actual source: 3248×2464, 2,850 frames, FPS 2997/100, time base 1/11988, PTS 0,400,...,1139600, duration 400 for every frame.
The reference demo is a separate 950×720 source, not resized target training data.

After encoding, FFprobe decodes every output frame and verifies frame count, native dimensions, exact time base, rational FPS, every PTS/duration, and absence of audio.
Actual PNG hashes and frame set are checked again; input changes invalidate the result. Treat inputs as immutable during encoding. Failed generated files are retained for diagnosis and must not be reused as successful outputs.

## Evidence meaning

The report distinguishes:

- source_sha256: the original video provenance recorded in the supplied source manifest; encoding does not re-hash or revalidate that original video.
- frame_manifest_sha256: file bytes SHA-256 for a file manifest, or explicitly labelled canonical JSON SHA-256 for an in-memory manifest.
- input_frames: SHA-256 and frame ID of each actual PNG sent to encoding. Drawn-frame hashes are expected to differ from raw source PNG hashes.
- encoded_sha256: the produced MP4 bytes.
- input_frames_unchanged / changed_frame_ids / frame_manifest_unchanged / frame_set_unchanged: consistency during the encoding operation.
- all_pts_equal / all_durations_equal / time_base_equal / fps_equal / no_audio / verified: mechanical encoding contract.

H.264 CRF encoding is lossy. Pixel equality between drawn PNGs and decoded video is not asserted.
A verified report proves timeline/size/input-consistency checks, not correct eight-corner predictions. Video accuracy still requires the plan's visual/geometry review.

## Reproducible CPU checks

~~~bash
cd /data/zy/vln/a-test &&
test "$(pwd -P)" = /data/zy/vln/a-test &&
source runtime/bootstrap/project_env.sh &&
export PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m unittest discover -s tests -p test_action4_video_io.py -v
python -m unittest discover -s tests -p test_action4_adapter.py -v
python -m unittest discover -s tests -p test_action4_training_eval.py -v
~~~

The new suite creates a complete seven-frame synthetic 64×48 source with the same rational timing as the real source, draws distinct red inspection rectangles, encodes, checks all seven timestamps/durations and source/raw/drawn hashes, and tests rejection cases. These rectangles are interface fixtures, not Action4 annotations.
Evidence is retained under outputs/action4_video_io_validation_v001/. Existing adapter/training-evaluation regression fixtures remain in their established project-local test directories.
No full 2,850-frame predicted video is claimed by these tests.

## VIO-Q01 resource-review correction

The independent quality review found that output-verification FFprobe previously did not receive the encoder thread budget. The encoder now forwards its explicit thread count to probe(). Existing direct probe callers remain compatible: optional threads=None retains previous defaults; an explicit count must be an integer 1–8. Full-frame decoding and every-PTS verification remain enabled. The real seven-frame command-capture regression executes FFmpeg/FFprobe and checks both verification commands for -threads 2. The original failed review and reproduced failing test are retained.
