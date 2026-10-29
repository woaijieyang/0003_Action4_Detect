"""Exact cross-NumPy replay of native box diagnostics; never tolerance matching."""
import copy
import numpy as np
from eval.action4_sam_image_eval import _box_iou


def replay_native_diagnostics(native, boxes_xywh01, *, width, height, target_box,
                              producer_load, source_identity):
    """Reproduce bound producer float32 scalar promotion, retaining all fields.

    Unknown/unproven runtime or dtype receives no fallback. Callers must still
    compare the returned complete rows, object identity and binding reason
    EXACTLY against the saved receipt. This function never edits raw arrays,
    saved receipts, mask support or any binding eligibility decision.
    """
    version=producer_load.get("numpy")
    producer_major=version.split(".",1)[0] if isinstance(version,str) else None
    consumer_major=np.__version__.split(".",1)[0]
    if (producer_major not in ("1","2") or consumer_major not in ("1","2")
            or producer_major==consumer_major or not isinstance(boxes_xywh01,np.ndarray)
            or boxes_xywh01.dtype!=np.float32
            or producer_load.get("source_snapshot_identity")!=source_identity):
        return native
    if boxes_xywh01.shape!=(len(native),4):
        raise ValueError("Native diagnostic rows/boxes differ")
    result=copy.deepcopy(native);target=np.asarray(target_box,float)
    for i,row in enumerate(result):
        x,y,bw,bh=boxes_xywh01[i]
        # Addition remains float32 in both versions. Only scalar multiplication
        # with the Python dimension differs under NumPy1 versus NEP50 NumPy2.
        values=(x,y,np.float32(x+bw),np.float32(y+bh))
        scales=(width,height,width,height)
        box=np.asarray([float(value)*scale if producer_major=="1"
            else float(np.float32(value)*np.float32(scale))
            for value,scale in zip(values,scales)],dtype=float)
        row["bbox_native_xyxy"]=box.tolist()
        row["bbox_iou_prompt"]=_box_iou(box,target)
    return result
