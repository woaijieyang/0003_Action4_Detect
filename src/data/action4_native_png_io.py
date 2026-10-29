"""Read source-authenticated native PNG pixels for the isolated CPU tracking runtime."""
from pathlib import Path
import hashlib,io
import numpy as np


def read_native_bgr(root,ref,*,shape_hw):
    from PIL import Image
    root=Path(root).resolve(strict=True)
    path=(root/ref["path"]).resolve(strict=True)
    if not path.is_relative_to(root):raise ValueError("Native PNG outside project")
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ref["sha256"]:raise ValueError("Native PNG source changed")
    if len(shape_hw)!=2 or any(type(n) is not int or n<=0 for n in shape_hw):raise ValueError("Explicit positive image shape required")
    with Image.open(io.BytesIO(raw)) as im:
        if im.format!="PNG" or im.size!=(shape_hw[1],shape_hw[0]):
            raise ValueError("Expected same-size native PNG")
        rgb=np.asarray(im.convert("RGB"))
    return np.ascontiguousarray(rgb[:,:,::-1])
