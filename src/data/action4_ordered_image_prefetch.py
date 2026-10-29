"""Bounded ordered PNG decoding for read-only video rendering; identical pixels."""
import hashlib,io
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from data.action4_paths import PROJECT_ROOT,contained_path


def iter_rgb_previews(frames, *, image_size, preview_size=(950,720), workers=0, verify_sha=False):
    if type(workers) is not int or not 0<=workers<=16:
        raise ValueError("Explicit bounded0..16 PNG workers required")
    def decode(frame):
        path=contained_path(PROJECT_ROOT,frame['image'],exists=True)
        source=path
        if verify_sha:
            payload=path.read_bytes()
            if hashlib.sha256(payload).hexdigest()!=frame['image_sha256']:raise ValueError("Source RGB bytes differ from frame manifest")
            source=io.BytesIO(payload)
        with Image.open(source) as image:
            if image.size!=tuple(image_size):raise ValueError("Native source geometry changed")
            return image.convert('RGB').resize(tuple(preview_size))
    if workers==0:
        for frame in frames:yield frame,decode(frame)
        return
    source=iter(frames);pending=deque()
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='action4-png') as pool:
        def enqueue():
            try:frame=next(source)
            except StopIteration:return False
            pending.append((frame,pool.submit(decode,frame)));return True
        for _ in range(2*workers):
            if not enqueue():break
        try:
            while pending:
                frame,future=pending.popleft();image=future.result()
                enqueue()
                yield frame,image
        finally:
            for _,future in pending:future.cancel()
