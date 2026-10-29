"""Bounded, ordered RGB decode overlap; no model or target-selection changes."""
from concurrent.futures import ThreadPoolExecutor

def ordered_items(data,workers=4,ahead=8):
    if not 0<=workers<=8 or not 1<=ahead<=16:raise ValueError('Bounded evaluator CPU/memory budget required')
    if not workers:
        for i in range(len(data)):yield i,data[i]
        return
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='action4_eval_decode') as pool:
        pending={i:pool.submit(data.__getitem__,i) for i in range(min(ahead,len(data)))}
        for i in range(len(data)):
            item=pending.pop(i).result()
            next_index=i+ahead
            if next_index<len(data):pending[next_index]=pool.submit(data.__getitem__,next_index)
            yield i,item
