"""Fixed-domain route counters consumed by the existing training update loop.

Counts follow data consumption, not optimizer-step count. Checkpoint cursor is
authoritative; repeated append-only log entries share a logical update key.
"""
from data.action4_sam_roi_routing import summarize_routes,ROUTE_COUNT_FIELDS,ROUTE_COUNT_ROUTES,ROUTE_COUNT_FALLBACKS

SCHEMA="action4.training_route_counters.v1"


def empty_counts():
    return summarize_routes([])


def count_vector(counts):
    if not isinstance(counts,dict) or set(counts)!={"active_units","requested","effective","fallbacks"}:
        raise ValueError("Exact route count structure required")
    if (set(counts["requested"])!=set(ROUTE_COUNT_ROUTES) or
        set(counts["effective"])!=set(ROUTE_COUNT_ROUTES) or
        set(counts["fallbacks"])!=set(ROUTE_COUNT_FALLBACKS)):
        raise ValueError("Route count domain changed")
    values=[]
    for name in ROUTE_COUNT_FIELDS:
        parts=name.split(".")
        value=counts[parts[0]] if len(parts)==1 else counts[parts[0]][parts[1]]
        if type(value) is not int or value<0:raise ValueError("Nonnegative integer route counts required")
        values.append(value)
    if (sum(counts["requested"].values())!=counts["active_units"] or
        sum(counts["effective"].values())!=counts["active_units"] or
        sum(counts["fallbacks"].values())>counts["active_units"]):
        raise ValueError("Route count mass differs from active units")
    return values


def counts_from_vector(vector):
    if len(vector)!=len(ROUTE_COUNT_FIELDS):raise ValueError("Fixed11 route vector required")
    result=empty_counts()
    for name,value in zip(ROUTE_COUNT_FIELDS,vector):
        parts=name.split(".")
        if len(parts)==1:result[parts[0]]=value
        else:result[parts[0]][parts[1]]=value
    count_vector(result)
    return result


def local_route_counts(items):
    traces=[]
    for item in items:
        if type(item.get("active")) is not bool:raise ValueError("Explicit item active flag required")
        if not item["active"]:continue
        trace=item.get("metadata",{}).get("sam_roi_routing")
        if not isinstance(trace,dict) or trace.get("active") is not True:
            raise ValueError("Every active routed unit requires an active route trace")
        traces.append(trace)
    result=summarize_routes(traces)
    count_vector(result)
    return result


def reduce_route_update(items,*,enabled,device,all_reduce):
    """Exactly one11-int all_reduce when configured; zero work/collectives otherwise."""
    if type(enabled) is not bool:raise ValueError("Configured routing flag must be boolean")
    if not enabled:return None
    import torch
    values=torch.tensor(count_vector(local_route_counts(items)),device=device,dtype=torch.int64)
    all_reduce(values)
    return counts_from_vector(values.cpu().tolist())


def _position(epoch,cursor,units_per_epoch):
    if any(type(v) is not int for v in (epoch,cursor,units_per_epoch)) or epoch<0 or units_per_epoch<1 or not 0<=cursor<=units_per_epoch:
        raise ValueError("Explicit data epoch/cursor/domain required")
    return epoch*units_per_epoch+cursor


def advance_counts(cumulative,update,*,epoch,cursor,units_per_epoch):
    expected=_position(epoch,cursor,units_per_epoch)
    left=count_vector(cumulative);right=count_vector(update)
    if cumulative["active_units"]!=expected:raise ValueError("Route counters differ from pre-update data cursor")
    if update["active_units"]>units_per_epoch-cursor:raise ValueError("Route update crosses epoch boundary")
    return counts_from_vector([a+b for a,b in zip(left,right)])


def checkpoint_route_counts(counts,*,epoch,cursor,units_per_epoch):
    expected=_position(epoch,cursor,units_per_epoch)
    copied=counts_from_vector(count_vector(counts))
    if copied["active_units"]!=expected:raise ValueError("Checkpoint route counts differ from committed data cursor")
    return {"schema":SCHEMA,"field_order":list(ROUTE_COUNT_FIELDS),"epoch_zero_based":epoch,
            "cursor":cursor,"units_per_epoch":units_per_epoch,"counts":copied}


def restore_route_counts(state,*,epoch,cursor,units_per_epoch):
    if not isinstance(state,dict) or set(state)!={"schema","field_order","epoch_zero_based","cursor","units_per_epoch","counts"}:
        raise ValueError("Route-enabled recovery requires complete route counter state")
    if (state["schema"]!=SCHEMA or state["field_order"]!=list(ROUTE_COUNT_FIELDS) or
        (state["epoch_zero_based"],state["cursor"],state["units_per_epoch"])!=(epoch,cursor,units_per_epoch)):
        raise ValueError("Route counter recovery source/cursor/domain mismatch")
    return checkpoint_route_counts(state["counts"],epoch=epoch,cursor=cursor,units_per_epoch=units_per_epoch)["counts"]


def route_log_fields(update,cumulative,*,epoch,cursor_after,units_per_epoch):
    """The cumulative checkpoint is authoritative; do not sum replayed log lines."""
    checkpoint_route_counts(cumulative,epoch=epoch,cursor=cursor_after,units_per_epoch=units_per_epoch)
    before=cursor_after-update["active_units"]
    if before<0:raise ValueError("Invalid route log cursor")
    return {"roi_route_counts_update":counts_from_vector(count_vector(update)),
            "roi_route_counts_cumulative":counts_from_vector(count_vector(cumulative)),
            "roi_route_progress":{"epoch_zero_based":epoch,"cursor_before":before,"cursor_after":cursor_after,
                "logical_update_key":f"{epoch}:{before}:{cursor_after}",
                "semantics":"cumulative_from_checkpoint_cursor; deduplicate replayed logical_update_key, never sum cumulative lines"}}
