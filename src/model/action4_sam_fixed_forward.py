"""Explicit new-frame computation for registered PVS IDs; preserve native replay otherwise."""
import functools
import hashlib
import inspect
from pathlib import Path

def fixed_forward_parser(original):
    @functools.wraps(original)
    def parse(state):
        requested=state.get("_action4_fixed_forward_ids")
        if requested is None:return original(state)
        if (not isinstance(requested,tuple) or not 1<=len(requested)<=2
            or any(type(i) is not int or i<0 for i in requested) or len(set(requested))!=len(requested)):
            raise ValueError("Explicit one/two fixed instance IDs required")
        active=state.get("tracker_metadata",{}).get("obj_ids_all_gpu",[])
        if set(map(int,active))!=set(requested):
            raise ValueError("Fixed instance ownership differs from native state")
        kind,_=original(state)
        if kind=="propagation_full":
            raise ValueError("Fixed PVS continuation requires actual instance seeds; no concept fallback")
        if kind not in ("propagation_partial","propagation_fetch"):
            raise ValueError("Unexpected native propagation route")
        return "propagation_partial",list(requested)
    return parse

def install_fixed_forward(predictor,official_root):
    model=predictor.model
    if type(model).__name__!="Sam3MultiplexTrackingWithInteractivity":
        raise ValueError("Unexpected model for fixed PVS forward route")
    original=model.parse_action_history_for_propagation
    source=Path(inspect.getsourcefile(inspect.unwrap(original))).resolve(strict=True)
    expected=Path(official_root).resolve(strict=True)/"sam3/model/sam3_multiplex_tracking.py"
    if source!=expected or hashlib.sha256(source.read_bytes()).hexdigest()!="65e1cea27ee243dad228fb05967a8be92c90ef304665ada76930b899aa034d24":
        raise ValueError("Native propagation parser changed")
    if getattr(model,"_action4_fixed_forward_installed",False):raise ValueError("Parser bridge already installed")
    model.parse_action_history_for_propagation=fixed_forward_parser(original)
    model._action4_fixed_forward_installed=True
    return {"kind":"explicit_fixed_PVS_forward_all_registered_objects","default_replay_unchanged":True,
            "network_weights_changed":False,"requires_request_scoped_marker":True}
