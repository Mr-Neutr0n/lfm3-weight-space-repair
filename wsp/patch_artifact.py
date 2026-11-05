"""Apply or undo the saved BF16 columns without recomputing a scaling edit."""
from pathlib import Path
from .missions import digest
from .patches import tensor_hash, validate_patch


def replay_columns(model, patch, receipt, path, protocol, reverse=False):
    import torch
    from safetensors.torch import load_file
    path = Path(path)
    if digest(path) != receipt["columns_file_sha256"]:
        raise ValueError("Saved column artifact changed")
    parameters = dict(model.named_parameters())
    validate_patch(patch, protocol, {k: tuple(v.shape) for k, v in parameters.items()})
    columns = load_file(str(path))
    if set(columns) != {f"edit_{i}.{side}" for i in range(len(patch["edits"])) for side in ["before", "after"]}:
        raise ValueError("Unexpected saved columns")
    names = {item["tensor"] for item in patch["edits"]}
    for item in patch["edits"]:
        expected = receipt["after_sha256"][item["tensor"]] if reverse else item["before_sha256"]
        if tensor_hash(parameters[item["tensor"]]) != expected:
            raise ValueError("Current weights do not match the artifact")
    backups = {name: parameters[name].detach().clone() for name in names}
    try:
        with torch.no_grad():
            for i, item in enumerate(patch["edits"]):
                target = parameters[item["tensor"]]
                saved = columns[f"edit_{i}.{'before' if reverse else 'after'}"]
                if saved.dtype != target.dtype or tuple(saved.shape) != (target.shape[0], len(item["columns"])):
                    raise ValueError("Saved column shape or dtype mismatch")
                target[:, item["columns"]] = saved.to(target.device)
            for item in patch["edits"]:
                expected = item["before_sha256"] if reverse else receipt["after_sha256"][item["tensor"]]
                if tensor_hash(parameters[item["tensor"]]) != expected:
                    raise ValueError("Replayed tensor hash mismatch")
    except BaseException:
        with torch.no_grad():
            for name, value in backups.items():
                parameters[name].copy_(value)
        raise
    return {"direction": "revert" if reverse else "apply", "verified_tensors": sorted(names)}
