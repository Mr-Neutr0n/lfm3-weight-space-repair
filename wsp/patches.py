"""Validated, replayable sparse column edits to MLP output weights."""
import hashlib
import math
import re


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def validate_patch(patch, protocol, shapes):
    if patch.get("model_revision") != protocol["model_revision"]:
        raise ValueError("Patch model revision mismatch")
    total = 0
    seen = set()
    for item in patch["edits"]:
        name, columns, factor = item["tensor"], item["columns"], item["factor"]
        if not re.fullmatch(r"model\.layers\.\d+\.feed_forward\.w2\.weight", name) or name not in shapes:
            raise ValueError("Patch must target an existing MLP output weight")
        if type(factor) not in [int, float] or not math.isfinite(factor) or abs(factor) > 10:
            raise ValueError("Invalid patch factor")
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("Empty or duplicate columns")
        for column in columns:
            if type(column) is not int or column < 0 or column >= shapes[name][1]:
                raise ValueError("Column out of range")
            if (name, column) in seen:
                raise ValueError("Overlapping edits")
            seen.add((name, column))
        if not re.fullmatch(r"[a-f0-9]{64}", item.get("before_sha256", "")):
            raise ValueError("Missing original tensor hash")
        total += len(columns) * shapes[name][0]
    if not total or total > protocol["edit_scope"]["maximum_changed_scalars"]:
        raise ValueError("Patch exceeds scalar budget")
    return total


def apply_patch(model, patch, protocol):
    import torch
    parameters = dict(model.named_parameters())
    validate_patch(patch, protocol, {k: tuple(v.shape) for k, v in parameters.items()})
    names = {item["tensor"] for item in patch["edits"]}
    # Verify every binding before the first mutation.
    for item in patch["edits"]:
        if tensor_hash(parameters[item["tensor"]]) != item["before_sha256"]:
            raise ValueError("Original tensor hash mismatch")
    backups = {name: parameters[name].detach().clone() for name in names}
    try:
        with torch.no_grad():
            for item in patch["edits"]:
                target = parameters[item["tensor"]]
                target[:, item["columns"]] = (target[:, item["columns"]].float() * item["factor"]).to(target.dtype)
            if any(not torch.isfinite(parameters[name]).all() for name in names):
                raise ValueError("Patch produced nonfinite weights")
    except BaseException:
        restore(model, backups)
        raise
    changed = sum(int((parameters[name] != original).sum()) for name, original in backups.items())
    return backups, {"actual_changed_scalars": changed,
                     "after_sha256": {name: tensor_hash(parameters[name]) for name in names}}


def restore(model, backups):
    import torch
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in backups.items():
            parameters[name].copy_(tensor)
