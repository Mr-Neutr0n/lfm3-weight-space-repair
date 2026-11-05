"""Inspect published BF16 before/after column artifacts using the standard library."""
import hashlib
import json
from pathlib import Path
import struct


def audit(path, patch, receipt):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt['columns_file_sha256']:
        raise ValueError('Column artifact digest mismatch')
    size = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8 + size])
    payload = raw[8 + size:]
    expected = {f'edit_{i}.{side}' for i in range(len(patch['edits'])) for side in ['before', 'after']}
    if set(header) - {'__metadata__'} != expected:
        raise ValueError('Unexpected artifact tensors')
    changed = 0
    for i, edit in enumerate(patch['edits']):
        vectors = {}
        for side in ['before', 'after']:
            item = header[f'edit_{i}.{side}']
            if item['dtype'] != 'BF16' or item['shape'] != [2048, len(edit['columns'])]:
                raise ValueError('Unexpected BF16 column shape')
            a, b = item['data_offsets']
            block = payload[a:b]
            if len(block) != 2 * 2048 * len(edit['columns']):
                raise ValueError('Invalid column byte count')
            vectors[side] = list(struct.unpack('<' + 'H' * (len(block) // 2), block))
        changed += sum(a != b for a, b in zip(vectors['before'], vectors['after']))
        if edit['factor'] == 0 and any(v & 0x7fff for v in vectors['after']):
            raise ValueError('Suppressed columns contain a nonzero BF16 value')
    if changed != receipt['actual_changed_scalars']:
        raise ValueError('Actual changed scalar count mismatch')
    return {'actual_changed_scalars': changed, 'artifact_sha256': hashlib.sha256(raw).hexdigest()}
