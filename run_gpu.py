"""Optional fresh inference on public or caller-supplied cases. No training."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from wsp.missions import messages, tools
from wsp.edit_metrics import assess

ROOT = Path(__file__).resolve().parent
ARMS = {'stock-direct': None, 'sol-direct': 'patches/sol/selected.json',
        'astra-direct': 'patches/astra/selected.json',
        'astra-random-direct': 'patches/astra/random-control.json', 'stock-thinking': None}


def run(args):
    if not args.confirm_inference:
        raise ValueError('Pass --confirm-inference to download/load the model and use a GPU')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output directory; existing runs are never overwritten')
    cases = [json.loads(s) for s in Path(args.cases).read_text().splitlines()]
    if not cases or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Cases must be nonempty and unique')
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from wsp.patches import apply_patch, restore
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A BF16-capable NVIDIA GPU is required')
    snapshot = Path(snapshot_download(protocol['model_id'], revision=protocol['model_revision'],
                                    allow_patterns=list(json.loads((ROOT / 'provenance/model-files.json').read_text())['files'])))
    manifest = json.loads((ROOT / 'provenance/model-files.json').read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Pinned model file mismatch: ' + name)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
    torch.manual_seed(protocol['seed'])
    model = AutoModelForCausalLM.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
    if sum(p.numel() for p in model.parameters()) != protocol['parameters']:
        raise ValueError('Incorrect target checkpoint')
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    selected_arms = list(ARMS) if args.arm == 'all' else [args.arm]
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'checkpoint': protocol['model_id'], 'revision': protocol['model_revision'],
        'cases_sha256': hashlib.sha256(Path(args.cases).read_bytes()).hexdigest(), 'arms': selected_arms,
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'note': 'Fresh replication. The original private final sets are not included.'}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for arm in selected_arms:
        backups = None
        if ARMS[arm]:
            patch = json.loads((ROOT / ARMS[arm]).read_text())
            backups, receipt = apply_patch(model, patch, protocol)
            (output / f'{arm}-patch.json').write_text(json.dumps(receipt, indent=2) + '\n')
        mode = 'thinking' if arm == 'stock-thinking' else 'direct'
        try:
            with (output / f'{arm}.jsonl').open('x') as out:
                for case in cases:
                    prompt = tokenizer.apply_chat_template(messages(case), tools=tools(), tokenize=False,
                        add_generation_prompt=True, enable_thinking=mode == 'thinking')
                    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
                    prompt_ids = inputs['input_ids'][0].tolist()
                    if len(prompt_ids) > protocol['max_input_tokens']:
                        raise ValueError('Input exceeds the frozen token limit')
                    started = time.monotonic()
                    with torch.inference_mode():
                        answer = model.generate(**inputs, do_sample=False, num_beams=1, use_cache=True,
                            max_new_tokens=protocol['max_new_tokens'], repetition_penalty=protocol['repetition_penalty'],
                            pad_token_id=tokenizer.pad_token_id, eos_token_id=eos)
                    torch.cuda.synchronize()
                    ids = answer[0, len(prompt_ids):].tolist()
                    row = {'case_id': case['case_id'], 'arm': arm, 'mode': mode, 'prompt': prompt,
                        'prompt_ids': prompt_ids, 'generated_ids': ids, 'raw_text': tokenizer.decode(ids, skip_special_tokens=False),
                        'truncated': not ids or ids[-1] not in eos_ids, 'seconds': time.monotonic() - started}
                    result = assess(case, row)
                    row['success'], row['category'] = result['success'], result['category']
                    out.write(json.dumps(row) + '\n'); out.flush()
        finally:
            if backups is not None:
                restore(model, backups)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cases', default=str(ROOT / 'data/development.jsonl'))
    p.add_argument('--output', required=True)
    p.add_argument('--arm', choices=['all', *ARMS], default='all')
    p.add_argument('--confirm-inference', action='store_true')
    run(p.parse_args())
