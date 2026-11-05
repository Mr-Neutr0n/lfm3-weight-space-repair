"""Budgeted sparse editing probes and a separate prompt-only evaluation worker."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import time

from .edit_metrics import assess, summary
from .missions import ROOT, digest, messages, oracle, tools
from .patches import apply_patch, restore, tensor_hash
from .runner import atomic, source_binding, verify_model


PROBE_STAGES = {"discovery", "candidate-a", "candidate-b", "control"}


def verify_ops_binding(args, plan):
    manifest_path = Path(os.environ["WSP_PHASE_MANIFEST"])
    ledger_path = Path(os.environ["WSP_BUDGET_LEDGER"])
    manifest = json.loads(manifest_path.read_text())
    # sbatch can start the worker before returning its job ID to the submitter.
    # The submitter holds the exclusive budget lock through its final ledger write.
    with ledger_path.with_name(".budget.lock").open("r") as budget_lock:
        fcntl.flock(budget_lock, fcntl.LOCK_SH)
        ledger = json.loads(ledger_path.read_text())
    content = {k: v for k, v in manifest.items() if k != "binding_sha256"}
    binding = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (binding != manifest["binding_sha256"] or binding != os.environ["WSP_PHASE_BINDING_SHA256"]
            or manifest["stage"] != args.stage or Path(manifest["source"]).resolve() != ROOT
            or manifest["plan_sha256"] != digest(Path(args.plan))
            or manifest["request_sha256"] != digest(Path(args.request))):
        raise ValueError("Ops source/config binding mismatch")
    for name, expected in manifest["files"].items():
        path = ROOT / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT) or digest(path) != expected:
            raise ValueError("Staged source changed after submission")
    entry = ledger["stages"][args.stage]
    if (entry["binding_sha256"] != binding or not os.environ.get("SLURM_JOB_ID")
            or str(entry.get("job_id")) != os.environ["SLURM_JOB_ID"]
            or sum(e["adaptive_episodes"] for e in ledger["stages"].values()) > 32):
        raise ValueError("Missing Slurm ownership or probe reservation")
    if args.stage in {"discovery", "candidate-a", "candidate-b"} and ledger.get("freeze_sha256"):
        raise ValueError("Adaptive selection is already closed")
    return {"binding_sha256": binding, "manifest_sha256": digest(manifest_path),
            "budget_at_start": ledger, "slurm_job_id": os.environ["SLURM_JOB_ID"]}


def read_bound(path, expected=None):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Configuration cannot be a symlink")
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Configuration must be inside the staged source")
    if expected and digest(path) != expected:
        raise ValueError("Configuration hash mismatch")
    return json.loads(path.read_text())


def native(action):
    # Canonical case JSON sorts mapping keys. Native calls follow tool-schema order.
    schema = next(t["function"]["parameters"]["properties"] for t in tools()
                  if t["function"]["name"] == action["name"])
    args = "\n".join(f'<parameter={k}>\n{action["arguments"][k]}\n</parameter>' for k in schema)
    return f'<tool_call>\n<function={action["name"]}>\n{args}\n</function>\n</tool_call>'


def validate_request(stage, request, plan, protocol):
    if stage not in PROBE_STAGES | {"evaluation"} or request["stage"] != stage:
        raise ValueError("Invalid stage identity")
    if plan["protocol_sha256"] != digest(ROOT / "protocol.json"):
        raise ValueError("Frozen protocol changed")
    if plan["adaptive_budget"]["maximum_episodes"] != protocol["edit_scope"]["maximum_probe_episodes"]:
        raise ValueError("Probe budget changed")
    if stage in PROBE_STAGES:
        repair = stage == "candidate-a" and request.get("role") == "localization-repair"
        if repair:
            amendment = read_bound(ROOT / request["amendment"], request["amendment_sha256"])
            if (amendment["id"] != "localization-repair-v1" or amendment["base_plan_sha256"] != digest(ROOT / "editing/plan.json")
                    or amendment["protocol_changed"] or amendment["budgets_changed"]
                    or amendment["maximum_probe_episodes"] != 32 or amendment["maximum_changed_scalars"] != 16384):
                raise ValueError("Invalid localization repair amendment")
        if request["case_ids"] != plan["probe_case_ids"] or request["mode"] != "direct":
            raise ValueError("Probe request changed the frozen missions or mode")
        if len(request["case_ids"]) != plan["adaptive_budget"]["episodes_per_stage"][stage]:
            raise ValueError("Probe stage exceeds its episode reservation")
        if request.get("capture_activations", False) != (stage == "discovery" or repair):
            raise ValueError("Activation inspection is reserved for stock discovery")
        if (request.get("patch") is None) != (stage == "discovery" or repair):
            raise ValueError("Incorrect probe patch binding")
    elif not request.get("freeze_sha256"):
        raise ValueError("Evaluation requires a frozen candidate")


def load_practice(request):
    manifest = read_bound(ROOT / "data/manifest.json")
    path = ROOT / "data/practice.jsonl"
    if digest(path) != manifest["splits"]["practice"]["sha256"]:
        raise ValueError("Practice cases changed")
    cases = [json.loads(line) for line in path.read_text().splitlines()]
    indexed = {c["case_id"]: c for c in cases}
    if len(indexed) != len(cases) or any(not k.startswith("practice-") for k in request["case_ids"]):
        raise ValueError("Invalid practice identity")
    return [indexed[k] for k in request["case_ids"]]


def validate_probe_patch(stage, patch, request, plan):
    coordinates = {(e["tensor"], c) for e in patch["edits"] for c in e["columns"]}
    if stage in {"candidate-a", "candidate-b"}:
        expected = plan["candidates"][stage]
        if len(coordinates) > 8 or any(e["factor"] != expected["factor"] for e in patch["edits"]):
            raise ValueError("Candidate factor or footprint differs from the frozen procedure")
        decision = read_bound(ROOT / request["columns_decision"], request["columns_decision_sha256"])
        allowed = {(e["tensor"], c) for e in decision["selected_columns"] for c in e["columns"]}
        if coordinates != allowed:
            raise ValueError("Candidate columns changed after localization")
    elif stage == "control":
        freeze = read_bound(ROOT / request["freeze"], request["freeze_sha256"])
        if not freeze["adaptive_selection_closed"] or digest(ROOT / request["patch"]) != freeze["control_patch_sha256"]:
            raise ValueError("Control differs from the frozen selection")
        selected = read_bound(ROOT / freeze["selected_patch"], freeze["selected_patch_sha256"])
        selected_coordinates = {(e["tensor"], c) for e in selected["edits"] for c in e["columns"]}
        if coordinates & selected_coordinates:
            raise ValueError("Random control overlaps selected columns")
        from collections import Counter
        def footprint(spec):
            counts = Counter()
            for e in spec["edits"]:
                counts[(e["tensor"], e["factor"])] += len(e["columns"])
            return counts
        if footprint(patch) != footprint(selected):
            raise ValueError("Random control must match selected layers, counts, and factors")


class Runtime:
    def __init__(self, args, protocol, output):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.protocol, self.output = torch, protocol, output
        model_path = Path(args.model_path).resolve()
        if protocol["model_revision"] not in model_path.name:
            raise ValueError("Unpinned model directory")
        manifest = model_path.parent / "model-files.json"
        verify_model(model_path, manifest, protocol)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Exactly one allocated CUDA GPU is required")
        gpu = torch.cuda.get_device_properties(0)
        if args.expected_gpu.lower() not in gpu.name.lower() or not torch.cuda.is_bf16_supported():
            raise ValueError("Incorrect GPU or unsupported BF16")
        torch.manual_seed(protocol["seed"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
            local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
            use_safetensors=True).to("cuda").eval().requires_grad_(False)
        if sum(p.numel() for p in self.model.parameters()) != protocol["parameters"]:
            raise ValueError("Incorrect model size")
        self.eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(self.eos if isinstance(self.eos, list) else [self.eos])
        self.backups = None
        self.restorations = []
        self.active_patch_label = None
        atomic(output / "hardware.json", {"gpu": gpu.name, "total_memory": gpu.total_memory,
               "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
               "requires_grad": any(p.requires_grad for p in self.model.parameters())})
        config = self.model.generation_config.to_dict()
        config.update(do_sample=False, num_beams=1, use_cache=True,
            max_new_tokens=protocol["max_new_tokens"], repetition_penalty=protocol["repetition_penalty"],
            pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.eos)
        atomic(output / "generation.json", config)

    def render(self, case, mode):
        return self.tokenizer.apply_chat_template(messages(case), tools=tools(), tokenize=False,
                    add_generation_prompt=True, enable_thinking=mode == "thinking")

    def generate(self, prompt, case_id, mode, capture=False, persist_prompt=True):
        torch, model = self.torch, self.model
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        prompt_ids = inputs["input_ids"][0].tolist()
        if len(prompt_ids) > self.protocol["max_input_tokens"]:
            raise ValueError("Frozen prompt limit exceeded")
        activations, handles = {}, []
        if capture:
            for name, module in model.named_modules():
                if name.endswith(".feed_forward.w2"):
                    activations[name + ".weight"] = []
                    def hook(_module, args, name=name):
                        activations[name + ".weight"].append(args[0][0, -1].detach().float().cpu())
                    handles.append(module.register_forward_pre_hook(hook))
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        try:
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, use_cache=True,
                    max_new_tokens=self.protocol["max_new_tokens"],
                    repetition_penalty=self.protocol["repetition_penalty"], num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.eos)
            torch.cuda.synchronize()
        finally:
            for handle in handles:
                handle.remove()
        ids = generated[0, len(prompt_ids):].tolist()
        record = {"case_id": case_id, "mode": mode, "generated_ids": ids,
            "raw_text": self.tokenizer.decode(ids, skip_special_tokens=False),
            "truncated": not ids or ids[-1] not in self.eos_ids,
            "seconds": time.monotonic() - start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt_tokens": len(prompt_ids)}
        if persist_prompt:
            record.update(prompt=prompt, prompt_ids=prompt_ids)
        return record, activations

    def inspect(self, case, record, activations):
        """A descriptive projection, not a gradient or a fitted repair."""
        torch = self.torch
        expected = self.tokenizer(native(oracle(case)), add_special_tokens=False)["input_ids"]
        actual = record["generated_ids"]
        divergence = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), None)
        result = {"case_id": case["case_id"], "outcome": assess(case, record),
                  "oracle_native_text": native(oracle(case)), "first_token_divergence": divergence}
        if divergence is None or divergence >= min(len(v) for v in activations.values()):
            result["attributions"] = []
            return result
        desired, emitted = expected[divergence], actual[divergence]
        result.update(desired_token=desired, emitted_token=emitted,
                      desired_text=self.tokenizer.decode([desired]), emitted_text=self.tokenizer.decode([emitted]))
        parameters = dict(self.model.named_parameters())
        # No autograd graph is built. Use the actual unembedding rows and frozen column vectors.
        with torch.inference_mode():
            head = self.model.get_output_embeddings().weight
            direction = (head[desired].float() - head[emitted].float())
            rows = []
            for name, steps in activations.items():
                weight = parameters[name]
                projection = torch.matmul(direction, weight.float()).cpu()
                activation = steps[divergence]
                contribution = activation * projection
                indices = torch.topk(contribution, min(12, len(contribution)), largest=False).indices.tolist()
                for column in indices:
                    rows.append({"tensor": name, "column": column,
                        "approx_contribution": float(contribution[column]),
                        "activation": float(activation[column]), "projection": float(projection[column]),
                        "column_l2": float(weight[:, column].float().norm())})
        result["attributions"] = sorted(rows, key=lambda x: x["approx_contribution"])
        return result

    def set_patch(self, patch, label):
        torch = self.torch
        if self.backups is not None:
            restore(self.model, self.backups)
            verified = {}
            for name, tensor in self.backups.items():
                actual_hash = tensor_hash(dict(self.model.named_parameters())[name])
                if actual_hash != tensor_hash(tensor):
                    raise ValueError("Restoration failed")
                verified[name] = actual_hash
            self.restorations.append({"patch_label": self.active_patch_label, "restored_sha256": verified})
            atomic(self.output / "restore.json", self.restorations)
            self.backups = None
        if patch is None:
            return None
        from safetensors.torch import save_file
        self.backups, receipt = apply_patch(self.model, patch, self.protocol)
        self.active_patch_label = label
        tensors, details = {}, []
        parameters = dict(self.model.named_parameters())
        for i, item in enumerate(patch["edits"]):
            name, columns = item["tensor"], item["columns"]
            original, current = self.backups[name], parameters[name].detach()
            mask = torch.ones(current.shape[1], dtype=torch.bool, device=current.device)
            all_columns = [c for edit in patch["edits"] if edit["tensor"] == name for c in edit["columns"]]
            mask[all_columns] = False
            if not torch.equal(current[:, mask], original[:, mask]):
                raise ValueError("Weights outside the declared columns changed")
            before, after = original[:, columns].cpu().contiguous(), current[:, columns].cpu().contiguous()
            tensors[f"edit_{i}.before"], tensors[f"edit_{i}.after"] = before, after
            delta = after.float() - before.float()
            details.append({**item, "shape": list(current.shape), "actual_changed_scalars": int((after != before).sum()),
                            "delta_l2": float(delta.norm()), "delta_max_abs": float(delta.abs().max())})
        patch_file = self.output / f"{label}-columns.safetensors"
        save_file(tensors, str(patch_file))
        receipt.update(edits=details, columns_file=patch_file.name, columns_file_sha256=digest(patch_file))
        atomic(self.output / f"{label}-patch-receipt.json", receipt)
        # Verify reversal and exact replay from the persisted bytes before measuring the patch.
        from .patch_artifact import replay_columns
        replay_columns(self.model, patch, receipt, patch_file, self.protocol, reverse=True)
        replay_columns(self.model, patch, receipt, patch_file, self.protocol)
        atomic(self.output / f"{label}-replay.json", {"saved_artifact_revert_and_apply_verified": True,
                                                    "columns_file_sha256": digest(patch_file)})
        return receipt


def append_record(path, record):
    import os
    with path.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def probe(args, plan, request, protocol, output, runtime, start):
    cases = load_practice(request)
    if request.get("patch"):
        patch = read_bound(ROOT / request["patch"], request["patch_sha256"])
        validate_probe_patch(args.stage, patch, request, plan)
        runtime.set_patch(patch, args.stage)
    if request.get("capture_activations"):
        atomic(output / "tensors.json", {name: {"shape": list(tensor.shape), "before_sha256": tensor_hash(tensor)}
            for name, tensor in runtime.model.named_parameters() if name.endswith(".feed_forward.w2.weight")})
    records, inspection = [], []
    for case in cases:
        if time.monotonic() - start >= args.deadline_seconds:
            raise TimeoutError("Phase generation deadline reached")
        record, activations = runtime.generate(runtime.render(case, "direct"), case["case_id"], "direct",
                                               capture=request.get("capture_activations", False))
        record["probe_episode"] = len(records) + 1
        append_record(output / "responses.jsonl", record)
        records.append(record)
        if activations:
            inspection.append(runtime.inspect(case, record, activations))
            atomic(output / "inspection.json", inspection)
        atomic(output / "summary.json", summary(cases, records))
        atomic(output / "status.json", {"phase": "generating", "stage": args.stage,
            "responses": len(records), "probe_episodes_used": len(records), "seconds": time.monotonic() - start})
    runtime.set_patch(None, "restored")
    return len(records)


def evaluate_socket(args, plan, request, protocol, output, runtime, start):
    import os
    freeze = read_bound(ROOT / request["freeze"], request["freeze_sha256"])
    if not freeze["adaptive_selection_closed"] or freeze["final_feedback_seen"]:
        raise ValueError("Candidate selection must close before evaluation")
    if freeze["plan_sha256"] != digest(Path(args.plan)) or freeze["protocol_sha256"] != digest(ROOT / "protocol.json"):
        raise ValueError("Evaluation freeze changed")
    expected_arms = plan["evaluation"]["arms"]
    if set(request["arms"]) != set(expected_arms):
        raise ValueError("Evaluation arms changed")
    for name, arm in request["arms"].items():
        if arm["mode"] != expected_arms[name]["mode"]:
            raise ValueError("Evaluation mode changed")
        if name in {"stock-direct", "stock-thinking"}:
            if arm.get("patch") is not None:
                raise ValueError("Stock evaluation cannot have a patch")
        else:
            prefix = "selected" if name == "edited-direct" else "control"
            if (arm["patch"], arm["patch_sha256"]) != (freeze[prefix + "_patch"], freeze[prefix + "_patch_sha256"]):
                raise ValueError("Evaluation patch changed after selection")
            read_bound(ROOT / arm["patch"], arm["patch_sha256"])
    if request["expected_cases"] != plan["evaluation"]["expected_cases"] or request["maximum_generations"] != 800:
        raise ValueError("Evaluation case budget changed")
    sock_path = Path(args.socket_path).resolve()
    if sock_path != output.parents[1] / "evaluation.sock" or sock_path.exists():
        raise ValueError("Unexpected or existing evaluation socket")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    os.chmod(sock_path, 0o600)
    server.listen(1)
    server.settimeout(max(1, args.deadline_seconds - (time.monotonic() - start)))
    atomic(output / "status.json", {"phase": "waiting_for_sealed_evaluator", "responses": 0,
        "freeze_sha256": request["freeze_sha256"], "seconds": time.monotonic() - start})
    count = 0
    try:
        connection, _ = server.accept()
        with connection, connection.makefile("rwb") as stream:
            incoming = stream.readline(16 * 1024 * 1024 + 1)
            if not incoming.endswith(b"\n") or len(incoming) > 16 * 1024 * 1024:
                raise ValueError("Evaluation payload exceeds the cap")
            payload = json.loads(incoming)
            if set(payload) != {"freeze_sha256", "jobs"} or payload["freeze_sha256"] != request["freeze_sha256"]:
                raise ValueError("Sealed evaluator supplied the wrong freeze")
            jobs = payload["jobs"]
            if len(jobs) != 200 or len({j["case_id"] for j in jobs}) != 200:
                raise ValueError("Duplicate or incomplete sealed evaluation jobs")
            from collections import Counter
            if dict(Counter(j["split"] for j in jobs)) != request["expected_cases"]:
                raise ValueError("Sealed split sizes changed")
            for job in jobs:
                if set(job) != {"case_id", "split", "messages", "tools"}:
                    raise ValueError("Do not send scorer labels or case metadata to the model")
                if not job["case_id"].startswith(job["split"] + "-") or job["tools"] != tools():
                    raise ValueError("Sealed identity or tools changed")
                if [m["role"] for m in job["messages"]] != ["system", "user", "assistant", "tool"]:
                    raise ValueError("Native conversation roles changed")
            for arm_name, arm in request["arms"].items():
                patch = read_bound(ROOT / arm["patch"], arm["patch_sha256"]) if arm.get("patch") else None
                runtime.set_patch(patch, arm_name)
                for job in jobs:
                    if count >= request["maximum_generations"] or time.monotonic() - start >= args.deadline_seconds:
                        raise TimeoutError("Frozen evaluation budget reached")
                    prompt = runtime.tokenizer.apply_chat_template(job["messages"], tools=job["tools"],
                        tokenize=False, add_generation_prompt=True, enable_thinking=arm["mode"] == "thinking")
                    record, _ = runtime.generate(prompt, job["case_id"], arm["mode"])
                    record.update(arm=arm_name, split=job["split"])
                    # Prompts are in RAM and on the local evaluator only, never in cluster files.
                    remote_record = {k: v for k, v in record.items() if k not in {"prompt", "prompt_ids"}}
                    append_record(output / "responses.jsonl", remote_record)
                    stream.write((json.dumps({"type": "record", "record": record}, ensure_ascii=False) + "\n").encode())
                    stream.flush()
                    count += 1
                    atomic(output / "status.json", {"phase": "evaluating", "arm": arm_name,
                        "responses": count, "probe_episodes_used": 0, "seconds": time.monotonic() - start})
                runtime.set_patch(None, "restored")
            stream.write((json.dumps({"type": "complete", "responses": count}) + "\n").encode())
            stream.flush()
    finally:
        server.close()
        sock_path.unlink(missing_ok=True)
    return count


def run(args):
    if not args.approve_run:
        raise ValueError("Editing phase approval is required")
    protocol = read_bound(ROOT / "protocol.json")
    plan, request = read_bound(args.plan), read_bound(args.request)
    validate_request(args.stage, request, plan, protocol)
    if (ROOT / "data/final.jsonl").exists() or (ROOT / "data/heldout-seed.json").exists():
        raise ValueError("Final cases and seed must not be accessible to this worker")
    ops_binding = verify_ops_binding(args, plan)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(p.name != ".run.lock" for p in output.iterdir()):
            raise ValueError("Refusing to overwrite an existing stage or repeat probe episodes")
        binding = {"stage": args.stage, "source": source_binding(), "plan_sha256": digest(Path(args.plan)),
            "request_sha256": digest(Path(args.request)), "model_revision": protocol["model_revision"],
            "data_manifest_sha256": digest(ROOT / "data/manifest.json"),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata.get("Name")},
            "ops": ops_binding}
        atomic(output / "binding.json", binding)
        start = time.monotonic()
        atomic(output / "status.json", {"phase": "loading", "stage": args.stage, "responses": 0})
        runtime = Runtime(args, protocol, output)
        count = probe(args, plan, request, protocol, output, runtime, start) if args.stage in PROBE_STAGES else evaluate_socket(
            args, plan, request, protocol, output, runtime, start)
        atomic(output / "status.json", {"phase": "complete", "stage": args.stage, "responses": count,
            "probe_episodes_used": count if args.stage in PROBE_STAGES else 0, "seconds": time.monotonic() - start})
        return 0


def main():
    parser = argparse.ArgumentParser()
    for name in ["stage", "plan", "request", "model-path", "output", "expected-gpu"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--approve-run", action="store_true")
    parser.add_argument("--deadline-seconds", type=float, default=5400)
    parser.add_argument("--socket-path")
    args = parser.parse_args()
    try:
        raise SystemExit(run(args))
    except Exception as error:
        path = Path(args.output)
        if path.is_dir():
            atomic(path / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
