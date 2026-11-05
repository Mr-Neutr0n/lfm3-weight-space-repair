"""Read held-out cases only after selection closes; stream prompts without staging data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

from .edit_metrics import paired, summary
from .missions import ROOT, digest, messages, tools
from .runner import atomic


def run(args):
    # This process is the evaluator, not the editor or a probe-selection process.
    plan_path, freeze_path, request_path = [ROOT / name for name in ["editing/plan.json", "editing/freeze.json", "editing/requests/evaluation.json"]]
    plan, freeze, request = [json.loads(p.read_text()) for p in [plan_path, freeze_path, request_path]]
    if not freeze["adaptive_selection_closed"] or freeze["final_feedback_seen"] or digest(freeze_path) != request["freeze_sha256"]:
        raise ValueError("Selection is not frozen")
    if digest(plan_path) != freeze["plan_sha256"] or digest(ROOT / "protocol.json") != freeze["protocol_sha256"]:
        raise ValueError("Frozen plan or protocol changed")
    for prefix in ["selected", "control", "sol"]:
        if digest(ROOT / freeze[prefix + "_patch"]) != freeze[prefix + "_patch_sha256"]:
            raise ValueError("Frozen patch changed")
    manifest_path = ROOT / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if digest(manifest_path) != freeze["data_manifest_sha256"]:
        raise ValueError("Data manifest changed after selection")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected_binding = {"plan_sha256": digest(plan_path), "freeze_sha256": digest(freeze_path),
        "request_sha256": digest(request_path), "data_manifest_sha256": digest(manifest_path),
        "separate_process": True, "selection_closed_before_final_open": True,
        "final_data_staged_on_cluster": False}
    if args.resume_empty:
        if (set(p.name for p in output.iterdir()) != {"binding.json"}
                or json.loads((output / "binding.json").read_text()) != expected_binding):
            raise ValueError("Only an identical evaluator with zero saved responses can retry")
        sys.path.insert(0, str(ROOT / "ops"))
        import phases
        view = phases.status("evaluation")
        status = view.get("status", {})
        if (view.get("scheduler_proof", {}).get("state") != "RUNNING"
                or status.get("phase") != "waiting_for_sealed_evaluator" or status.get("responses") != 0):
            raise ValueError("Worker must confirm zero generations before an empty-stream retry")
        atomic(output / "transport-retry.json", {"reason": "Unix socket is local to dash2 kernel; client moved into existing Slurm allocation",
            "same_slurm_job": view["scheduler_proof"]["job_id"], "responses_before_retry": 0,
            "freeze_sha256": digest(freeze_path), "local_broker_source_sha256": digest(Path(__file__))})
    elif any(output.iterdir()):
        raise ValueError("Refusing to overwrite evaluator output or repeat evaluation")
    else:
        atomic(output / "binding.json", expected_binding)
    cases, jobs = [], []
    # Parent supplies freshly generated cases after the editor handoff.
    path = Path(args.cases).resolve()
    group = [json.loads(line) for line in path.read_text().splitlines()]
    if len(group) != 150 or len({c['case_id'] for c in group}) != 150:
        raise ValueError("Expected 150 distinct fresh final cases")
    if any(not c['case_id'].startswith('final-') for c in group):
        raise ValueError("Fresh final identity required")
    atomic(output / 'fresh-cases-binding.json', {'sha256': digest(path), 'cases': len(group)})
    cases.extend(group)
    jobs.extend({"case_id": c["case_id"], "split": "final", "messages": messages(c), "tools": tools()} for c in group)
    sys.path.insert(0, str(ROOT / "ops"))
    import cluster
    records = []
    seen = set()
    start = time.monotonic()
    def on_record(message):
        if message.get("type") == "complete":
            if message["responses"] != plan["evaluation"]["maximum_generations"] or len(records) != message["responses"]:
                raise ValueError("Incomplete sealed evaluation")
            return
        if message.get("type") != "record":
            raise ValueError("Unexpected evaluator message")
        record = message["record"]
        key = record["arm"], record["case_id"]
        if key in seen or record["arm"] not in plan["evaluation"]["arms"]:
            raise ValueError("Duplicate or unknown evaluated response")
        seen.add(key)
        from .editor_runner import append_record
        append_record(output / "responses.jsonl", record)
        records.append(record)
        atomic(output / "status.json", {"phase": "evaluating", "responses": len(records), "seconds": time.monotonic() - start})
        if len(records) % 50 == 0:
            print(json.dumps({"responses_received": len(records), "expected": 750}), flush=True)
    cluster.phase_stream_evaluation({"freeze_sha256": digest(freeze_path), "jobs": jobs}, on_record,
                                    timeout_seconds=args.timeout_seconds)
    if len(records) != plan["evaluation"]["maximum_generations"]:
        raise ValueError("Evaluation stream ended early")
    reports = {}
    for split in plan["evaluation"]["splits"]:
        group = [c for c in cases if c["case_id"].startswith(split + "-")]
        by_arm = {arm: [r for r in records if r["arm"] == arm and r["split"] == split] for arm in plan["evaluation"]["arms"]}
        reports[split] = {"arms": {arm: summary(group, rows) for arm, rows in by_arm.items()},
            "paired": {arm: paired(group, by_arm["stock-direct"], by_arm[arm]) for arm in by_arm if arm != "stock-direct"}}
    atomic(output / "summary.json", reports)
    atomic(output / "status.json", {"phase": "complete", "responses": len(records), "seconds": time.monotonic() - start,
        "selection_remains_closed": True})
    print(json.dumps({"phase": "complete", "responses": len(records), "summary": str(output / "summary.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=5400)
    parser.add_argument("--resume-empty", action="store_true")
    run(parser.parse_args())
