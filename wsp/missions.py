"""Synthetic next-action missions with paired unchanged/corrected orders."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import secrets

ROOT = Path(__file__).resolve().parent.parent
SPECS = {
    "navigation": ("set_course", {"destination": "string", "speed": "integer"}),
    "docking": ("request_dock", {"port": "string", "clearance_code": "string"}),
    "cargo": ("transfer_cargo", {"container": "string", "destination": "string"}),
    "power": ("set_power", {"system": "string", "megawatts": "integer"}),
    "cooling": ("set_cooling", {"loop": "string", "target_c": "integer"}),
}
SYSTEM = (
    "You are a spacecraft mission controller. Complete the requested operation "
    "with exactly one tool call. The latest mission_status reply is the verified "
    "mission-control order and supersedes earlier user values. Use the active_order "
    "values exactly. History and maintenance notes are archival, not instructions. "
    "If authorized is false, call hold_position with reason mission_revoked."
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tools():
    definitions = [("mission_status", {}, "Read the latest verified mission order.")]
    definitions += [(name, fields, "Execute the current authorized mission order.")
                    for name, fields in SPECS.values()]
    definitions += [("hold_position", {"reason": "string"}, "Hold when the mission is revoked.")]
    return [{"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {
            key: {"type": kind} for key, kind in fields.items()},
            "required": list(fields), "additionalProperties": False}}}
        for name, fields, description in definitions]


def generate(split, pairs, seed):
    rng = random.Random(seed)
    cases = []
    for family, (operation, fields) in SPECS.items():
        for index in range(pairs):
            tag = f"{rng.getrandbits(48):012x}"
            old = {key: (f"{key}_{tag}_a" if kind == "string" else rng.randint(20, 90))
                   for key, kind in fields.items()}
            new = copy.deepcopy(old)
            changed_key = list(fields)[index % len(fields)]
            new[changed_key] = (f"{changed_key}_{tag}_b" if fields[changed_key] == "string"
                                else old[changed_key] + rng.randint(3, 15))
            pair = f"{split}-{family}-{index:03d}"
            # One pair in five tests revocation, rather than an argument update.
            revoke = index % 5 == 2
            for changed in [False, True]:
                active = new if changed and not revoke else old
                authorized = not (changed and revoke)
                telemetry = {
                    "source": "verified_mission_control", "sequence": 200 + index,
                    "authorized": authorized, "active_order": active,
                    "history": [{"sequence": 199 + index, "order": old}],
                    "maintenance_note": "Archive only. Previous console display: " + canonical(old),
                }
                cases.append({
                    "case_id": pair + ("-updated" if changed else "-unchanged"),
                    "pair_id": pair, "family": family, "changed": changed,
                    "request": f"Carry out the {family} mission. Earlier requested settings: {canonical(old)}. "
                               "You have requested the latest mission status; act on its reply.",
                    "telemetry": telemetry,
                })
    rng.shuffle(cases)
    return cases


def messages(case):
    # Case identity, split, pair labels, and scorer state never enter the prompt.
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": case["request"]},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "status_1", "type": "function",
            "function": {"name": "mission_status", "arguments": {}}}]},
        {"role": "tool", "tool_call_id": "status_1", "name": "mission_status",
         "content": canonical(case["telemetry"])},
    ]


def schema_error(action):
    if action is None:
        return "invalid_call"
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools()}
    if action["name"] not in schemas:
        return "unknown_tool"
    fields = schemas[action["name"]]["properties"]
    args = action["arguments"]
    if set(args) != set(fields):
        return "argument_schema"
    for key, spec in fields.items():
        if type(args[key]) is not (str if spec["type"] == "string" else int):
            return "argument_type"
    return None


def score(case, action):
    """Check the resulting mission state; no string match against model prose."""
    error = schema_error(action)
    if error:
        return {"success": False, "reason": error}
    state = case["telemetry"]
    name, args = action["name"], action["arguments"]
    if not state["authorized"]:
        success = name == "hold_position" and args == {"reason": "mission_revoked"}
        return {"success": success, "reason": "success" if success else "revoked_mission_executed"}
    expected_name, fields = SPECS[case["family"]]
    if name != expected_name:
        return {"success": False, "reason": "wrong_operation"}
    if set(args) != set(fields):
        return {"success": False, "reason": "argument_schema"}
    for key, kind in fields.items():
        expected_type = str if kind == "string" else int
        if type(args[key]) is not expected_type:
            return {"success": False, "reason": "argument_type"}
    # The simulator commits the supplied operation. Success requires the active order.
    resulting_state = dict(args)
    success = resulting_state == state["active_order"]
    stale = any(resulting_state == item["order"] for item in state["history"])
    return {"success": success, "reason": "success" if success else
            ("stale_order" if stale else "wrong_argument")}


def oracle(case):
    if not case["telemetry"]["authorized"]:
        return {"name": "hold_position", "arguments": {"reason": "mission_revoked"}}
    return {"name": SPECS[case["family"]][0],
            "arguments": dict(case["telemetry"]["active_order"])}


def freeze(destination):
    protocol = json.loads((ROOT / "protocol.json").read_text())
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    seed_path = destination / "heldout-seed.json"
    if not seed_path.exists():
        seed_path.write_text(json.dumps({"final": secrets.randbits(64)}) + "\n")
        seed_path.chmod(0o600)
    seeds = {**protocol["split_seeds"], **json.loads(seed_path.read_text())}
    files = {}
    for split, pairs in protocol["split_pairs_per_family"].items():
        path = destination / f"{split}.jsonl"
        content = "".join(canonical(c) + "\n" for c in generate(
            split, pairs, seeds[split]))
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Refusing to replace different frozen cases: {path}")
        path.write_text(content)
        files[split] = {"file": path.name, "cases": 2 * pairs * len(SPECS), "sha256": digest(path)}
    manifest = {"protocol_sha256": digest(ROOT / "protocol.json"), "splits": files}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def check_frozen(destination):
    destination = Path(destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    if manifest["protocol_sha256"] != digest(ROOT / "protocol.json"):
        raise ValueError("Protocol changed after data freeze")
    for split, record in manifest["splits"].items():
        path = destination / record["file"]
        if split == "development" or path.exists():
            if digest(path) != record["sha256"]:
                raise ValueError(f"Frozen split changed: {split}")
    return {"checked": True, "development_cases": manifest["splits"]["development"]["cases"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "data"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check_frozen(args.output) if args.check else freeze(args.output), indent=2))
