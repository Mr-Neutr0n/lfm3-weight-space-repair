"""Recompute the published measurements without loading a model."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def calculate(rows):
    groups = defaultdict(dict)
    for row in rows:
        key = row['suite'], row['arm']
        if row['case_ref'] in groups[key]:
            raise ValueError('Duplicate suite/arm/case')
        if type(row['success']) is not bool or type(row['tokens']) is not int:
            raise ValueError('Invalid outcome type')
        if row['tokens'] < 0:
            raise ValueError('Invalid token count')
        groups[key][row['case_ref']] = row
    result = {}
    for suite in sorted({k[0] for k in groups}):
        arms = {arm: cases for (name, arm), cases in groups.items() if name == suite}
        reference = arms['stock-direct']
        report = {'arms': {}, 'paired_vs_stock_direct': {}}
        for arm, cases in sorted(arms.items()):
            if set(cases) != set(reference):
                raise ValueError('Comparators require identical complete cases')
            counts = Counter(row['category'] for row in cases.values())
            n = len(cases)
            successes = sum(row['success'] for row in cases.values())
            tokens = sum(row['tokens'] for row in cases.values())
            report['arms'][arm] = {
                'n': n, 'successes': successes, 'accuracy': successes / n,
                'tokens': tokens, 'tokens_per_case': tokens / n,
                'categories': dict(sorted(counts.items())),
                'unchanged_n': sum(not r['changed'] for r in cases.values()),
                'unchanged_successes': sum(r['success'] for r in cases.values() if not r['changed']),
                'updated_n': sum(r['changed'] for r in cases.values()),
                'updated_successes': sum(r['success'] for r in cases.values() if r['changed']),
            }
            if arm == 'stock-direct':
                continue
            wins = sum(r['success'] and not reference[k]['success'] for k, r in cases.items())
            losses = sum(not r['success'] and reference[k]['success'] for k, r in cases.items())
            discordant = wins + losses
            p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant)
            report['paired_vs_stock_direct'][arm] = {
                'gains': wins, 'regressions': losses,
                'difference_pp': 100 * (wins - losses) / n,
                'exact_mcnemar_p': p,
            }
        result[suite] = report
    return result


def verify_manifest():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Published file changed: ' + name)
    return len(manifest['files'])


def replay_public_raw():
    from wsp.edit_metrics import assess
    case_paths = [ROOT / 'data' / f'{split}.jsonl' for split in ['practice', 'development']]
    cases = {c['case_id']: c for p in case_paths for c in map(json.loads, p.read_text().splitlines())}
    checked = 0
    for line in (ROOT / 'data/public-responses.jsonl').read_text().splitlines():
        row = json.loads(line)
        outcome = assess(cases[row['case_id']], row)
        if outcome['success'] != row['recorded_success'] or outcome['category'] != row['recorded_category']:
            raise ValueError('Public raw response does not replay: ' + row['case_id'])
        checked += 1
    return checked


def main():
    rows = [json.loads(line) for line in (ROOT / 'data/evaluation-outcomes.jsonl').read_text().splitlines()]
    actual = calculate(rows)
    if actual != json.loads((ROOT / 'results/summary.json').read_text()):
        raise ValueError('Recorded outcomes do not reproduce the published summary')
    raw_count = replay_public_raw()
    print(json.dumps({'file_hashes_verified': verify_manifest(), 'raw_practice_development_responses_rescored': raw_count,
                      'summary': actual, 'private_final_inputs_included': False}, indent=2))


if __name__ == '__main__':
    main()
