"""Inspect historical evidence and reproduce verdict weaknesses without training."""
import ast
import hashlib
import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'scripts/tea_to_tea_discriminating.py'
tree = ast.parse(source.read_text())
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'_median', 'verdict'}]
thresholds = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {'V_PROVEN', 'V_FAILED'}:
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'dict':
                    thresholds[target.id] = {kw.arg: ast.literal_eval(kw.value) for kw in node.value.keywords}
                else:
                    thresholds[target.id] = ast.literal_eval(node.value)
env = {'np': SimpleNamespace(median=median), **thresholds}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), env)
good = {'log_area_mae': 0.1, 'position_error_sections': 1., 'max_area_error_cm2': 0.1, 'plateaued': False}
bad = {**good, 'log_area_mae': 1., 'position_error_sections': 30.}
historical = json.loads((ROOT/'task009_results/smoke/report.json').read_text())
result = {
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'historical_config': historical['config'],
    'historical_verdict': historical['verdict'],
    'historical_converged': all(r['plateaued'] for rows in historical['per_seed_metrics'].values() for r in rows),
    'controls_are_synthetic_rows_not_new_model_measurements': True,
    'one_target_one_unconverged_seed': env['verdict']({'T1': [good]}),
    'median_hides_one_failed_seed': env['verdict']({'T1': [good, good, bad], 'T2': [good, good, good]}),
    'missing_tracked_artifacts': [str(p.relative_to(ROOT)) for t in historical['targets'].values() for p in [ROOT/t['wav'], ROOT/t['gt']] if not p.exists()],
}
print(json.dumps(result, indent=2))
