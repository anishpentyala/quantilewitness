"""Development-only assertion-based mutation controls in disposable copies."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

MUTATIONS = [
    ('reversed_linear_weights',
     'return ((i, 1 - g), (i + 1, g)) if g else ((i, F(1)),)',
     'return ((i, g), (i + 1, 1 - g)) if g else ((i, F(1)),)'),
    ('floor_nearest_rank',
     'math.ceil(n * p) - 1',
     'math.floor(n * p) - 1'),
    ('unweighted_segment_mean',
     'row = [F(count, spec.n) for _, count in segments]',
     'row = [F(1, len(segments)) for _, count in segments]'),
    ('reverse_order', 'row[i], row[i + 1] = F(1), F(-1)',
     'row[i], row[i + 1] = F(-1), F(1)'),
    ('admit_negative_dual_coefficients',
     '_bounded_sum(y[i] * model.rows[i][j] for i in range(len(y))) >= 0',
     '_bounded_sum(y[i] * model.rows[i][j] for i in range(len(y))) >= -1'),
    ('omit_nonnegative_weights',
     'len(y) == len(model.rows) and all(v >= 0 for v in y)',
     'len(y) == len(model.rows) and True'),
    ('open_claim_endpoints', 'if not lo <= value <= hi:', 'if not lo < value < hi:'),
    ('omit_scope_binding', "report.get('scope') != SCOPE", 'False'),
    ('omit_certificate_arithmetic_cap', '> VERIFY_BITS:', '> (10**9):'),
]


def main(output_dir=None):
    root = Path(__file__).resolve().parent
    original = (root / 'quantilewitness/core.py').read_text()
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, before, after in MUTATIONS:
        if original.count(before) != 1:
            raise ValueError('mutation no longer matches exactly once: ' + name)
        source = original.replace(before, after)
        compile(source, 'core.py', 'exec')
        with tempfile.TemporaryDirectory(prefix='quantilewitness-mutation-') as directory:
            path = Path(directory)
            for folder in ('quantilewitness', 'tests'):
                shutil.copytree(root / folder, path / folder,
                                ignore=shutil.ignore_patterns('__pycache__'))
            (path / 'quantilewitness/core.py').write_text(source)
            run = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-f', '-v'],
                                 cwd=path, capture_output=True, text=True)
        result = {'name': name, 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                  'returncode': run.returncode, 'assertion_failure': 'FAIL:' in run.stderr,
                  'error': 'ERROR:' in run.stderr}
        results.append(result)
        if output_dir is not None:
            (output_dir / (name + '.log')).write_text(run.stdout + run.stderr)
    report = {'original_core_sha256': hashlib.sha256(original.encode()).hexdigest(), 'mutants': results}
    if output_dir is not None:
        (output_dir / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if all(r['returncode'] == 1 and r['assertion_failure'] and not r['error'] for r in results) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', help='optional directory for logs and manifest')
    args = parser.parse_args()
    raise SystemExit(main(args.output_dir))
