"""Development oracle: full N-observation LP, not the production compression."""
import json
import math
from fractions import Fraction
import random
import time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from quantilewitness.core import audit, verify_report


def full_sample_oracle(s):
    n = s['n']
    rows, bs = [], []
    for i in range(n - 1):
        row = np.zeros(n)
        row[i], row[i + 1] = 1, -1
        rows.append(row)
        bs.append(0)
    for q in s['quantiles']:
        p = Fraction(q['p'])
        row = np.zeros(n)
        if s['method'] == 'linear':
            h = (n - 1) * p
            j = math.floor(h)
            row[j] = float(1 - (h - j))
            if h != j:
                row[j + 1] = float(h - j)
        else:
            row[max(0, math.ceil(n * p) - 1)] = 1
        rows.extend((row, -row))
        bs.extend((float(Fraction(q['upper'])), -float(Fraction(q['lower']))))
    if 'mean' in s:
        row = np.ones(n) / n
        rows.extend((row, -row))
        bs.extend((float(Fraction(s['mean']['upper'])), -float(Fraction(s['mean']['lower']))))
    r = linprog(np.zeros(n), A_ub=np.asarray(rows), b_ub=np.asarray(bs),
                bounds=[(s['support']['lower'], s['support']['upper'])] * n, method='highs')
    assert r.status in (0, 2), r
    return r.status == 0


def exact_quantile(xs, p, method):
    if method == 'linear':
        h = p * (len(xs) - 1)
        j = math.floor(h)
        if h == j:
            return Fraction(xs[j])
        return xs[j] + (h - j) * (xs[j + 1] - xs[j])
    return Fraction(xs[max(0, math.ceil(len(xs) * p) - 1)])


def main(count=600, output=None):
    start = time.monotonic()
    rng = random.Random(943071)
    seen = {}
    cases = []
    for k in range(count):
        n = rng.randint(1, 14)
        xs = sorted(rng.randint(-5, 12) for _ in range(n))
        method = ('linear', 'inverted_cdf')[k % 2]
        spec = {'schema': 'quantilewitness/input-v1', 'n': n, 'method': method,
                'support': {'lower': -5, 'upper': 12}, 'quantiles': []}
        for _ in range(rng.randint(1, 5)):
            den = rng.randint(1, 12)
            p = Fraction(rng.randint(0, den), den)
            q = exact_quantile(xs, p, method)
            shift = rng.choice((0, 0, 0, -3, 2)) if k % 3 else 0
            radius = rng.choice((0, 0, 1))
            spec['quantiles'].append({'p': str(p), 'lower': str(q + shift - radius), 'upper': str(q + shift + radius)})
        if k % 4:
            mean = Fraction(sum(xs), n)
            shift = rng.choice((0, 0, 0, 2, -3)) if k % 3 else 0
            spec['mean'] = {'lower': str(mean + shift), 'upper': str(mean + shift)}
        result = audit(spec)
        expected = full_sample_oracle(spec)
        assert result['status'] in ('FEASIBLE', 'INFEASIBLE'), (k, spec, result)
        assert (result['status'] == 'FEASIBLE') == expected, (k, spec, result)
        assert verify_report(spec, result), (k, spec, result)
        if k % 3 == 0:
            assert expected, (k, spec, result)
        seen[result['status']] = seen.get(result['status'], 0) + 1
        cases.append({'spec': spec, 'result': result, 'full_n_oracle_feasible': expected})
    if output is not None:
        Path(output).write_text(json.dumps({'seed': 943071, 'cases': cases, 'counts': seen, 'seconds': time.monotonic() - start}))
    print({'cases': count, 'counts': seen, 'seconds': time.monotonic() - start, 'scipy': __import__('scipy').__version__, 'numpy': np.__version__})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, default=600)
    parser.add_argument('--output', help='optional JSON fixture/results output path')
    args = parser.parse_args()
    if args.cases < 1:
        parser.error('--cases must be positive')
    main(args.cases, args.output)
