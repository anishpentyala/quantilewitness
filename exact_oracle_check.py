"""Development-only exact oracle: enumerate vertices of the full N-value LP.

Small samples only. Independent of the production compression and simplex.
Bounded closed polyhedra are nonempty exactly when they have a vertex.
"""
import argparse
from fractions import Fraction as Q
from itertools import combinations
import json
from pathlib import Path
import random
import time

from quantilewitness import audit, verify_report


def full_constraints(s):
    n = s['n']
    rows = []

    def add(a, b):
        rows.append((tuple(a), Q(b)))

    for i in range(n):
        a = [Q(0)] * n;a[i] = Q(1)
        add(a, s['support']['upper'])
        add([-v for v in a], -Q(s['support']['lower']))
    for i in range(n - 1):
        a = [Q(0)] * n;a[i], a[i + 1] = Q(1), Q(-1)
        add(a, 0)
    for claim in s['quantiles']:
        p = Q(claim['p']);a = [Q(0)] * n
        if s['method'] == 'linear':
            j, numerator = divmod((n - 1) * p.numerator, p.denominator)
            g = Q(numerator, p.denominator)
            a[j] = 1 - g
            if g:
                a[j + 1] = g
        else:
            j = (n * p.numerator + p.denominator - 1) // p.denominator - 1
            a[max(0, min(n - 1, j))] = Q(1)
        add(a, claim['upper'])
        add([-v for v in a], -Q(claim['lower']))
    if 'mean' in s:
        a = [Q(1, n)] * n
        add(a, s['mean']['upper'])
        add([-v for v in a], -Q(s['mean']['lower']))
    return rows


def intersection(equalities, n):
    matrix = [list(row) + [bound] for row, bound in equalities]
    for j in range(n):
        i = next((i for i in range(j, n) if matrix[i][j]), None)
        if i is None:
            return None
        matrix[i], matrix[j] = matrix[j], matrix[i]
        divisor = matrix[j][j]
        matrix[j] = [v / divisor for v in matrix[j]]
        for i in range(n):
            if i != j and matrix[i][j]:
                factor = matrix[i][j]
                matrix[i] = [v - factor * w for v, w in zip(matrix[i], matrix[j])]
    return [row[-1] for row in matrix]


def full_exact_feasible(s):
    n = s['n']
    if not 1 <= n <= 4:
        raise ValueError('vertex oracle supports n=1..4 only')
    rows = full_constraints(s)
    for selected in combinations(rows, n):
        x = intersection(selected, n)
        if x is not None and all(sum(a * b for a, b in zip(row, x)) <= bound for row, bound in rows):
            return True
    return False


def empirical_quantile(xs, p, method):
    if method == 'inverted_cdf':
        j = (len(xs) * p.numerator + p.denominator - 1) // p.denominator
        return xs[max(0, j - 1)]
    j, remainder = divmod((len(xs) - 1) * p.numerator, p.denominator)
    if not remainder:
        return xs[j]
    return xs[j] + Q(remainder, p.denominator) * (xs[j + 1] - xs[j])


def main(count=48, output=None):
    rng = random.Random(110922)
    start = time.monotonic()
    cases, counts = [], {}
    epsilon = Q(1, 10**30)
    for k in range(count):
        n = 1 + k % 4
        method = ('linear', 'inverted_cdf')[(k // 4) % 2]
        xs = sorted(Q(rng.randint(-6, 15), 3) for _ in range(n))
        s = {'schema': 'quantilewitness/input-v1', 'n': n, 'method': method,
             'support': {'lower': -2, 'upper': 5}, 'quantiles': []}
        for _ in range(rng.randint(1, 3)):
            p = Q(rng.randint(0, 5), 5)
            value = empirical_quantile(xs, p, method)
            shift = 0 if k % 3 == 0 else rng.choice((0, epsilon, -epsilon, Q(1, 2)))
            radius = 0 if k % 3 == 0 else rng.choice((0, epsilon))
            s['quantiles'].append({'p': str(p), 'lower': str(value + shift - radius),
                                   'upper': str(value + shift + radius)})
        if k % 5:
            value = sum(xs) / n
            shift = 0 if k % 3 == 0 else rng.choice((0, epsilon, -epsilon))
            s['mean'] = {'lower': str(value + shift), 'upper': str(value + shift)}
        if n == 1 and k % 3:
            s['quantiles'] = [{'p': '1/2', 'lower': str(xs[0]), 'upper': str(xs[0])}]
            s['mean'] = {'lower': str(xs[0] + epsilon), 'upper': str(xs[0] + epsilon)}
        expected = full_exact_feasible(s)
        result = audit(s)
        assert result['status'] in ('FEASIBLE', 'INFEASIBLE'), (k, s, result)
        assert (result['status'] == 'FEASIBLE') == expected, (k, s, expected, result)
        assert verify_report(s, result), (k, s, result)
        if k % 3 == 0:
            assert expected, (k, s)
        counts[result['status']] = counts.get(result['status'], 0) + 1
        cases.append({'spec': s, 'report': result, 'full_n_exact_feasible': expected})
    summary = {'seed': 110922, 'cases': count, 'counts': counts, 'seconds': time.monotonic() - start}
    if output is not None:
        Path(output).write_text(json.dumps(dict(summary, fixtures=cases), indent=2))
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, default=48)
    parser.add_argument('--output', help='optional JSON fixture/results output path')
    args = parser.parse_args()
    if not 1 <= args.cases <= 128:
        parser.error('--cases must be between 1 and 128')
    main(args.cases, args.output)
