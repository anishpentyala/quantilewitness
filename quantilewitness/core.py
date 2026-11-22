"""Exact, bounded continuous-sample compatibility. No data provenance inference."""
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import json
import math
import re

INPUT_SCHEMA = 'quantilewitness/input-v1'
REPORT_SCHEMA = 'quantilewitness/report-v1'
SCOPE = 'bounded continuous finite sample; closed claim intervals; not data provenance'
MAX_CLAIMS = 16
MAX_N = 10**12
VERIFY_BITS = 8192
_NUMBER = re.compile(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+|/[1-9][0-9]*)?\Z', re.ASCII)


class Invalid(ValueError):
    pass


class Limit(Exception):
    pass


def number(value, *, proof=False):
    chars, bits = (2500, 4096) if proof else (100, 256)
    if type(value) is int:
        if value.bit_length() > bits:
            raise Invalid('number exceeds bit limit')
        return F(value)
    if not isinstance(value, str) or len(value) > chars or not _NUMBER.fullmatch(value):
        raise Invalid('numbers must be integers or bounded fixed-decimal/rational strings')
    result = F(value)
    if max(result.numerator.bit_length(), result.denominator.bit_length()) > bits:
        raise Invalid('number exceeds bit limit')
    return result


def _object(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or set(value) - set(required) - set(optional):
        raise Invalid('missing or unknown object fields')
    return value


def interval(value):
    obj = _object(value, ('lower', 'upper'))
    lo, hi = number(obj['lower']), number(obj['upper'])
    if lo > hi:
        raise Invalid('interval lower exceeds upper')
    return lo, hi


@dataclass(frozen=True)
class Spec:
    n: int
    method: str
    lower: F
    upper: F
    claims: tuple
    mean: tuple | None


def parse_spec(data):
    obj = _object(data, ('schema', 'n', 'method', 'support', 'quantiles'), ('mean',))
    if obj['schema'] != INPUT_SCHEMA:
        raise Invalid('unsupported input schema')
    n = obj['n']
    if type(n) is not int or not 1 <= n <= MAX_N:
        raise Invalid('n must be an integer from 1 through 1000000000000')
    method = obj['method']
    if method not in ('linear', 'inverted_cdf'):
        raise Invalid('method must be linear or inverted_cdf')
    lo, hi = interval(obj['support'])
    claims = obj['quantiles']
    if not isinstance(claims, list) or not 1 <= len(claims) <= MAX_CLAIMS:
        raise Invalid('quantiles must contain 1 through 16 claims')
    qs = []
    for claim in claims:
        _object(claim, ('p', 'lower', 'upper'))
        p, a, b = (number(claim[k]) for k in ('p', 'lower', 'upper'))
        if not 0 <= p <= 1 or a > b:
            raise Invalid('invalid probability or claim interval')
        qs.append((p, a, b))
    return Spec(n, method, lo, hi, tuple(qs), interval(obj['mean']) if 'mean' in obj else None)


def spec_json(spec):
    result = {'schema': INPUT_SCHEMA, 'n': spec.n, 'method': spec.method,
              'support': {'lower': str(spec.lower), 'upper': str(spec.upper)},
              'quantiles': [{'p': str(p), 'lower': str(lo), 'upper': str(hi)} for p, lo, hi in spec.claims]}
    if spec.mean is not None:
        result['mean'] = dict(zip(('lower', 'upper'), map(str, spec.mean)))
    return result


def digest(spec):
    return hashlib.sha256(json.dumps(spec_json(spec), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def positions(n, method, p):
    if method == 'inverted_cdf':
        rank = max(0, min(n - 1, math.ceil(n * p) - 1))
        return ((rank, F(1)),)
    h = (n - 1) * p
    i = h.numerator // h.denominator
    g = h - i
    return ((i, 1 - g), (i + 1, g)) if g else ((i, F(1)),)


@dataclass(frozen=True)
class Model:
    segments: tuple  # (zero-based first rank, positive run length)
    rows: tuple
    rhs: tuple
    names: tuple


def compile_model(spec):
    pins = {0, spec.n - 1}
    for p, _, _ in spec.claims:
        pins.update(i for i, _ in positions(spec.n, spec.method, p))
    segments, index = [], {}
    cursor = 0
    for rank in sorted(pins):
        if rank > cursor:
            segments.append((cursor, rank - cursor))
        index[rank] = len(segments)
        segments.append((rank, 1))
        cursor = rank + 1
    assert cursor == spec.n
    width = len(segments)
    rows, rhs, names = [], [], []

    def add(name, coefficients, bound):
        rows.append(tuple(coefficients))
        rhs.append(bound)
        names.append(name)

    # Variables are shifted values z = sample_value - support.lower, with z >= 0.
    for i in range(width - 1):
        row = [F(0)] * width
        row[i], row[i + 1] = F(1), F(-1)
        add('order:' + str(i), row, F(0))
    row = [F(0)] * width
    row[-1] = F(1)
    add('support:upper', row, spec.upper - spec.lower)
    for k, (p, lo, hi) in enumerate(spec.claims):
        row = [F(0)] * width
        for rank, weight in positions(spec.n, spec.method, p):
            row[index[rank]] += weight
        add('quantile:' + str(k) + ':upper', row, hi - spec.lower)
        add('quantile:' + str(k) + ':lower', [-v for v in row], spec.lower - lo)
    if spec.mean is not None:
        row = [F(count, spec.n) for _, count in segments]
        add('mean:upper', row, spec.mean[1] - spec.lower)
        add('mean:lower', [-v for v in row], spec.lower - spec.mean[0])
    return Model(tuple(segments), tuple(rows), tuple(rhs), tuple(names))


def _bounded_sum(terms):
    total = F(0)
    for value in terms:
        total += value
        if max(total.numerator.bit_length(), total.denominator.bit_length()) > VERIFY_BITS:
            raise Limit('certificate arithmetic budget reached')
    return total


def primal_valid(model, x):
    try:
        return len(x) == len(model.segments) and all(v >= 0 for v in x) and all(
            _bounded_sum(a * v for a, v in zip(row, x)) <= b for row, b in zip(model.rows, model.rhs))
    except Limit:
        return False


def dual_valid(model, y):
    try:
        return (len(y) == len(model.rows) and all(v >= 0 for v in y)
                and _bounded_sum(v * b for v, b in zip(y, model.rhs)) < 0
                and all(_bounded_sum(y[i] * model.rows[i][j] for i in range(len(y))) >= 0
                        for j in range(len(model.segments))))
    except Limit:
        return False


def solve(model, *, pivot_limit=2000, bit_limit=4096):
    """Phase-I simplex with exact arithmetic, Bland choices, and row multipliers."""
    m, n = len(model.rows), len(model.segments)
    d = [list(row) + [F(-1), b] for row, b in zip(model.rows, model.rhs)]
    d.append([F(0)] * n + [F(1), F(0)])
    multipliers = [[F(int(i == j)) for j in range(m)] for i in range(m)] + [[F(0)] * m]
    basis, nonbasis = list(range(n, n + m)), list(range(n)) + [-1]
    pivots = 0

    def pivot(r, s):
        nonlocal pivots
        if pivots >= pivot_limit:
            raise Limit('pivot budget reached')
        inverse = 1 / d[r][s]
        old, old_y = d[r][:], multipliers[r][:]
        for i in range(m + 1):
            if i == r:
                continue
            factor = d[i][s] * inverse
            if factor:
                for j in range(n + 2):
                    if j != s:
                        d[i][j] -= old[j] * factor
                for j in range(m):
                    multipliers[i][j] -= old_y[j] * factor
            d[i][s] = -factor
        for j in range(n + 2):
            d[r][j] = old[j] * inverse if j != s else inverse
        multipliers[r] = [v * inverse for v in old_y]
        basis[r], nonbasis[s] = nonbasis[s], basis[r]
        pivots += 1
        for matrix in (d, multipliers):
            if any(max(v.numerator.bit_length(), v.denominator.bit_length()) > bit_limit
                   for row in matrix for v in row):
                raise Limit('rational arithmetic budget reached')

    r = min(range(m), key=lambda i: (d[i][-1], basis[i]))
    if d[r][-1] < 0:
        pivot(r, n)
        while True:
            eligible = [j for j in range(n + 1) if d[m][j] < 0]
            if not eligible:
                break
            s = min(eligible, key=lambda j: nonbasis[j])
            eligible_rows = [i for i in range(m) if d[i][s] > 0]
            if not eligible_rows:
                raise Limit('unexpected unbounded phase-I objective')
            r = min(eligible_rows, key=lambda i: (d[i][-1] / d[i][s], basis[i]))
            pivot(r, s)
        if d[m][-1] < 0:
            y = multipliers[m]
            if not dual_valid(model, y):
                raise Limit('internal contradiction certificate check failed')
            return 'INFEASIBLE', y, pivots
        if d[m][-1] != 0:
            raise Limit('unexpected phase-I objective')
    x = [F(0)] * n
    for i, variable in enumerate(basis):
        if 0 <= variable < n:
            x[variable] = d[i][-1]
    if not primal_valid(model, x):
        raise Limit('internal sample certificate check failed')
    return 'FEASIBLE', x, pivots


def runs_from_solution(spec, model, x):
    runs = []
    for (_, count), z in zip(model.segments, x):
        value = str(z + spec.lower)
        if runs and runs[-1]['value'] == value:
            runs[-1]['count'] += count
        else:
            runs.append({'count': count, 'value': value})
    return runs


def witness_valid(spec, runs):
    """Check sample claims from run lengths, without the model or simplex."""
    if not isinstance(runs, list) or not 1 <= len(runs) <= 4 * MAX_CLAIMS + 3:
        return False
    total, weighted, previous, parsed = 0, F(0), None, []
    for run in runs:
        _object(run, ('count', 'value'))
        count = run['count']
        if type(count) is not int or not 1 <= count <= spec.n:
            return False
        value = number(run['value'], proof=True)
        if not spec.lower <= value <= spec.upper or (previous is not None and previous > value):
            return False
        total += count
        parsed.append((total, value))
        if spec.mean is not None:
            try:
                weighted = _bounded_sum((weighted, count * value))
            except Limit:
                return False
        previous = value
    if total != spec.n or (spec.mean is not None and not spec.mean[0] <= weighted / total <= spec.mean[1]):
        return False

    def order_stat(rank):
        for end, value in parsed:
            if rank < end:
                return value
        raise Invalid('witness rank outside sample')

    for p, lo, hi in spec.claims:
        # Separately implement the declared quantile definition, not positions().
        if spec.method == 'linear':
            h = p * (spec.n - 1)
            a = h.numerator // h.denominator
            g = h - a
            value = (1 - g) * order_stat(a)
            if g:
                value += g * order_stat(a + 1)
        else:
            product = p * spec.n
            rank = (product.numerator + product.denominator - 1) // product.denominator - 1
            value = order_stat(max(0, min(spec.n - 1, rank)))
        if not lo <= value <= hi:
            return False
    return True


def verify_report(data, report):
    """Verify a result's certificate; never accept a verdict without a witness."""
    try:
        spec = parse_spec(data)
        if (not isinstance(report, dict) or report.get('schema') != REPORT_SCHEMA
                or report.get('input_sha256') != digest(spec)):
            return False
        if report.get('status') not in ('FEASIBLE', 'INFEASIBLE'):
            return False
        proof_field = 'witness' if report['status'] == 'FEASIBLE' else 'contradiction'
        if set(report) != {'schema', 'input_sha256', 'scope', 'model', 'status', 'pivots', proof_field}:
            return False
        model = compile_model(spec)
        metadata = report.get('model')
        if (report.get('scope') != SCOPE or type(report.get('pivots')) is not int or report['pivots'] < 0
                or not isinstance(metadata, dict) or any(type(v) is not int for v in metadata.values())
                or metadata != {'n': spec.n, 'segments': len(model.segments), 'constraints': len(model.rows)}):
            return False
        if report['status'] == 'FEASIBLE':
            return witness_valid(spec, report.get('witness'))
        weights = report.get('contradiction')
        if not isinstance(weights, list) or not 1 <= len(weights) <= len(model.rows):
            return False
        by_name = dict.fromkeys(model.names, F(0))
        seen = set()
        for item in weights:
            _object(item, ('constraint', 'weight'))
            name = item['constraint']
            if not isinstance(name, str) or name not in by_name or name in seen:
                return False
            seen.add(name)
            by_name[name] = number(item['weight'], proof=True)
        return dual_valid(model, [by_name[name] for name in model.names])
    except (Invalid, ValueError, TypeError, KeyError, OverflowError):
        return False


def audit(data, *, pivot_limit=2000, bit_limit=4096):
    try:
        if type(pivot_limit) is not int or pivot_limit < 0 or type(bit_limit) is not int or bit_limit < 1:
            raise Invalid('pivot_limit must be a nonnegative integer and bit_limit a positive integer')
        spec = parse_spec(data)
    except (Invalid, ValueError, TypeError) as exc:
        return {'schema': REPORT_SCHEMA, 'status': 'INVALID', 'reason': str(exc)}
    result = {'schema': REPORT_SCHEMA, 'input_sha256': digest(spec),
              'scope': SCOPE}
    model = compile_model(spec)
    result['model'] = {'n': spec.n, 'segments': len(model.segments), 'constraints': len(model.rows)}
    try:
        status, solution, pivots = solve(model, pivot_limit=pivot_limit, bit_limit=bit_limit)
        result.update(status=status, pivots=pivots)
        if status == 'FEASIBLE':
            result['witness'] = runs_from_solution(spec, model, solution)
        else:
            result['contradiction'] = [{'constraint': name, 'weight': str(value)}
                                       for name, value in zip(model.names, solution) if value]
        if not verify_report(data, result):
            raise Limit('independent result certificate check failed')
    except Limit as exc:
        result.update(status='UNRESOLVED', reason=str(exc))
        result.pop('witness', None)
        result.pop('contradiction', None)
    return result
