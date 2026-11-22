import copy
from fractions import Fraction as F
import math
import random
import unittest

from quantilewitness.core import (INPUT_SCHEMA, Invalid, audit, compile_model, digest,
                                 number, parse_spec, verify_report, witness_valid)


def spec(n=2, method='linear', lower=0, upper=10, claims=None, mean=None):
    obj = {'schema': INPUT_SCHEMA, 'n': n, 'method': method,
           'support': {'lower': lower, 'upper': upper},
           'quantiles': claims or [{'p': '0.5', 'lower': 5, 'upper': 5}]}
    if mean is not None:
        obj['mean'] = {'lower': mean[0], 'upper': mean[1]}
    return obj


def q(p, lo, hi=None):
    return {'p': str(p), 'lower': str(lo), 'upper': str(lo if hi is None else hi)}


def quantile(xs, p, method):
    p = F(p)
    if method == 'inverted_cdf':
        return F(xs[max(0, math.ceil(len(xs) * p) - 1)])
    h = (len(xs) - 1) * p
    lo, hi = math.floor(h), math.ceil(h)
    return (1 - (h - lo)) * xs[lo] + (h - lo) * xs[hi]


class CoreTests(unittest.TestCase):
    def check(self, s, status):
        r = audit(s)
        self.assertEqual(r['status'], status, r)
        self.assertTrue(verify_report(s, r), r)
        return r

    def test_method_is_load_bearing(self):
        s = spec(claims=[q('0.25', 0), q('0.75', 10)])
        self.check(s, 'INFEASIBLE')
        s['method'] = 'inverted_cdf'
        self.check(s, 'FEASIBLE')

    def test_interpolated_median_mean_joint_contradiction(self):
        # Four values in [0,10] with a linear median of9 have sum at least27.
        # Choosing x1=x2=x3=9 and x0=0 attains27, so a mean of5 is impossible.
        s = spec(n=4, claims=[q('0.5', 9)], mean=(5, 5))
        self.check(s, 'INFEASIBLE')
        s['mean'] = {'lower': '27/4', 'upper': '27/4'}
        self.check(s, 'FEASIBLE')

    def test_mean_uses_gap_multiplicities(self):
        s = spec(n=101, claims=[q(0, 0), q(1, 10)], mean=('10/101', '10/101'))
        self.check(s, 'FEASIBLE')
        s['mean'] = {'lower': '9/101', 'upper': '9/101'}
        self.check(s, 'INFEASIBLE')

    def test_trillion_observations_are_not_allocated(self):
        n = 10**12
        s = spec(n=n, claims=[q(0, 0), q(1, 10)], mean=(5, 5))
        r = self.check(s, 'FEASIBLE')
        self.assertLessEqual(r['model']['segments'], 7)
        self.assertEqual(sum(run['count'] for run in r['witness']), n)
        self.assertLessEqual(len(r['witness']), 7)

    def test_singleton(self):
        for method in ('linear', 'inverted_cdf'):
            s = spec(n=1, method=method, claims=[q(0, 3), q('1/2', 3), q(1, 3)], mean=(3, 3))
            self.check(s, 'FEASIBLE')
            s['mean'] = {'lower': 4, 'upper': 4}
            self.check(s, 'INFEASIBLE')

    def test_closed_interval_endpoints(self):
        self.check(spec(n=1, claims=[q('0.5', 4, 5)], mean=(5, 6)), 'FEASIBLE')
        self.check(spec(n=1, claims=[q('0.5', 4, 5)], mean=('5000001/1000000', 6)), 'INFEASIBLE')

    def test_negative_support(self):
        self.check(spec(n=3, lower=-20, upper=-2, claims=[q(0, -20), q('0.5', -10), q(1, -2)], mean=('-32/3', '-32/3')), 'FEASIBLE')

    def test_zero_width_support(self):
        self.check(spec(n=100, lower=-3, upper=-3, claims=[q('0.3', -3)]), 'FEASIBLE')
        self.check(spec(n=100, lower=-3, upper=-3, claims=[q('0.3', -2)]), 'INFEASIBLE')

    def test_small_rational_contradiction_has_no_tolerance(self):
        self.check(spec(n=1, lower=0, upper=0, claims=[q('0.5', '1/' + '1' + '0'*60)]), 'INFEASIBLE')

    def test_duplicate_probabilities_are_not_dropped(self):
        self.check(spec(claims=[q('0.5', 3), q('0.5', 4)]), 'INFEASIBLE')
        self.check(spec(claims=[q('0.5', 3, 5), q('0.5', 4, 6)]), 'FEASIBLE')

    def test_unsorted_claims(self):
        self.check(spec(n=7, claims=[q(1, 10), q(0, 0), q('0.5', 5)]), 'FEASIBLE')

    def test_outside_support_is_contradiction_not_parse_error(self):
        self.check(spec(claims=[q('0.5', -2, -1)]), 'INFEASIBLE')
        self.check(spec(mean=(11, 12)), 'INFEASIBLE')

    def test_model_compression_size_bound(self):
        s = spec(n=1000000, claims=[q(F(i, 17), 0, 10) for i in range(1, 17)])
        model = compile_model(parse_spec(s))
        self.assertLessEqual(len(model.segments), 4*16+3)
        self.assertEqual(sum(count for _, count in model.segments), s['n'])

    def test_400_seeded_sample_and_disjoint_claim_controls(self):
        rng = random.Random(10179)
        for k in range(200):
            n = rng.randint(1, 35)
            values = sorted(F(rng.randint(-30, 40), 3) for _ in range(n))
            method = ('linear', 'inverted_cdf')[k % 2]
            ps = [F(rng.randint(0, 19), 19) for _ in range(rng.randint(1, 7))]
            claims = [q(p, quantile(values, p, method)) for p in ps]
            s = spec(n, method, -10, '40/3', claims, (str(sum(values) / n), str(sum(values) / n)))
            with self.subTest(k=k):
                self.check(s, 'FEASIBLE')
                bad = copy.deepcopy(s)
                old = F(bad['quantiles'][0]['upper'])
                bad['quantiles'].append(q(ps[0], old+1, old+2))
                self.check(bad, 'INFEASIBLE')

    def test_pivot_budget_is_not_a_verdict(self):
        r = audit(spec(), pivot_limit=0)
        self.assertEqual(r['status'], 'UNRESOLVED')
        self.assertNotIn('witness', r)
        self.assertNotIn('contradiction', r)
        self.assertFalse(verify_report(spec(), r))

    def test_rational_budget_is_not_a_verdict(self):
        r = audit(spec(), bit_limit=1)
        self.assertEqual(r['status'], 'UNRESOLVED')
        self.assertFalse(verify_report(spec(), r))

    def test_invalid_input_table(self):
        changes = [('schema', 'other'), ('n', True), ('n', 2.0), ('n', 0), ('n', -1), ('n', 10**12+1),
                   ('method', 'nearest'), ('method', []), ('support', None), ('mean', None),
                   ('support', {'lower': 2, 'upper': 1}), ('support', {'lower': 0, 'upper': '1e999'}),
                   ('support', {'lower': 0, 'upper': float('inf')}), ('quantiles', []),
                   ('quantiles', [q('0.5', 0)]*17), ('quantiles', [q('-0.1', 0)]),
                   ('quantiles', [q('1.01', 0)]), ('quantiles', [q('0.5', 3, 2)]),
                   ('quantiles', [{'p': '0.5', 'lower': 1, 'upper': 2, 'extra': 3}])]
        for key, value in changes:
            s = spec();s[key] = value
            with self.subTest(key=key, value=value):
                self.assertEqual(audit(s)['status'], 'INVALID')
        for value in [None, [], 42, {}, True, dict(spec(), extra=1)]:
            self.assertEqual(audit(value)['status'], 'INVALID')

    def test_number_admission(self):
        for value in [True, 1.0, 'NaN', 'Infinity', '1e-9', '+1', '.5', '01', '1/0', '1/01', '１', '1'*101, 2**257]:
            with self.subTest(value=value), self.assertRaises(Invalid):
                number(value)
        self.assertEqual(number('-0.50'), F(-1, 2))
        self.assertEqual(number('1/3'), F(1, 3))

    def test_fingerprint_normalizes_numbers_but_not_claims(self):
        a = spec();b = spec();b['quantiles'][0]['p'] = '1/2'
        self.assertEqual(digest(parse_spec(a)), digest(parse_spec(b)))
        b['quantiles'][0]['upper'] = 6
        self.assertNotEqual(digest(parse_spec(a)), digest(parse_spec(b)))

    def test_report_bound_to_declared_inputs(self):
        a = spec();r = audit(a);b = spec();b['quantiles'][0]['upper'] = 6
        self.assertFalse(verify_report(b, r))

    def test_false_or_metadata_only_verdict_is_not_a_certificate(self):
        for r in [None, {}, {'status': 'FEASIBLE'}, {'schema': 'other'}, {'status': 'INFEASIBLE'}]:
            self.assertFalse(verify_report(spec(), r))
        r = audit(spec());r.pop('witness')
        self.assertFalse(verify_report(spec(), r))

    def test_witness_count_sort_bounds_and_mean(self):
        s = spec(n=2, claims=[q(0, 0), q(1, 10)])
        parsed = parse_spec(s)
        for runs in [[{'count': 1, 'value': '0'}, {'count': 2, 'value': '10'}],
                     [{'count': True, 'value': '0'}, {'count': 1, 'value': '10'}],
                     [{'count': 1, 'value': '10'}, {'count': 1, 'value': '0'}],
                     [{'count': 1, 'value': '-1'}, {'count': 1, 'value': '10'}]]:
            self.assertFalse(witness_valid(parsed, runs))
        r = audit(s);r['witness'][0]['value'] = '1'
        self.assertFalse(verify_report(s, r))

    def test_forged_negative_dual_weights_are_rejected(self):
        s = spec(n=1)
        r = audit(s);r.pop('witness');r['status'] = 'INFEASIBLE'
        r['contradiction'] = [{'constraint': 'support:upper', 'weight': '-1'},
                              {'constraint': 'quantile:0:lower', 'weight': '-1'}]
        self.assertFalse(verify_report(s, r))

    def test_forged_negative_combination_coefficient_is_rejected(self):
        s = spec(n=1)
        r = audit(s);r.pop('witness');r['status'] = 'INFEASIBLE'
        r['contradiction'] = [{'constraint': 'quantile:0:lower', 'weight': '1'}]
        self.assertFalse(verify_report(s, r))

    def test_unknown_duplicate_or_bad_weight_is_rejected(self):
        s = spec(claims=[q('0.25', 0), q('0.75', 10)])
        good = audit(s)
        for weights in [[], [{'constraint': 'unknown', 'weight': '1'}],
                        [{'constraint': [], 'weight': '1'}],
                        [good['contradiction'][0]]*2,
                        [{'constraint': 'support:upper', 'weight': True}],
                        [{'constraint': 'support:upper', 'weight': '1e100000000'}]]:
            bad = copy.deepcopy(good);bad['contradiction'] = weights
            self.assertFalse(verify_report(s, bad))

    def test_report_scope_and_metadata_are_checked(self):
        s = spec(n=1);good = audit(s)
        for field, value in [('model', {'n': True, 'segments': 1, 'constraints': 3}),
                             ('pivots', True), ('pivots', -1), ('scope', 'proof of real data'),
                             ('schema', 'other'), ('extra', 'not admitted')]:
            bad = copy.deepcopy(good);bad[field] = value
            self.assertFalse(verify_report(s, bad))

    def test_deterministic_reports(self):
        s = spec(n=29, claims=[q('0.1', 0, 1), q('0.9', 8, 9)], mean=(4, 6))
        self.assertEqual(audit(s), audit(s))


    def test_invalid_execution_budgets_are_invalid_not_unbounded(self):
        for kwargs in [{'pivot_limit': True}, {'pivot_limit': -1}, {'pivot_limit': '2'},
                       {'bit_limit': False}, {'bit_limit': 0}, {'bit_limit': float('nan')},
                       {'bit_limit': '4096'}]:
            with self.subTest(kwargs=kwargs):
                self.assertEqual(audit(spec(), **kwargs)['status'], 'INVALID')


    def test_certificate_mean_arithmetic_budget(self):
        s = spec(n=3, lower=0, upper=1, claims=[q(0, 0, 1)], mean=(0, 1))
        r = audit(s)
        values = sorted(F(1, (1 << 4000) - k) for k in (1, 3, 5))
        self.assertGreater(sum(values).denominator.bit_length(), 8192)
        r['witness'] = [{'count': 1, 'value': str(v)} for v in values]
        self.assertFalse(verify_report(s, r))
        # Without a mean no sum of unrelated large denominators is necessary.
        s.pop('mean');r = audit(s)
        r['witness'] = [{'count': 1, 'value': str(v)} for v in values]
        self.assertTrue(verify_report(s, r))


if __name__ == '__main__':
    unittest.main()
