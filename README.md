# quantilewitness

Check whether a finite, bounded, continuous-valued sample could satisfy declared percentile intervals and an optional mean interval, without collecting the observations.

A successful check returns a small synthetic sample witness. A contradiction returns exact rational weights that can be checked without rerunning the optimizer. Both are certificates about the supplied mathematical claims, not about the existence or truth of anyone's data.

Experimental software. Treat its certificates as statements about the declared mathematical claims, never as evidence that a dataset existed.

## Why use it?

A dashboard export or benchmark summary can combine a count, percentiles and a mean computed by different code paths. Check their joint compatibility before treating the summary as one sample. Individually plausible values can be jointly impossible. The declared percentile method matters, especially for small samples.

This is not a raw-data checker, a fraud detector, a distribution estimator or a test of statistical significance. Here continuous-valued means observations may be real numbers, with ties allowed; no probability distribution is assumed. FEASIBLE does not establish that observations ever existed. INFEASIBLE can indicate a wrong method, support, count, interval, aggregation assumption or transcription, not dishonest reporting.

## Run locally

Python 3.10 or later. Runtime uses the standard library only. No network access, solver service or runtime dependency is needed.

```sh
python -m pip install .
python -m quantilewitness check examples/method-sensitive.json > report.json
python -m quantilewitness verify examples/method-sensitive.json report.json
```

`check` returns 1 for this example, so do not use `&&` before verification when checking a contradiction. The CLI reads explicit paths or one `-` stdin input. It writes JSON to stdout and does not create or overwrite result files itself. Shell redirection above is intentional.

```python
from quantilewitness import audit, verify_report

report = audit(claim_document)
assert report['status'] in {'FEASIBLE', 'INFEASIBLE', 'INVALID', 'UNRESOLVED'}
if report['status'] in {'FEASIBLE', 'INFEASIBLE'}:
    assert verify_report(claim_document, report)
```

## Input

```json
{
  "schema": "quantilewitness/input-v1",
  "n": 2,
  "method": "linear",
  "support": {"lower": 0, "upper": 10},
  "quantiles": [
    {"p": "0.25", "lower": 0, "upper": 0},
    {"p": "0.75", "lower": 10, "upper": 10}
  ]
}
```

These claims are INFEASIBLE under `linear`. Change only the method to `inverted_cdf`, and `[0, 10]` is a witness. The method is required, never inferred.

- `n`: JSON integer from 1 through 1,000,000,000,000. Booleans are not integers here.
- `support`: finite lower and upper bounds that apply to every observation, inclusively. These are bounds, not claims that the sample attains its extrema.
- `quantiles`: 1 through 16 claims. `p` is a probability in `[0, 1]`, not a percentage in `[0, 100]`. Input order is preserved and duplicate probabilities are allowed.
- Optional `mean`: an object with `lower` and `upper`, describing the arithmetic mean of these same `n` observations.
- All claim intervals are closed. A reported value of `9.0` does not automatically become a rounding interval. The caller must supply any intended interval explicitly.
- Numbers are JSON integers or ASCII fixed-decimal/rational strings, such as `"-0.125"` or `"7/9"`. JSON floats, exponents, NaN, Infinity, whitespace inside numeric strings, leading plus signs and zero denominators are refused. Ordinary numbers are limited to 100 characters and 256 numerator/denominator bits.
- Unknown fields, duplicate JSON keys, reversed intervals and unsupported methods are refused. The CLI accepts at most 64 KiB of claims and 1 MiB of saved report input.

### Declared methods

For sorted observations `x[0] <= ... <= x[n-1]`:

- `linear`, Hyndman-Fan type 7: let `h = (n-1)p`, `j = floor(h)`, `g = h-j`. The quantile is `x[j]` when `g=0`, otherwise `(1-g)x[j] + g*x[j+1]`.
- `inverted_cdf`, Hyndman-Fan type 1: use `x[max(0, ceil(np)-1)]`, clipped to the last observation. In particular, `p=0` is the minimum and `p=1` is the maximum.

These names and formulas follow the conventions documented by [NumPy quantile](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html). Production calculations here use exact rational arithmetic, not NumPy.

## Reports and exit codes

| Check status | Exit | Meaning |
| --- | ---: | --- |
| FEASIBLE | 0 | A verified synthetic sample satisfies every supplied claim. |
| INFEASIBLE | 1 | A verified exact contradiction rules out such a sample. |
| INVALID | 2 | The input or API execution-budget settings are outside the supported limits. |
| UNRESOLVED | 3 | The bounded calculation could not produce a verified answer. This is neither feasible nor infeasible. |

An output failure also returns 2. Argument errors use argparse's exit 2. `verify` returns 0 for an accepted certificate, 1 for a rejected certificate, or 2 for malformed/unreadable CLI input or output failure. It does not certify INVALID or UNRESOLVED reports.

A FEASIBLE `witness` contains ordered `{count, value}` runs, with exact rational string values. Expanding those runs gives a synthetic sample, not recovered observations. The verifier checks count, order, support, declared quantiles and mean directly from the runs, without calling the solver. Witness claim arithmetic separately implements the quantile formulas; model metadata validation still reconstructs the compressed model.

An INFEASIBLE `contradiction` lists named constraints and nonnegative rational weights. The model has `Az <= b` and `z >= 0`; a certificate `y` satisfies `y >= 0`, `yA >= 0`, and `yb < 0`. Multiplying and adding the inequalities would require a nonnegative quantity to be negative. Verification reconstructs this model from the supplied claims and checks the exact arithmetic, but does not run simplex.

`input_sha256` binds the report to canonical normalized claims, including claim order. It is not a hash of the original file bytes, an author signature or an authenticity check. Equivalent rational spellings normalize to the same value. The reported pivot count is descriptive metadata, not an independently reproduced execution trace.

Certificate numbers are limited to 2,500 characters and 4,096 numerator/denominator bits. Verification also stops an accumulating rational sum above 8,192 numerator/denominator bits, rather than multiplying arbitrarily many unrelated denominators. `verify` can therefore reject a mathematically valid certificate that exceeds its supported arithmetic budget. Rejection does not establish the opposite feasibility result. The optimizer checks its own certificates under these same rules and returns UNRESOLVED if they cannot be accepted.

## Why a trillion observations do not require a trillion variables

Each quantile touches one or two order-statistic ranks. Keep those ranks and the two endpoint ranks. In any gap between kept ranks, replace all observations by their mean. This preserves order, the kept observations, every quantile and the overall sum. Conversely, expanding a constant gap gives a full sample with the same properties.

There are at most `4q + 3` constant segments for `q` claims, at most 67 here. The mean constraint weights each segment by its actual run count divided by `n`, not by the number of segments. This compression applies to continuous-valued observations only. It does not establish that an integer-valued or quantized sample exists.

The implementation shifts values by the support lower bound and runs an exact rational phase-I simplex with Bland choices and tracked row combinations. A default 2,000-pivot budget and a 4,096-bit post-pivot arithmetic threshold stop expensive runs with UNRESOLVED. Python callers can set `audit(..., pivot_limit=2000, bit_limit=4096)` explicitly; the first must be a nonnegative integer and the second a positive integer, not booleans. Zero pivots can still certify inputs whose initial feasible point already works. These are algorithmic limits, not a wall-clock or process-memory sandbox. A large count alone does not enlarge the matrix, but difficult rational claims can still take time.

## Scope and limits

No weights, missing observations, group pooling, variance claims, unknown-method inference, integer lattices, automatic rounding interpretation, distribution fitting, confidence intervals or provenance inference. Inputs must describe the same finite sample. Open interval endpoints are unsupported. A mathematical witness can be fractional even when a real application measures counts.

The verifier is deliberately smaller than the optimizer, but it shares schema parsing and, for contradiction certificates, model construction. It is not a separate formal proof system. Validate the assumptions and certificate for your use before relying on a result.

Related work includes [qonlyreg](https://github.com/masaiwasawa/qonlyreg) for quantile-only identification and regression. This project does not claim new quantile mathematics, new linear programming theory, exhaustive novelty, adoption or saved analyst time.

## Development

```sh
python -m unittest discover -s tests -v
```

Ordinary tests need only the standard library. Two separate development oracles check the compressed model:

```sh
python exact_oracle_check.py --cases 128
python oracle_check.py --cases 2000
python mutation_check.py
```

`exact_oracle_check.py` uses only the standard library and enumerates vertices of the full, uncompressed one-variable-per-observation polyhedron for samples of one through four observations. Its fixtures include exact `1/10^30` claim shifts, where a floating-point tolerance is not sufficient. It neither calls the production model builder nor uses simplex.

`oracle_check.py` requires NumPy and SciPy. It constructs one variable per observation for small samples and compares HiGHS feasibility with the compressed exact solver. Floating-point oracle agreement is differential evidence on those fixtures, not an exact proof at arbitrary near-boundary values. Report-tampering controls are tested separately. Neither development oracle writes files by default; `--output PATH` explicitly saves its fixtures and results.

`mutation_check.py` runs nine syntax-valid semantic and proof-control changes in disposable copies. A mutant must produce an assertion failure, not an exception-only failure, to count as caught. It uses the current Python interpreter and the standard-library tests. `--output-dir PATH` optionally saves its logs and manifest.
