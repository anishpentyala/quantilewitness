import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from quantilewitness import audit
from quantilewitness.core import Invalid
from quantilewitness.cli import INPUT_BYTES, PROOF_BYTES, _emit, _read, main


def spec():
    return {'schema': 'quantilewitness/input-v1', 'n': 2, 'method': 'linear',
            'support': {'lower': 0, 'upper': 10},
            'quantiles': [{'p': '0.25', 'lower': 0, 'upper': 0},
                          {'p': '0.75', 'lower': 10, 'upper': 10}]}


class ChunkedInput(io.IOBase):
    def __init__(self, content, size, ending='eof'):
        super().__init__()
        self.content, self.size, self.ending = content, size, ending
        self.position = 0
        self.requests = []
        self.end_reads = 0

    def read(self, size=-1):
        if self.closed:
            raise ValueError('closed input')
        self.requests.append(size)
        if self.position < len(self.content):
            count = len(self.content) - self.position if size < 0 else min(self.size, size, len(self.content) - self.position)
            value = self.content[self.position:self.position + count]
            self.position += count
            return value
        self.end_reads += 1
        if self.ending == 'none':
            return None
        if self.ending == 'zero':
            return 0
        if self.ending == 'false':
            return False
        if self.ending == 'block':
            raise BlockingIOError('input would block')
        if self.ending == 'error':
            raise OSError('input read failed')
        if self.ending == 'value':
            raise ValueError('input closed during read')
        return self.content[:0]


class CliTests(unittest.TestCase):
    def invoke(self, argv, content=None, stdout=None):
        stream = io.StringIO() if stdout is None else stdout
        with patch('sys.stdin', io.StringIO(content or '')), patch('sys.stdout', stream):
            code = main(argv)
        return code, stream.getvalue() if not stream.closed else None

    def test_check_stdin_and_status_codes(self):
        code, text = self.invoke(['check', '-'], json.dumps(spec()))
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(text)['status'], 'INFEASIBLE')
        s = spec();s['method'] = 'inverted_cdf'
        code, text = self.invoke(['check', '-'], json.dumps(s))
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(text)['status'], 'FEASIBLE')

    def test_check_file_has_no_input_mutation_or_added_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'claims.json'
            text = json.dumps(spec());path.write_text(text)
            code, report = self.invoke(['check', str(path)])
            self.assertEqual(code, 1)
            self.assertEqual(path.read_text(), text)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ['claims.json'])

    def test_verify_saved_contradiction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'claims.json';path.write_text(json.dumps(spec()))
            result = audit(spec())
            code, text = self.invoke(['verify', str(path), '-'], json.dumps(result))
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(text)['certificate_valid'])
            result['contradiction'][0]['weight'] = '0'
            code, text = self.invoke(['verify', str(path), '-'], json.dumps(result))
            self.assertEqual(code, 1)
            self.assertFalse(json.loads(text)['certificate_valid'])

    def test_verify_saved_witness(self):
        s = spec();s['method'] = 'inverted_cdf'
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.json', Path(directory) / 'b.json'
            a.write_text(json.dumps(s));b.write_text(json.dumps(audit(s)))
            code, text = self.invoke(['verify', str(a), str(b)])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(text)['certificate_valid'])

    def test_two_stdin_inputs_refused(self):
        code, text = self.invoke(['verify', '-', '-'])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(text)['status'], 'INVALID')

    def test_json_duplicates_nonfinite_bad_utf8_and_missing_file(self):
        for content in ['{"n":2,"n":3}', '{"x":NaN}', '{"x":Infinity}', '{', '[]']:
            with self.subTest(content=content):
                code, text = self.invoke(['check', '-'], content)
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(text)['status'], 'INVALID')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json';path.write_bytes(b'\xff\xfe')
            for name in [str(path), str(path) + '.missing', directory]:
                code, text = self.invoke(['check', name])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(text)['status'], 'INVALID')

    def test_input_byte_cap_and_deep_json_are_invalid(self):
        for content in [' ' * (65536 + 1), '"' + 'é' * 40000 + '"', '[' * 2000 + ']' * 2000]:
            code, text = self.invoke(['check', '-'], content)
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(text)['status'], 'INVALID')

    def test_schema_and_numeric_float_refusals(self):
        for mutation in ['unknown_field', 'float', 'bool']:
            s = spec()
            if mutation == 'unknown_field':
                s['weights'] = []
            elif mutation == 'float':
                s['quantiles'][0]['upper'] = 0.0
            else:
                s['n'] = True
            self.assertEqual(self.invoke(['check', '-'], json.dumps(s))[0], 2)

    def test_closed_stdout_main_does_not_replace_stream(self):
        closed = io.StringIO();closed.close()
        with patch('sys.stdin', io.StringIO(json.dumps(spec()))), patch('sys.stdout', closed):
            self.assertEqual(main(['check', '-']), 2)
            self.assertIs(sys.stdout, closed)

    def test_argparse_errors_survive_closed_stdout(self):
        closed = io.StringIO();closed.close()
        with patch('sys.stdout', closed), patch('sys.stderr', io.StringIO()):
            with self.assertRaises(SystemExit) as exc:
                main(['not-a-command'])
            self.assertEqual(exc.exception.code, 2)

    def test_flush_failure_returns_two(self):
        class Broken(io.StringIO):
            def flush(self):
                raise OSError('full output')
        self.assertEqual(self.invoke(['check', '-'], json.dumps(spec()), stdout=Broken())[0], 2)

    def test_cli_module_and_verifier(self):
        p = subprocess.run([sys.executable, '-m', 'quantilewitness', 'check', '-'],
                           input=json.dumps(spec()), text=True, capture_output=True)
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertEqual(json.loads(p.stdout)['status'], 'INFEASIBLE')
        self.assertEqual(p.stderr, '')

    @unittest.skipUnless(os.path.exists('/dev/full'), 'Linux /dev/full only')
    def test_module_output_flush_failure_is_two_not_shutdown_120(self):
        with open('/dev/full', 'wb') as output:
            p = subprocess.run([sys.executable, '-m', 'quantilewitness', 'check', '-'],
                               input=json.dumps(spec()).encode(), stdout=output, stderr=subprocess.PIPE)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertNotIn(b'Traceback', p.stderr)

    def test_main_does_not_close_caller_output(self):
        output = io.StringIO()
        code, _ = self.invoke(['check', '-'], '{}', stdout=output)
        self.assertEqual(code, 2)
        self.assertFalse(output.closed)


    def test_help_root_and_both_subcommands(self):
        for argv in [['--help'], ['check', '--help'], ['verify', '--help']]:
            p = subprocess.run([sys.executable, '-m', 'quantilewitness'] + argv,
                               text=True, capture_output=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn('usage:', p.stdout)
            self.assertEqual(p.stderr, '')

    def test_help_closed_stdout_has_no_exception(self):
        for argv in [['--help'], ['check', '--help'], ['verify', '--help']]:
            stream = io.StringIO();stream.close()
            with patch('sys.stdout', stream):
                self.assertEqual(main(argv), 2)

    @unittest.skipUnless(os.path.exists('/dev/full'), 'Linux /dev/full only')
    def test_help_full_stdout_is_two_not_shutdown_120(self):
        for argv in [['--help'], ['check', '--help'], ['verify', '--help']]:
            with open('/dev/full', 'wb') as output:
                p = subprocess.run([sys.executable, '-m', 'quantilewitness'] + argv,
                                   stdout=output, stderr=subprocess.PIPE)
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertNotIn(b'Traceback', p.stderr)

    @unittest.skipUnless(os.path.exists('/dev/full'), 'Linux /dev/full only')
    def test_argparse_full_stderr_is_two_not_shutdown_120(self):
        with open('/dev/full', 'wb') as error:
            p = subprocess.run([sys.executable, '-m', 'quantilewitness', 'unknown'],
                               stdout=subprocess.PIPE, stderr=error)
        self.assertEqual(p.returncode, 2)
        self.assertEqual(p.stdout, b'')

    def test_proof_input_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'claims.json';path.write_text(json.dumps(spec()))
            code, text = self.invoke(['verify', str(path), '-'], ' ' * (1024 * 1024 + 1))
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(text)['status'], 'INVALID')

    def test_closed_stdin_is_invalid(self):
        stream = io.StringIO();stream.close()
        output = io.StringIO()
        with patch('sys.stdin', stream), patch('sys.stdout', output):
            self.assertEqual(main(['check', '-']), 2)
        self.assertEqual(json.loads(output.getvalue())['status'], 'INVALID')


    def test_check_positive_short_writes_finish_complete_reports(self):
        class Short(io.StringIO):
            def __init__(self, size):
                super().__init__();self.size = size
            def write(self, text):
                return super().write(text[:self.size])
        for size in (1, 7):
            for method, expected in [('linear', 1), ('inverted_cdf', 0)]:
                s = spec();s['method'] = method
                output = Short(size)
                code, text = self.invoke(['check', '-'], json.dumps(s), stdout=output)
                self.assertEqual(code, expected)
                self.assertEqual(json.loads(text)['status'], 'INFEASIBLE' if expected else 'FEASIBLE')
                self.assertTrue(text.endswith('\n'))
                self.assertFalse(output.closed)

    def test_verify_positive_short_writes_finish_complete_reports(self):
        class Short(io.StringIO):
            def __init__(self, size):
                super().__init__();self.size = size
            def write(self, text):
                return super().write(text[:self.size])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'claims.json';path.write_text(json.dumps(spec()))
            for size in (1, 7):
                for valid in (True, False):
                    report = audit(spec())
                    if not valid:
                        report['scope'] = 'changed'
                    output = Short(size)
                    code, text = self.invoke(['verify', str(path), '-'], json.dumps(report), stdout=output)
                    self.assertEqual(code, 0 if valid else 1)
                    self.assertIs(json.loads(text)['certificate_valid'], valid)
                    self.assertTrue(text.endswith('\n'))
                    self.assertFalse(output.closed)

    def test_zero_progress_check_and_verify_are_output_failures(self):
        class Zero(io.StringIO):
            def write(self, text):
                return 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'claims.json';path.write_text(json.dumps(spec()))
            calls = [(['check', '-'], json.dumps(spec())),
                     (['verify', str(path), '-'], json.dumps(audit(spec())))]
            for argv, content in calls:
                output = Zero()
                code, text = self.invoke(argv, content, stdout=output)
                self.assertEqual(code, 2)
                self.assertEqual(text, '')
                self.assertFalse(output.closed)

    def test_partial_then_zero_is_failure_not_normal_verdict(self):
        class Stops(io.StringIO):
            def write(self, text):
                return super().write(text[:7]) if not self.tell() else 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'claims.json';path.write_text(json.dumps(spec()))
            s = spec();s['method'] = 'inverted_cdf'
            calls = [(['check', '-'], json.dumps(spec())), (['check', '-'], json.dumps(s)),
                     (['verify', str(path), '-'], json.dumps(audit(spec())))]
            for argv, content in calls:
                output = Stops()
                code, text = self.invoke(argv, content, stdout=output)
                self.assertEqual(code, 2)
                self.assertEqual(len(text), 7)
                self.assertFalse(output.closed)

    def test_invalid_write_counts_fail_check_and_verify(self):
        class InvalidCount(io.StringIO):
            def __init__(self, count):
                super().__init__();self.count = count
            def write(self, text):
                return len(text) + 1 if self.count == 'oversized' else self.count
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'claims.json';path.write_text(json.dumps(spec()))
            calls = [(['check', '-'], json.dumps(spec())),
                     (['verify', str(path), '-'], json.dumps(audit(spec())))]
            for count in (None, -1, True, 'bad', 'oversized'):
                for argv, content in calls:
                    output = InvalidCount(count)
                    self.assertEqual(self.invoke(argv, content, stdout=output)[0], 2)
                    self.assertFalse(output.closed)

    def test_write_error_after_partial_progress_is_failure(self):
        class Broken(io.StringIO):
            def write(self, text):
                if self.tell():
                    raise OSError('write failed')
                return super().write(text[:7])
        output = Broken()
        code, text = self.invoke(['check', '-'], json.dumps(spec()), stdout=output)
        self.assertEqual(code, 2)
        self.assertEqual(len(text), 7)
        self.assertFalse(output.closed)

    def test_short_write_flush_error_is_failure(self):
        class Broken(io.StringIO):
            def write(self, text):
                return super().write(text[:7])
            def flush(self):
                raise OSError('flush failed')
        output = Broken()
        code, text = self.invoke(['check', '-'], json.dumps(spec()), stdout=output)
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(text)['status'], 'INFEASIBLE')
        self.assertFalse(output.closed)


    def test_help_positive_short_writes_match_complete_output(self):
        class Short(io.StringIO):
            def __init__(self, size):
                super().__init__();self.size = size
            def write(self, text):
                return super().write(text[:self.size])
        for argv in [['--help'], ['check', '--help'], ['verify', '--help']]:
            expected = subprocess.run([sys.executable, '-m', 'quantilewitness'] + argv,
                                      capture_output=True, text=True)
            self.assertEqual(expected.returncode, 0)
            for size in (1, 7):
                output = Short(size)
                with patch('sys.stdout', output), self.assertRaises(SystemExit) as exc:
                    main(argv)
                self.assertEqual(exc.exception.code, 0)
                self.assertEqual(output.getvalue(), expected.stdout)
                self.assertFalse(output.closed)

    def test_help_zero_progress_is_output_failure(self):
        class Zero(io.StringIO):
            def write(self, text):
                return 0
        for argv in [['--help'], ['check', '--help'], ['verify', '--help']]:
            output = Zero()
            with patch('sys.stdout', output):
                self.assertEqual(main(argv), 2)
            self.assertFalse(output.closed)

    def test_argparse_short_stderr_completes_or_fails_with_zero_progress(self):
        class Short(io.StringIO):
            def __init__(self, size):
                super().__init__();self.size = size
            def write(self, text):
                return super().write(text[:self.size])
        expected = subprocess.run([sys.executable, '-m', 'quantilewitness', 'unknown'],
                                  capture_output=True, text=True)
        self.assertEqual(expected.returncode, 2)
        for size in (1, 7):
            error = Short(size)
            with patch('sys.stderr', error), self.assertRaises(SystemExit) as exc:
                main(['unknown'])
            self.assertEqual(exc.exception.code, 2)
            self.assertEqual(error.getvalue(), expected.stderr)
            self.assertFalse(error.closed)
        error = Short(0)
        with patch('sys.stderr', error):
            self.assertEqual(main(['unknown']), 2)
        self.assertFalse(error.closed)


    @staticmethod
    def _main_code(argv):
        try:
            return main(argv)
        except SystemExit as exc:
            return exc.code

    @staticmethod
    def _missing_stream_cases(directory):
        claim = Path(directory) / 'claim.json'
        good = Path(directory) / 'good.json'
        bad = Path(directory) / 'bad.json'
        claim.write_text(json.dumps(spec()))
        proof = audit(spec())
        good.write_text(json.dumps(proof))
        proof['scope'] = 'changed'
        bad.write_text(json.dumps(proof))
        feasible = spec();feasible['method'] = 'inverted_cdf'
        return [(['check', '-'], json.dumps(feasible), 0),
                (['check', '-'], json.dumps(spec()), 1),
                (['check', '-'], '{}', 2),
                (['verify', str(claim), str(good)], '', 0),
                (['verify', str(claim), str(bad)], '', 1),
                (['--help'], '', 0), (['check', '--help'], '', 0),
                (['verify', '--help'], '', 0),
                (['unknown'], '', 2), (['check'], '', 2)]

    def test_emission_rejects_missing_stdout(self):
        with patch('sys.stdout', None):
            self.assertFalse(_emit({'status': 'INVALID'}))
            self.assertIsNone(sys.stdout)

    def test_missing_stdout_is_output_failure_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            for argv, content, _ in self._missing_stream_cases(directory):
                with self.subTest(argv=argv, content=content):
                    error = io.StringIO()
                    with patch('sys.stdin', io.StringIO(content)), patch('sys.stdout', None), patch('sys.stderr', error):
                        self.assertEqual(self._main_code(argv), 2)
                        self.assertIsNone(sys.stdout)
                        self.assertIs(sys.stderr, error)
                    self.assertEqual(error.getvalue(), '')
                    self.assertFalse(error.closed)

    def test_missing_optional_stderr_preserves_report_help_and_error_status(self):
        with tempfile.TemporaryDirectory() as directory:
            for argv, content, expected in self._missing_stream_cases(directory):
                with self.subTest(argv=argv, content=content):
                    golden = io.StringIO()
                    with patch('sys.stdin', io.StringIO(content)), patch('sys.stdout', golden), patch('sys.stderr', io.StringIO()):
                        self.assertEqual(self._main_code(argv), expected)
                    output = io.StringIO()
                    with patch('sys.stdin', io.StringIO(content)), patch('sys.stdout', output), patch('sys.stderr', None):
                        self.assertEqual(self._main_code(argv), expected)
                        self.assertIsNone(sys.stderr)
                        self.assertIs(sys.stdout, output)
                    self.assertEqual(output.getvalue(), golden.getvalue())
                    self.assertFalse(output.closed)

    def test_both_missing_output_streams_are_output_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            for argv, content, _ in self._missing_stream_cases(directory):
                with self.subTest(argv=argv, content=content):
                    with patch('sys.stdin', io.StringIO(content)), patch('sys.stdout', None), patch('sys.stderr', None):
                        self.assertEqual(self._main_code(argv), 2)
                        self.assertIsNone(sys.stdout)
                        self.assertIsNone(sys.stderr)

    def test_process_missing_output_streams_have_deliberate_exit_codes(self):
        program = ('import sys\nfrom quantilewitness.cli import run\n'
                   'for name in sys.argv[1].split(","): setattr(sys,name,None)\n'
                   'sys.argv=["quantilewitness"]+sys.argv[2:]\n'
                   'raise SystemExit(run())\n')
        with tempfile.TemporaryDirectory() as directory:
            for argv, content, expected in self._missing_stream_cases(directory):
                golden = subprocess.run([sys.executable, '-m', 'quantilewitness'] + argv,
                                        input=content, text=True, capture_output=True)
                self.assertEqual(golden.returncode, expected, golden.stderr)
                for mask in ('stdout', 'stderr', 'stdout,stderr'):
                    with self.subTest(argv=argv, content=content, mask=mask):
                        result = subprocess.run([sys.executable, '-c', program, mask] + argv,
                                                input=content, text=True, capture_output=True)
                        self.assertEqual(result.returncode, expected if mask == 'stderr' else 2, result.stderr)
                        self.assertEqual(result.stdout, golden.stdout if mask == 'stderr' else '')
                        self.assertEqual(result.stderr, '')

    def test_missing_stdin_is_invalid_only_when_input_required(self):
        with tempfile.TemporaryDirectory() as directory:
            claim = Path(directory) / 'claim.json'
            report = Path(directory) / 'report.json'
            claim.write_text(json.dumps(spec()))
            report.write_text(json.dumps(audit(spec())))
            for argv, expected in [(['check', '-'], 2), (['verify', str(claim), '-'], 2),
                                   (['verify', str(claim), str(report)], 0)]:
                with self.subTest(argv=argv):
                    output = io.StringIO()
                    with patch('sys.stdin', None), patch('sys.stdout', output):
                        self.assertEqual(self._main_code(argv), expected)
                        self.assertIsNone(sys.stdin)
                    value = json.loads(output.getvalue())
                    self.assertEqual(value.get('status'), 'INVALID' if expected else None)
                    self.assertFalse(output.closed)

    def test_process_missing_stdin_is_invalid_only_when_input_required(self):
        program = ('import sys\nfrom quantilewitness.cli import run\n'
                   'sys.stdin=None\nsys.argv=["quantilewitness"]+sys.argv[1:]\n'
                   'raise SystemExit(run())\n')
        with tempfile.TemporaryDirectory() as directory:
            claim = Path(directory) / 'claim.json'
            report = Path(directory) / 'report.json'
            claim.write_text(json.dumps(spec()))
            report.write_text(json.dumps(audit(spec())))
            for argv, expected in [(['check', '-'], 2), (['verify', str(claim), '-'], 2),
                                   (['verify', str(claim), str(report)], 0)]:
                with self.subTest(argv=argv):
                    result = subprocess.run([sys.executable, '-c', program] + argv, text=True, capture_output=True)
                    self.assertEqual(result.returncode, expected, result.stderr)
                    self.assertEqual(json.loads(result.stdout).get('status'), 'INVALID' if expected else None)
                    self.assertEqual(result.stderr, '')


    def _chunked_call(self, argv, incoming):
        output = io.StringIO()
        with patch('sys.stdin', incoming), patch('sys.stdout', output):
            code = main(argv)
            self.assertIs(sys.stdin, incoming)
        self.assertFalse(incoming.closed)
        self.assertFalse(output.closed)
        return code, json.loads(output.getvalue())

    def test_chunked_check_consumes_the_complete_document(self):
        for method, expected in [('linear', 1), ('inverted_cdf', 0)]:
            s = spec();s['method'] = method
            for binary in (False, True):
                content = json.dumps(s)
                if binary:
                    content = content.encode()
                for size in (1, 7, 64):
                    with self.subTest(method=method, binary=binary, size=size):
                        incoming = ChunkedInput(content, size)
                        code, value = self._chunked_call(['check', '-'], incoming)
                        self.assertEqual(code, expected)
                        self.assertEqual(value, audit(s))
                        self.assertEqual(incoming.position, len(content))
                        self.assertEqual(incoming.end_reads, 1)

    def test_chunked_verify_consumes_the_complete_document(self):
        with tempfile.TemporaryDirectory() as directory:
            claim = Path(directory) / 'claim.json'
            for method in ('linear', 'inverted_cdf'):
                s = spec();s['method'] = method
                claim.write_text(json.dumps(s))
                for valid in (True, False):
                    proof = audit(s)
                    if not valid:
                        proof['scope'] = 'changed'
                    for binary in (False, True):
                        content = json.dumps(proof)
                        if binary:
                            content = content.encode()
                        for size in (1, 7):
                            with self.subTest(method=method, valid=valid, binary=binary, size=size):
                                incoming = ChunkedInput(content, size)
                                code, value = self._chunked_call(['verify', str(claim), '-'], incoming)
                                self.assertEqual(code, 0 if valid else 1)
                                self.assertIs(value['certificate_valid'], valid)
                                self.assertEqual(incoming.position, len(content))
                                self.assertEqual(incoming.end_reads, 1)

    def test_chunked_check_and_verify_reject_data_after_valid_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            claim = Path(directory) / 'claim.json';claim.write_text(json.dumps(spec()))
            for argv, document in [(['check', '-'], spec()),
                                   (['verify', str(claim), '-'], audit(spec()))]:
                prefix = json.dumps(document) + '\n'
                for binary in (False, True):
                    content = prefix + 'INVALID TRAILING INPUT\n'
                    if binary:
                        content = content.encode()
                    with self.subTest(command=argv[0], binary=binary):
                        incoming = ChunkedInput(content, len(prefix))
                        code, value = self._chunked_call(argv, incoming)
                        self.assertEqual(code, 2)
                        self.assertEqual(value['status'], 'INVALID')
                        self.assertEqual(incoming.position, len(content))
                        self.assertEqual(incoming.end_reads, 1)

    def test_chunked_check_and_verify_keep_exact_byte_limits(self):
        s = spec();s['method'] = 'inverted_cdf'
        with tempfile.TemporaryDirectory() as directory:
            claim = Path(directory) / 'claim.json';claim.write_text(json.dumps(s))
            cases = [(['check', '-'], s, INPUT_BYTES),
                     (['verify', str(claim), '-'], audit(s), PROOF_BYTES)]
            for argv, document, cap in cases:
                prefix = json.dumps(document)
                for binary in (False, True):
                    for extra in (0, 1):
                        content = prefix + ' ' * (cap - len(prefix) + extra)
                        if binary:
                            content = content.encode()
                        with self.subTest(command=argv[0], binary=binary, extra=extra):
                            incoming = ChunkedInput(content, 4096)
                            code, value = self._chunked_call(argv, incoming)
                            self.assertEqual(code, 2 if extra else 0)
                            if extra:
                                self.assertEqual(value['status'], 'INVALID')
                            else:
                                self.assertEqual(incoming.end_reads, 1)
                            self.assertEqual(incoming.position, cap + extra)
                            self.assertTrue(all(0 < n <= cap + 1 for n in incoming.requests))

    def test_chunked_read_failure_after_valid_prefix_does_not_certify_it(self):
        with tempfile.TemporaryDirectory() as directory:
            claim = Path(directory) / 'claim.json';claim.write_text(json.dumps(spec()))
            for argv, document in [(['check', '-'], spec()),
                                   (['verify', str(claim), '-'], audit(spec()))]:
                for binary in (False, True):
                    content = json.dumps(document)
                    if binary:
                        content = content.encode()
                    for ending in ('none', 'zero', 'false', 'block', 'error', 'value'):
                        with self.subTest(command=argv[0], binary=binary, ending=ending):
                            incoming = ChunkedInput(content, len(content), ending)
                            code, value = self._chunked_call(argv, incoming)
                            self.assertEqual(code, 2)
                            self.assertEqual(value['status'], 'INVALID')
                            self.assertEqual(incoming.end_reads, 1)

    def test_utf8_chunks_join_before_decode_and_use_byte_not_character_cap(self):
        expected = {'label': 'é🙂'}
        text = json.dumps(expected, ensure_ascii=False)
        cap = len(text.encode('utf-8'))
        for content in (text, text.encode('utf-8')):
            for size in (1, 7):
                incoming = ChunkedInput(content, size)
                with patch('sys.stdin', incoming):
                    self.assertEqual(_read('-', cap), expected)
                self.assertFalse(incoming.closed)
                self.assertEqual(incoming.end_reads, 1)
                incoming = ChunkedInput(content, size)
                with patch('sys.stdin', incoming), self.assertRaises(Invalid):
                    _read('-', cap - 1)
                self.assertFalse(incoming.closed)

    @unittest.skipUnless(sys.platform.startswith('linux') and hasattr(os, 'openpty'), 'Linux canonical tty only')
    def test_raw_tty_process_rejects_already_queued_trailing_input(self):
        import termios
        program = ('import io,sys\nfrom quantilewitness.cli import run\n'
                   'sys.stdin=io.TextIOWrapper(io.FileIO(0,mode="rb",closefd=False),encoding="utf-8")\n'
                   'sys.argv=["quantilewitness"]+sys.argv[1:]\nraise SystemExit(run())\n')
        with tempfile.TemporaryDirectory() as directory:
            claim = Path(directory) / 'claim.json';claim.write_text(json.dumps(spec()))
            for argv, document in [(['check', '-'], spec()),
                                   (['verify', str(claim), '-'], audit(spec()))]:
                with self.subTest(command=argv[0]):
                    master, slave = os.openpty()
                    try:
                        self.assertTrue(termios.tcgetattr(slave)[3] & termios.ICANON)
                        payload = json.dumps(document).encode() + b'\nINVALID TRAILING INPUT\n\x04'
                        self.assertEqual(os.write(master, payload), len(payload))
                        child = subprocess.run([sys.executable, '-c', program] + argv, stdin=slave,
                                               capture_output=True, text=True, timeout=10)
                        self.assertEqual(child.returncode, 2, child.stderr)
                        self.assertEqual(json.loads(child.stdout)['status'], 'INVALID')
                        self.assertEqual(child.stderr, '')
                    finally:
                        os.close(master);os.close(slave)


if __name__ == '__main__':
    unittest.main()
