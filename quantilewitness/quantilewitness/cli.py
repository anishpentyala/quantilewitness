"""Read-only claim/report processing; no application-level file writes."""
import argparse
import functools
import inspect
import json
import sys
from .core import Invalid, REPORT_SCHEMA, audit, verify_report

INPUT_BYTES = 64 * 1024
PROOF_BYTES = 1024 * 1024


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Invalid('duplicate JSON object key')
        result[key] = value
    return result


def _constant(value):
    raise Invalid('non-finite JSON numbers are not admitted')


def _read_stream(stream, limit):
    if stream is None:
        raise Invalid('standard input is unavailable')
    chunks, size = [], 0
    while True:
        chunk = stream.read(limit + 1 - size)
        if chunk is None:
            raise Invalid('input read is unavailable or would block')
        if isinstance(chunk, str):
            chunk = chunk.encode('utf-8')
        if not isinstance(chunk, bytes):
            raise Invalid('input read must return text or bytes')
        if not chunk:
            return b''.join(chunks)
        size += len(chunk)
        if size > limit:
            raise Invalid('input exceeds byte limit')
        chunks.append(chunk)


def _read(path, limit):
    if path == '-':
        content = _read_stream(getattr(sys.stdin, 'buffer', sys.stdin), limit)
    else:
        with open(path, 'rb') as stream:
            content = _read_stream(stream, limit)
    return json.loads(content.decode('utf-8'), object_pairs_hook=_pairs, parse_constant=_constant)


def _write_all(stream, payload):
    if stream is None:
        return False
    offset = 0
    while offset < len(payload):
        count = stream.write(payload[offset:])
        if type(count) is not int or not 1 <= count <= len(payload) - offset:
            return False
        offset += count
    return True


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        if sys.stderr is None:
            raise OSError('standard error is unavailable')
        super().error(message)

    def _print_message(self, message, file=None):
        if message and not _write_all(sys.stderr if file is None else file, message):
            raise OSError('output write made no valid progress')


def _emit(report):
    try:
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
        stream = sys.stdout
        if not _write_all(stream, payload):
            return False
        stream.flush()
        return True
    except (OSError, ValueError):
        return False


def _parser():
    kwargs = {}
    if getattr(sys.stdout, 'closed', False):
        supported = inspect.signature(argparse.ArgumentParser).parameters
        for name in ('color', 'colored_help'):
            if name in supported:
                kwargs[name] = False
        if 'color' in inspect.signature(argparse.HelpFormatter).parameters:
            kwargs['formatter_class'] = functools.partial(argparse.HelpFormatter, color=False)
    parser = _Parser(prog='quantilewitness', description='Check finite continuous-sample summary compatibility, not data provenance.', **kwargs)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check', help='check a local claim document', **kwargs)
    check.add_argument('spec', help='JSON path or - for stdin')
    verify = commands.add_parser('verify', help='verify a saved witness or contradiction', **kwargs)
    verify.add_argument('spec', help='JSON claim document')
    verify.add_argument('report', help='JSON check report, or - for stdin')
    return parser


def main(argv=None):
    if sys.stdout is None:
        return 2
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        if exc.code == 0:
            try:
                sys.stdout.flush()
            except (OSError, ValueError):
                return 2
        raise
    except (OSError, ValueError):
        return 2
    try:
        if args.command == 'verify' and args.spec == '-' and args.report == '-':
            raise Invalid('only one input may use stdin')
        spec = _read(args.spec, INPUT_BYTES)
        if args.command == 'check':
            report = audit(spec)
            code = {'FEASIBLE': 0, 'INFEASIBLE': 1, 'INVALID': 2, 'UNRESOLVED': 3}[report['status']]
        else:
            proof = _read(args.report, PROOF_BYTES)
            valid = verify_report(spec, proof)
            report = {'schema': 'quantilewitness/verification-v1', 'certificate_valid': valid,
                      'scope': 'certificate for the supplied claim document only; not data provenance'}
            code = 0 if valid else 1
    except (Invalid, ValueError, TypeError, OSError, UnicodeError, RecursionError) as exc:
        report = {'schema': REPORT_SCHEMA, 'status': 'INVALID', 'reason': str(exc)}
        code = 2
    return code if _emit(report) else 2


def run():
    try:
        code = main()
    except SystemExit as exc:
        code = exc.code
    streams = tuple(stream for stream in (sys.stdout, sys.stderr) if stream is not None)
    for stream in streams:
        try:
            stream.flush()
        except (OSError, ValueError):
            code = 2
    if code == 2:
        for stream in streams:
            try:
                stream.close()
            except (OSError, ValueError):
                pass
    return code
