"""Readable .inp files and legacy JSON inputs for the named-constraint HF scan.

Only standard-library parsing is used. No input expressions are executed.
"""
import configparser
from decimal import Decimal, InvalidOperation
import difflib
import json
import math
from pathlib import Path
import re
import sys


class InputError(ValueError):
    """An input setting that needs correcting before a calculation can start."""


def _text(value):
    value = value.strip()
    if value[:1] in ('"', "'"):
        if len(value) < 2 or value[-1] != value[0]:
            raise InputError('unclosed quote')
        value = value[1:-1]
    return value


def _number(value):
    number = float(value.replace('D', 'e').replace('d', 'e'))
    if not math.isfinite(number):
        raise InputError('numbers must be finite')
    return number


def _integer(value):
    if not re.fullmatch(r'[+-]?\d+', value.strip()):
        raise InputError('expected an integer')
    return int(value)


def _boolean(value):
    choices = {'yes': True, 'true': True, 'on': True,
               'no': False, 'false': False, 'off': False}
    if value.lower() not in choices:
        raise InputError('expected yes/no or true/false')
    return choices[value.lower()]


def numbers(value, limit=10000):
    """Whitespace/comma lists or an inclusive start:stop:step range."""
    if ':' not in value:
        words = value.replace(',', ' ').split()
        if not words or len(words) > limit:
            raise InputError(f'need 1 to {limit} numbers')
        return [_number(word) for word in words]
    words = value.split(':')
    if len(words) != 3:
        raise InputError('a range must be start:stop:step, e.g. 1.0:2.0:0.5')
    try:
        start, stop, step = (Decimal(word.strip().replace('D', 'e').replace('d', 'e')) for word in words)
    except InvalidOperation as error:
        raise InputError('invalid range number') from error
    if not all(x.is_finite() for x in (start, stop, step)) or step == 0:
        raise InputError('a range needs finite numbers and a nonzero step')
    if (stop - start) * step < 0:
        raise InputError('range step points away from the stop value')
    intervals = (stop - start) / step
    if intervals != intervals.to_integral_value():
        raise InputError('the range must reach its stop exactly; use an explicit list otherwise')
    if intervals >= limit:
        raise InputError(f'range exceeds the {limit}-point limit')
    return [_number(str(start + i * step)) for i in range(int(intervals) + 1)]


def _settings(section, allowed):
    result = {}
    for key, value in section.items():
        key = key.lower()
        if key not in allowed:
            close = difflib.get_close_matches(key, allowed, n=1)
            hint = f'; did you mean {close[0]}?' if close else ''
            raise InputError(f'[{section.name}] unknown setting {key}{hint}')
        if key in result:
            raise InputError(f'[{section.name}] repeated setting {key}')
        result[key] = _text(value)
    return result


def _convert(values, key, converter):
    try:
        return converter(values[key])
    except (ValueError, ArithmeticError) as error:
        raise InputError(f'{key} = {values[key]}: {error}') from error


def _operator(section, basis, component=None):
    allowed = ('type', 'file', 'name', 'weights', 'units', 'rank', 'component',
               'parity', 'normal_ordering', 'basis', 'reference_file', 'hermitian_tolerance')
    values = _settings(section, allowed)
    if component is not None:
        for key in ('name', 'weights', 'rank', 'component', 'parity'):
            if key in values:
                raise InputError(f'[quadrupole] {key} is fixed by the Q20/Q22 definition')
        values.update(rank='2', component=str(component), parity='even')
    kind = values.get('type', 'tensor_snt' if 'file' in values else 'builtin')
    if component is not None and kind != 'tensor_snt':
        raise InputError('[quadrupole] requires a tensor_snt file shared by Q20 and Q22')
    by_type = {
        'builtin': {'type', 'name', 'weights', 'units'},
        'tensor_snt': {'type', 'file', 'units', 'rank', 'component', 'parity',
                       'normal_ordering', 'basis', 'reference_file', 'hermitian_tolerance'},
        'npz': {'type', 'file'},
    }
    if kind not in by_type:
        raise InputError(f'[{section.name}] type must be builtin, tensor_snt or npz')
    extra = set(values) - by_type[kind]
    if extra:
        raise InputError(f'[{section.name}] {kind} does not use {", ".join(sorted(extra))}')
    op = {'type': kind}
    if kind == 'builtin':
        for key in ('name', 'units'):
            if key in values:
                op[key] = values[key]
        if 'weights' in values:
            op['weights'] = _convert(values, 'weights', lambda v: numbers(v, 2))
            if len(op['weights']) != 2:
                raise InputError(f'[{section.name}] weights needs proton and neutron weights')
        return op
    required = {'file'}
    if kind == 'tensor_snt':
        required.update(('rank', 'component', 'parity', 'normal_ordering', 'units'))
    missing = required - set(values)
    if missing:
        raise InputError(f'[{section.name}] missing {", ".join(sorted(missing))}')
    op['path'] = values['file']
    if kind == 'tensor_snt':
        op['rank'] = _convert(values, 'rank', _integer)
        op['mu'] = _convert(values, 'component', _integer)
        if values['parity'].lower() not in ('even', 'odd', '0', '1'):
            raise InputError(f'[{section.name}] parity must be even/odd or 0/1')
        op['parity'] = 0 if values['parity'].lower() in ('even', '0') else 1
        op['units'] = values['units']
        op['normal_ordering'] = values['normal_ordering']
        op['basis_representation'] = values.get('basis', basis)
        if 'reference_file' in values:
            op['reference_npz'] = values['reference_file']
        if 'hermitian_tolerance' in values:
            op['hermitian_tolerance'] = _convert(values, 'hermitian_tolerance', _number)
    return op


def parse_text(text):
    parser = configparser.ConfigParser(interpolation=None, delimiters=('=',),
        comment_prefixes=('#', ';', '!'), inline_comment_prefixes=('#', ';'),
        strict=True, empty_lines_in_values=False)
    parser.optionxform = str
    try:
        parser.read_string(text)
    except configparser.Error as error:
        raise InputError(str(error)) from error
    if parser.defaults():
        raise InputError('[DEFAULT] is not supported; use explicit sections')
    for name in parser.sections():
        if name not in ('calculation', 'constraints', 'solver', 'quadrupole', 'path') and not name.startswith('operator '):
            raise InputError(f'unknown section [{name}]')
    for required in ('calculation', 'constraints'):
        if required not in parser:
            raise InputError(f'missing [{required}] section')
    values = _settings(parser['calculation'],
        ('nucleus', 'interaction', 'hw', 'memory_mb', 'basis', 'max_points', 'output', 'passes'))
    for key in ('nucleus', 'interaction', 'hw'):
        if key not in values or not values[key]:
            raise InputError(f'[calculation] missing {key}')
    config = {'nucleus': values['nucleus'], 'interaction': values['interaction'],
              'hw_MeV': _convert(values, 'hw', _number),
              'basis_representation': values.get('basis', 'HO')}
    for key, converter in (('memory_mb', _number), ('max_points', _integer)):
        if key in values:
            config[key] = _convert(values, key, converter)
    limit = config.get('max_points', 10000)
    if limit < 1:
        raise InputError('max_points must be positive')
    execution = {}
    if 'output' in values:
        if not values['output']:
            raise InputError('[calculation] output cannot be empty')
        execution['output'] = values['output']
    if 'passes' in values:
        execution['passes'] = values['passes'].replace(',', ' ').split()
        if (not execution['passes'] or len(set(execution['passes'])) != len(execution['passes']) or
                set(execution['passes']) - {'forward', 'reverse'}):
            raise InputError('passes must be forward, reverse, or forward reverse')
    if 'solver' in parser:
        integers = {'max_iterations', 'diagonalization_steps', 'gradient_steps', 'max_cg', 'seed'}
        floats = {'gradient_tolerance', 'constraint_tolerance', 'energy_tolerance',
                  'trust_radius', 'curvature_tolerance', 'precondition_floor'}
        fields = _settings(parser['solver'], integers | floats | {'method', 'check_stability', 'precondition'})
        config['solver'] = {key: _convert(fields, key, _integer if key in integers else
            _number if key in floats else _boolean if key in ('check_stability', 'precondition') else str) for key in fields}
    specs = []
    declarations = dict(parser['constraints'])
    for name, value in declarations.items():
        if value.strip().lower() in ('off', 'free'):
            continue
        spec = {'name': name, 'values': _convert({name: value}, name, lambda v: numbers(v, limit))}
        section = 'operator ' + name
        if section in parser:
            spec['operator'] = _operator(parser[section], config['basis_representation'])
        elif name in ('Q20', 'Q22') and 'quadrupole' in parser:
            spec['operator'] = _operator(parser['quadrupole'], config['basis_representation'],
                                         component=0 if name == 'Q20' else 2)
        specs.append(spec)
    if not specs:
        raise InputError('[constraints] needs at least one active constraint')
    for section in parser.sections():
        if section.startswith('operator ') and section[len('operator '):] not in declarations:
            raise InputError(f'[{section}] has no matching name in [constraints]')
    if 'quadrupole' in parser:
        used = [s for s in specs if s['name'] in ('Q20', 'Q22') and 'operator '+s['name'] not in parser]
        if not used:
            raise InputError('[quadrupole] is unused; enable Q20 or Q22 without an individual operator section')
    config['constraints'] = specs
    if 'path' in parser:
        fields = _settings(parser['path'], ('columns', 'points'))
        if set(fields) != {'columns', 'points'}:
            raise InputError('[path] requires columns and indented points rows')
        columns = fields['columns'].replace(',', ' ').split()
        active = {s['name']: s for s in specs}
        if not columns or len(set(columns)) != len(columns) or set(columns) - set(active):
            raise InputError('[path] columns must be unique active constraint names')
        fixed = {}
        for name, spec in active.items():
            if name not in columns:
                if len(spec['values']) != 1:
                    raise InputError(f'[path] {name} must be a column or have one fixed value')
                fixed[name] = spec['values'][0]
        points = []
        for row in fields['points'].splitlines():
            if not row.strip():
                continue
            if ':' in row:
                raise InputError('[path] points are rows of numbers, not ranges')
            data = numbers(row, len(columns))
            if len(data) != len(columns):
                raise InputError('[path] each point must have one value per column')
            points.append(dict(fixed, **dict(zip(columns, data))))
            if len(points) > limit:
                raise InputError('[path] exceeds max_points')
        if not points:
            raise InputError('[path] needs at least one point')
        config['points'] = points
    return config, execution


def read_job(path):
    path = Path(path).resolve()
    try:
        text = path.read_text(encoding='utf-8-sig')
        if path.suffix.lower() == '.json':
            config, execution = json.loads(text), {}
        else:
            config, execution = parse_text(text)
        def resolve(value):
            if sys.platform.startswith('linux') and re.match(r'^[A-Za-z]:[\\/]', value):
                value = '/mnt/' + value[0].lower() + '/' + value[3:].replace('\\', '/')
            target = Path(value).expanduser()
            return str((path.parent / target).resolve() if not target.is_absolute() else target.resolve())
        config['interaction'] = resolve(config['interaction'])
        for spec in config['constraints']:
            for key in ('path', 'reference_npz'):
                op = spec.get('operator', {})
                if key in op:
                    op[key] = resolve(op[key])
        if 'output' in execution:
            execution['output'] = resolve(execution['output'])
        return config, execution
    except (ValueError, KeyError, TypeError, ArithmeticError) as error:
        raise InputError(f'{path}: {error}') from error


def read_config(path):
    """Return physical settings only; output/pass controls do not affect physics hashes."""
    return read_job(path)[0]
