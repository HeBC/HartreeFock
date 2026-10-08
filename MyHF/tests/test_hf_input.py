"""Readable input preserves physical settings and rejects ambiguous input."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'PythonScript'))
from hf_input import InputError,numbers,parse_text,read_config,read_job
from scan_gcm_hf import main,target_points

BASE='''[calculation]
nucleus = Mg24
interaction = interaction with spaces.snt
hw = 16
[constraints]
Q20 = 1.0 1.5
Q22 = 0.5
Q21 = 0
'''


class ReadableInputTests(unittest.TestCase):
    def test_existing_examples_preserve_json_physics(self):
        for name in ('usda_mg24','evolved_ne20','multishell_o16'):
            path=ROOT/'examples/gcm'/name
            plain,execution=read_job(path.with_suffix('.inp'))
            legacy=read_config(path.with_suffix('.json'))
            self.assertEqual(plain,legacy,name)
            self.assertEqual(target_points(plain),target_points(legacy))
            self.assertTrue(Path(execution['output']).is_absolute())
            self.assertEqual(execution['passes'],['forward','reverse'])

    def test_lists_ranges_and_fortran_exponents(self):
        self.assertEqual(numbers('1, 2 3d-1'),[1,2,.3])
        self.assertEqual(numbers('0.1:0.3:0.1'),[.1,.2,.3])
        self.assertEqual(numbers('1.5:0.5:-0.5'),[1.5,1,.5])
        self.assertEqual(numbers('0:0:1'),[0])
        for invalid in ('0:1:0','0:1:-.1','0:1:.3','0:10000000:1','nan','inf','1:2','__import__("os")'):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                numbers(invalid)

    def test_comments_literal_windows_paths_and_solver(self):
        source=BASE.replace('interaction with spaces.snt',r'D:\Research\my input 100%.snt')
        source=source.replace('hw = 16','hw = 16 # oscillator energy')
        source+='''
[solver]
method = hybrid
check_stability = yes
seed = 81
constraint_tolerance = 1d-9
'''
        config,_=parse_text(source)
        self.assertEqual(config['interaction'],r'D:\Research\my input 100%.snt')
        self.assertEqual(config['solver'],dict(method='hybrid',check_stability=True,seed=81,constraint_tolerance=1e-9))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'case.inp';path.write_text(source)
            resolved=read_config(path)['interaction']
            if sys.platform.startswith('linux'):
                self.assertEqual(resolved,'/mnt/d/Research/my input 100%.snt')

    def test_path_pairs_and_free_constraints(self):
        config=read_config(ROOT/'examples/gcm/mg24_path.inp')
        self.assertEqual(target_points(config),[
            dict(Q20=1.,Q22=.5,Q21=0.),dict(Q20=1.5,Q22=1.,Q21=0.)])
        self.assertNotIn('Jx',[s['name'] for s in config['constraints']])
        config,_=parse_text(BASE.replace('Q21 = 0','Q21 = off'))
        self.assertEqual(len(config['constraints']),2)

    def test_operator_species_reference_and_cache_settings(self):
        source=BASE.replace('Q20 =','Q20p =')+'''
[operator Q20p]
name = Q20
weights = 1 0
units = oscillator
[operator Q22]
file = evolved tensor.snt
rank = 2
component = 2
parity = even
units = fm^2
normal_ordering = reference
reference_file = rho.npz
[operator Q21]
type = npz
file = Q21.npz
'''
        config,_=parse_text(source)
        self.assertEqual(config['constraints'][0]['operator'],dict(type='builtin',name='Q20',weights=[1,0],units='oscillator'))
        op=config['constraints'][1]['operator']
        self.assertEqual((op['rank'],op['mu'],op['parity'],op['reference_npz']),(2,2,0,'rho.npz'))
        self.assertEqual(config['constraints'][2]['operator'],dict(type='npz',path='Q21.npz'))

    def test_unknown_duplicate_or_incomplete_settings(self):
        invalid=[BASE.replace('hw =','hww ='),BASE+'\n[solvr]\nmethod = hybrid\n',
            BASE.replace('hw = 16','hw = 16\nhw = 20'),BASE+'\n[DEFAULT]\na=1\n',
            BASE+'\n[solver]\nmax_iteration=20\n',BASE+'\n[operator typo]\nname=Q20\n',
            BASE+'\n[quadrupole]\nfile=q.snt\n',BASE.replace('Q22 = 0.5','Q22 = 0.5 1')+
            '\n[path]\ncolumns=Q20\npoints=1\n',
            BASE+'\n[path]\ncolumns=Q20 Q22\npoints=1\n',
            BASE.replace('hw = 16','hw = 16\npasses = forward forward'),
            BASE+'\n[operator Q20]\ntype=npz\nfile=q.npz\nweights=1 0\n']
        for text in invalid:
            with self.subTest(text=text),self.assertRaises(InputError): parse_text(text)

    def test_error_includes_file_and_typo_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.inp';path.write_text(BASE+'\n[solver]\nmax_iteration=50\n')
            with self.assertRaisesRegex(InputError,r'bad.inp.*max_iteration.*max_iterations'):
                read_config(path)

    def test_cli_preview_is_readable_and_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'check.inp';out=Path(tmp)/'result'
            source=BASE.replace('interaction with spaces.snt',str(ROOT/'Interaction/usda.snt'))
            source=source.replace('hw = 16',f'hw = 16\noutput = {out}')
            path.write_text(source)
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream): self.assertEqual(main([str(path),'--check']),0)
            self.assertIn('distinct points: 2',stream.getvalue())
            self.assertIn('Constraints:',stream.getvalue())
            self.assertIn('Q20',stream.getvalue())
            self.assertFalse(out.exists())
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream): self.assertEqual(main(['--input',str(path),'--check-json']),0)
            self.assertEqual(len(json.loads(stream.getvalue())['points']),2)


if __name__=='__main__': unittest.main()
