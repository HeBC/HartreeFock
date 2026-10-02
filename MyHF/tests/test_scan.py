import csv
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'PythonScript'))
import scan_hybrid_pes as scan

class ScanTests(unittest.TestCase):
    def test_gamma_normalization(self):
        args=SimpleNamespace(beta=[.2],gamma=[30.],nucleus='Mg24',hw=16.,reverse=False)
        point=scan.grid(args)[0]
        self.assertAlmostEqual(point['q2']/point['q0'],(2/3)**.5)
        self.assertEqual(point['gamma_deg'],30.)

    def test_recover_partial_journal_excludes_failed_energy(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name)
            rows=[dict(point_id=0,converged=True,energy_MeV=-1.,fock_evaluations=2,hessian_evaluations=3),
                  dict(point_id=1,converged=False,energy_MeV=-10.,fock_evaluations=2,hessian_evaluations=3)]
            journal=''.join(json.dumps(row)+'\n' for row in rows)+'{"point_id":2'
            (p/'points.jsonl').write_text(journal)
            (p/'settings.json').write_text(json.dumps(dict(planned_points=3)))
            (p/'grid.csv').write_text('point_id\n0\n1\n2\n')
            summary=scan.recover_surface(p)
            self.assertEqual(summary['converged'],1)
            self.assertEqual(summary['minimum_converged_energy_MeV'],-1.)
            self.assertFalse(summary['scan_complete'])
            self.assertEqual((p/'points.jsonl').read_text(),journal)
            with (p/'surface.csv').open() as f: result=list(csv.DictReader(f))
            self.assertEqual(result[1]['accepted_energy_MeV'],'')

    def test_memory_admission(self):
        with self.assertRaises(MemoryError): scan.main(['--memory-mb','1','--check'])

if __name__=='__main__': unittest.main()
