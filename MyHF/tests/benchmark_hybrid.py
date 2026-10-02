"""Reproducible same-seed comparison; JSON includes all contractions, not just iterations."""
from test_hybrid import load, Solver, Options
import json
import time
import resource
import numpy as np

rows=[]
for isotope,interaction,target in [('Mg24','usda.snt',[1.,.5]),('Mg24','usdb.snt',[2.,.5]),
                                    ('O17','FCI_HF_FCI_O17_e3_hw16_E39.snt',[1.,.5])]:
    for method in ('gradient','hybrid'):
        h=load(isotope,512.,interaction)
        solver=Solver(h,active=(0,1,4),options=Options(method=method,max_iterations=1200))
        start=time.perf_counter()
        r=solver.solve(target+[0.])
        r.pop('occupied')
        # Peak RSS is process-wide high water; clearly label it, not per-solve allocation.
        rows.append(dict(isotope=isotope,interaction=interaction,method=method,seconds=time.perf_counter()-start,
                         process_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,**r))
print(json.dumps(rows,indent=2,allow_nan=False))
