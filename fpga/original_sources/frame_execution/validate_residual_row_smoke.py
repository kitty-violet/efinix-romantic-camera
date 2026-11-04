"""Bounded four-state gate suite, including a real80-pixel row and all faults."""
import json,sys,shutil
from pathlib import Path
import validate_residual_row as base
from run_units import HERE,sha
original_sources=base.row_sources
variant=HERE/'variants/shared_read_mux/residual_row_shared.sv'
base.row_sources=lambda:[variant if f.name=='residual_row_shared.sv' else f for f in original_sources()]
original_vectors=base.vectors
def smoke_vectors(root):
 result=original_vectors(root);lines=(root/'cases.txt').read_text().splitlines()
 cases=[list(map(int,s.split())) for s in lines[1:]]
 chosen=[cases[k] for k in (0,16,17,33,36)]
 (root/'cases.txt').write_text(f'{len(chosen)} {result["words"]}\n'+''.join(' '.join(map(str,r))+'\n' for r in chosen))
 result.update(cases=len(chosen),scalar_checks=(sum(r[2]*r[3] for r in chosen)+28)*24,recovery_width=7,full_suite=False)
 return result
base.vectors=smoke_vectors
if __name__=='__main__':
 before={str(p):sha(p) for p in (Path(__file__),Path(base.__file__),variant)}
 sys.argv+=['--testbench',str(HERE/'tb_residual_row_map_smoke.sv')];base.main()
 root=Path(sys.argv[sys.argv.index('--root')+1])
 for p,h in before.items():assert sha(p)==h
 shutil.copy2(Path(__file__),root/Path(__file__).name)
 (root/'override-runner.json').write_text(json.dumps(before,indent=2))

