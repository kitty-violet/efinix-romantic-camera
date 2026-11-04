"""Static-port row implementation; freeze override as well as delegated runner."""
import json,sys,shutil
from pathlib import Path
import validate_residual_row as base
from run_units import HERE,sha
original=base.row_sources
variant=HERE/'variants/shared_read_mux/residual_row_shared.sv'
base.row_sources=lambda:[variant if f.name=='residual_row_shared.sv' else f for f in original()]
if __name__=='__main__':
 before={str(p):sha(p) for p in (Path(__file__),Path(base.__file__),variant)}
 base.main()
 root=Path(sys.argv[sys.argv.index('--root')+1])
 for p,h in before.items():assert sha(p)==h
 shutil.copy2(Path(__file__),root/Path(__file__).name)
 (root/'override-runner.json').write_text(json.dumps(before,indent=2))
