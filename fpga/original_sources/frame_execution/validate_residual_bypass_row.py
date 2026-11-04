"""Exact real-arena residual two-pass adapter using the same RQ bypass pool."""
import json,sys,shutil
from pathlib import Path
import validate_residual_row as base
from validate_residual_bypass import selected
from run_units import HERE,sha
variant=HERE/'variants/residual_bypass'
base.row_sources=lambda:selected()+[HERE/'tensor_arena12.sv',variant/'residual_row_shared.sv']
if __name__=='__main__':
 before={str(p):sha(p) for p in (Path(__file__),Path(base.__file__),variant/'residual_row_shared.sv')}
 sys.argv+=['--testbench',str(variant/'tb_residual_row_shared.sv')]
 base.main()
 root=Path(sys.argv[sys.argv.index('--root')+1])
 for p,h in before.items():assert sha(p)==h
 shutil.copy2(Path(__file__),root/Path(__file__).name)
 (root/'override-runner.json').write_text(json.dumps(before,indent=2))
