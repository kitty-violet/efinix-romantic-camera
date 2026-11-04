"""Reuse fixed-window numeric/backpressure/reset scoreboard and official models."""
import argparse,json,os,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--reuse-clean',type=Path);a=p.parse_args()
root=a.root.resolve();candidate=a.source.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
base=HERE/'validate_fixed_window_dot8x32.py';code=base.read_text()
anchor="EF/'sim_models/verilog/efx_srl8.v']"
assert code.count(anchor)==1;code=code.replace(anchor,"EF/'sim_models/verilog/efx_srl8.v',EF/'sim_models/verilog/efx_lut4.v']")
code=code.replace('from run_units import ROOT,HERE,EF,MS,run,sha',
                  "from run_units import ROOT,HERE,EF,run,sha\nMS=Path('E:/quatra/questa_fse/win64')")
# This installed nodelocked Starter license allows exactly one active simulator.
# Keep synthesis/placement parallel; serialize simulator seeds and reuse only
# hash-identical completed logs using the original runner's checks.
code=code.replace('ThreadPoolExecutor(max_workers=3)','ThreadPoolExecutor(max_workers=1)')
driver=root.parent/(root.name+'-runner.py');assert not driver.exists();driver.write_text(code)
driver.with_suffix('.json').write_text(json.dumps(dict(original_runner=str(base),original_sha256=sha(base),
    wrapper_sha256=sha(__file__),generated_sha256=sha(driver),source_sha256=sha(candidate),hardware_access=False),indent=2))
os.environ['FPGA_FIXED_ARITHMETIC']=str(candidate);os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
sys.argv=[str(driver),'--root',str(root),'--testbench',str(HERE/'variants/fixed_shared_column_history_test/tb_fixed_window_dot8x32.sv')]
if a.reuse_clean:sys.argv+=['--reuse-clean',str(a.reuse_clean.resolve())]
runpy.run_path(str(driver),run_name='__main__')
