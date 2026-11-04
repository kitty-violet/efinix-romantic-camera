"""Expose PLL control arrival times on a copy, without changing acceptance SDC.

The virtual zero output delay is only a way to report source clock insertion,
clock-to-Q and routed output delay. Its slack is not hardware acceptance.
"""
import argparse,json,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,EF,sha
from windows_process_tree import stop_launched_tree

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 base=a.base.resolve();root=a.root.resolve()
 assert base.is_relative_to(ROOT/'tmp') and root.is_relative_to(ROOT/'tmp') and not root.exists()
 r=json.loads((base/'result.json').read_text());assert r['original_and_route_unchanged']
 frozen={str(f):sha(f) for f in [base/'ti60f225_oob.sdc',base/'result.json']+[base/'outflow'/('ti60f225_oob'+s) for s in ('.map.v','.place','.vdb','.troutingtraces')]}
 root.mkdir();shutil.copy2(__file__,root/'runner.py')
 for d in ('outflow','work_pnr'):shutil.copytree(base/d,root/d)
 for f in base.iterdir():
  if f.suffix in ('.xml','.sdc','.ini','.placeloc'):shutil.copy2(f,root/f.name)
 sdc=(root/'ti60f225_oob.sdc').read_text()
 ports='[get_ports {pll_shift[*] pll_shift_sel[*] pll_shift_ena}]'
 old='set_false_path -to '+ports
 assert sdc.count(old)==1
 sdc=sdc.replace(old,'# Diagnostic only: expose PLL control output arrival, acceptance unchanged.')
 sdc+='\nset_output_delay -clock process_clk_150 -max 0 '+ports+'\nset_output_delay -clock process_clk_150 -min 0 '+ports+'\n'
 (root/'ti60f225_oob.sdc').write_text(sdc)
 tcl='report_timing_summary -file shift_initialized.rpt\n'
 for kind in ('setup','hold'):
  tcl+='report_timing -'+kind+' -from_clock {process_clk_150} -to {pll_shift[*] pll_shift_sel[*] pll_shift_ena} -npaths 40 -nworst 8 -file shift_'+kind+'.rpt\n'
 (root/'probe.tcl').write_text(tcl)
 (root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\ncall efx_run.bat ti60f225_oob.xml --flow sta_tclsh --tcl_script probe.tcl --timeout 90\nexit /b %errorlevel%\n')
 (root/'inputs.json').write_text(json.dumps(dict(base=str(base),frozen=frozen,diagnostic_sdc_sha256=sha(root/'ti60f225_oob.sdc'),tcl_sha256=sha(root/'probe.tcl'),runner_sha256=sha(__file__),hardware_access=False),indent=2))
 start=time.monotonic()
 with (root/'probe.log').open('x') as log:
  proc=subprocess.Popen(['cmd.exe','/d','/c','run.cmd'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
  try:code=proc.wait(timeout=120)
  except subprocess.TimeoutExpired:stop_launched_tree(proc);raise
 log=(root/'probe.log').read_text(errors='replace')
 assert code==0 and not re.search(r'Failed to run|^\s*(ERROR|FATAL)\s*:',log,re.M),log[-1500:]
 for f,h in frozen.items():assert sha(f)==h
 reports={}
 for kind in ('setup','hold'):
  f=root/'outflow'/('shift_'+kind+'.rpt');assert f.exists() and 'Path Start' in f.read_text()
  reports[str(f)]=sha(f)
 (root/'result.json').write_text(json.dumps(dict(status='PLL_SHIFT_ARRIVAL_REPORTS_REQUIRE_BOUND_CHECK',seconds=time.monotonic()-start,reports=reports,inputs_sha256=sha(root/'inputs.json'),original_and_route_unchanged=True,diagnostic_slack_not_acceptance=True,hardware_access=False),indent=2))
 print('PLL_SHIFT_ARRIVAL_REPORTS_REQUIRE_BOUND_CHECK')

if __name__=='__main__':main()
