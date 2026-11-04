"""Conservative SHIFT setup/high bounds including clock insertion and routing."""
import argparse,json,re
from pathlib import Path
from run_units import ROOT,sha
from prepare_ddr_shift_guard_board import audit_delta,CAL,proof

def rows(path):
 out=[]
 for b in re.split(r'(?=^Slack\s*:)',path.read_text(),flags=re.M)[1:]:
  def value(pattern):
   m=re.search(pattern,b,re.M);assert m,pattern;return m[1].strip()
  assert value(r'^Launch Clock\s*:\s*(.*)').startswith('process_clk_150 ')
  out.append(dict(start=value(r'^Path Start\s*:\s*(.*)'),end=value(r'^Path End\s*:\s*(.*)'),
                  arrival=float(value(r'^End-of-path arrival time\s*:\s*([\d.]+)')),
                  uncertainty=float(value(r'^[+-] Clock Uncertainty\s*:\s*([\d.]+)'))))
 expected={'pll_shift[0]','pll_shift[1]','pll_shift[2]','pll_shift_ena','pll_shift_sel[2]'}
 assert {r['end'] for r in out}==expected
 assert all(r['start'].endswith('|CLK') for r in out)
 return out

def bounds(folder,setup,high,low=None):
 maximum=rows(folder/'outflow/shift_setup.rpt');minimum=rows(folder/'outflow/shift_hold.rpt')
 controls={'pll_shift_ena','pll_shift_sel[2]'}
 # Cross-corner extrema are deliberately pessimistic. Both reports include
 # clock insertion, clock-to-Q and output routing. Keep both uncertainty terms
 # plus 10ps report-rounding allowance instead of treating routing as zero.
 penalty=max(r['uncertainty'] for r in maximum)+max(r['uncertainty'] for r in minimum)+.010
 latest_code=max(r['arrival'] for r in maximum if r['end'] not in controls)
 earliest_control=min(r['arrival'] for r in minimum if r['end'] in controls)
 latest_control=max(r['arrival'] for r in maximum if r['end'] in controls)
 result=dict(setup_lower_ns=setup+earliest_control-latest_code-penalty,
             common_control_high_lower_ns=high+earliest_control-latest_control-penalty,
             logical_setup_ns=setup,logical_high_ns=high,penalty_ns=penalty,
             maximum=maximum,minimum=minimum)
 if low is not None:
  result.update(logical_low_ns=low,common_control_low_lower_ns=low+earliest_control-latest_control-penalty)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 proof();audit_delta(a.board);root=a.root.resolve();meta=json.loads((root/'inputs.json').read_text());r=json.loads((root/'result.json').read_text())
 assert r['inputs_sha256']==sha(root/'inputs.json')
 for f,h in meta['frozen'].items():assert sha(f)==h
 for f,h in r['reports'].items():assert sha(f)==h
 assert sha(Path(meta['base'])/'outflow/ti60f225_oob.map.v')==sha(a.board/'outflow/ti60f225_oob.map.v')
 vendor=json.loads((CAL/'result.json').read_text())
 setup=min(float(v) for line in vendor['measurements'] for v in re.findall(r'setup_ns=([\d.]+)',line))
 high=min(float(v) for line in vendor['measurements'] for v in re.findall(r'high_ns=([\d.]+)',line))
 gaproot=ROOT/'tmp/frame-execution/ddr-shift-min-gap1';gap=json.loads((gaproot/'result.json').read_text())
 assert gap['status']=='PLL_SHIFT_MINIMUM_GAP_PASS' and sha(gaproot/'simulation.log')==gap['log_sha256']
 for f,h in gap['inputs'].items():assert sha(gaproot/f)==h
 assert sha(a.board/'ddr516_pll_shift_guard.sv')==gap['inputs']['ddr516_pll_shift_guard.sv']
 b=bounds(root,setup,high,gap['low_ns'])
 assert b['setup_lower_ns']>=2.5,b
 assert b['common_control_high_lower_ns']>=10,b
 assert b['common_control_low_lower_ns']>=10,b
 out=dict(status='PLL_SHIFT_ROUTED_SETUP_HIGH_BOUND_PASS',bounds=b,source_inputs_sha256=sha(a.board/'inputs.json'),
          reports_inputs_sha256=sha(root/'inputs.json'),reports=r['reports'],vendor_result_sha256=sha(CAL/'result.json'),
          min_gap_result=str(gaproot/'result.json'),min_gap_result_sha256=sha(gaproot/'result.json'),
          acceptance_sdc_unchanged=True,physical_calibration_proven=False,hardware_access=False)
 (root/'bound.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))

if __name__=='__main__':main()
