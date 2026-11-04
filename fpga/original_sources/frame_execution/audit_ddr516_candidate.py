"""Bind a completed isolated DDR candidate result to its actual source evidence."""
import argparse,json,re,subprocess,sys
from pathlib import Path
from run_units import ROOT,HERE,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();result=json.loads((root/'result.json').read_text());meta=json.loads((root/'inputs.json').read_text())
 assert sha(root/'inputs.json')==result['inputs_sha256']
 log=(root/'simulation.log').read_text(errors='replace');assert sha(root/'simulation.log')==result['log_sha256']
 for item in meta['inputs']:assert sha(item['snapshot'])==item['sha256'],item['snapshot']
 subprocess.run([sys.executable,str(HERE/'audit_ddr516_pin_timing.py'),'--root',str(root)],check=True,stdout=subprocess.DEVNULL)
 pins=json.loads((root/'pin_timing_audit.json').read_text())
 original=(ROOT/'tmp/frame-execution/ddr516-generated2/ip/ddr516_reference/Testbench/modelsim/ddr516_reference.sv').read_text()
 snapshot=(root/'ddr516_reference.sv').read_text()
 assert snapshot.split('//pragma protect',1)[1]==original.split('//pragma protect',1)[1]
 rom=json.loads((ROOT/'tmp/frame-execution/ddr516-read-pre-guard1/rom_guard_audit.json').read_text())
 assert rom['status']=='ROM_GUARD_STATIC_AUDIT_PASS' and sha(root/'ddr3_controller.bin')==rom['candidate_rom_sha256']
 if meta['pll_protocol_fix']:
  pll=json.loads((root/'pll_protocol_proof.json').read_text());assert pll['status']=='PLL_PROTOCOL_CORRECTION_PASS'
  model=root/('EFX_PLL_V3_FAST_PROTOCOL.v' if meta['fast_pll'] else 'EFX_PLL_V3_PROTOCOL.v')
  assert sha(model)==pll['fast_corrected_sha256' if meta['fast_pll'] else 'corrected_sha256']
 micron_errors=[line for line in log.splitlines() if re.search(r'\b(?:ERROR|Error):',line)]
 cal=re.search(r'DDR516_CAL_DONE t_ns=([\d.]+) pass=([01x])',log)
 traffic=re.search(r'DDR516_TRAFFIC write_beats=(\d+) read_beats=(\d+) writes=(\d+) reads=(\d+) bytes_each_direction=(\d+)',log)
 assert cal and traffic,'Audit requires a completed calibration and normal traffic run'
 passed=(result['status']=='DDR516_STRICT_REFERENCE_PASS' and cal[2]=='1' and not micron_errors
  and pins['status']=='PIN_TRAS_TIMING_PASS' and tuple(map(int,traffic.groups()))==(256,256,4,4,4096))
 out=dict(status='ISOLATED_DDR_CANDIDATE_PASS' if passed else 'ISOLATED_DDR_CANDIDATE_FAIL',
  protected_controller_identical=True,all_input_hashes_match=True,command_rom_sha256=sha(root/'ddr3_controller.bin'),
  pll_protocol_correction=meta['pll_protocol_fix'],micron_error_count=len(micron_errors),
  calibration_done_ns=float(cal[1]),calibration_pass=cal[2]=='1',host_seconds=result['seconds'],
  write_beats=int(traffic[1]),read_beats=int(traffic[2]),bytes_each_direction=int(traffic[5]),
  minimum_act_to_pre_ns=pins['minimum_act_to_pre_ns'],minimum_read_row_act_to_pre_ns=pins['minimum_read_row_act_to_pre_ns'],
  closed_banks=pins['closed_banks'],refreshes=pins['refreshes'],result_sha256=sha(root/'result.json'),
  limits=['isolated candidate; not production board integration','project command ROM and public PLL simulation corrections; not vendor-approved',
  'explicit power-on memory reset assumption','no final resource/P&R/STA/board proof','not camera-to-screen frame latency or interval'],
  board_release=False,full_board=False)
 (root/'candidate_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
 if not passed:raise SystemExit(1)

if __name__=='__main__':main()
