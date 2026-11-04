"""Exercise actual mapped CPU/APB read mux through public JTAG scan pins."""
import argparse,json,os,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha,run
from validate_fixed_control_soc import DIRECT_TASKS
Q=Path('E:/quatra/questa_fse/win64')
def main():
 p=argparse.ArgumentParser();p.add_argument('--startup',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 source=a.startup.resolve();r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists()
 proof=json.loads((source/'result.json').read_text());meta=json.loads((source/'inputs.json').read_text())
 assert proof['status']=='FIRST_FAULT_MAP_STARTUP_AND_CAPTURE_PASS' and proof['inputs_sha256']==sha(source/'inputs.json')
 for n,h in meta['files'].items():assert sha(source/n)==h
 r.mkdir();ex=json.loads((source/'extraction.json').read_text());ports=ex['inputs'];lines=['`timescale 1ns/1ps','module tb;'];connections=[]
 for n,port in ports.items():
  lines.append(f'reg {ex["input_widths"][n]} {port}=0;');connections.append(f'.{port}({port})')
 for n in ex['outputs']:lines.append(f'wire {n};');connections.append(f'.{n}({n})')
 lines.append('mapped_control dut('+','.join(connections)+');')
 aliases={'clk':'process_clk_150','jtck':'jtag_inst1_TCK','jtdi':'jtag_inst1_TDI','jsel':'jtag_inst1_SEL','jcapture':'jtag_inst1_CAPTURE','jshift':'jtag_inst1_SHIFT'}
 tasks=(HERE/'board_integration/jtag_memory_tasks.svh').read_text().split('task jdebug(')[0]+DIRECT_TASKS
 tasks=tasks.replace('jupdate=0;','').replace('jupdate=1;','')
 for alias,n in aliases.items():tasks=re.sub(r'\b'+alias+r'\b',ports[n],tasks)
 tasks=re.sub(r'\bjtdo\b','jtag_inst1_TDO',tasks);lines.append(tasks)
 lines.append(f'always #3.333 {ports["process_clk_150"]}=~{ports["process_clk_150"]};')
 states={'core_count':0x12345678,'tdqss_count':0x23456789,'tac_count':0x3456789a,'twd_count':0x456789ab,'ticks':0x56789abc,
         'reset_rises':0x1234,'reset_falls':0x5678,'last_low':0x3456,'last_high':0x789a,
         'obs_sync':0xab25f,'age_saturated':1,'event_saturated':0,'reset_age':0x4567,'system_falls':0x89ab}
 # In this board only CLKOUT2 is selected. Actual map ties the other four
 # selection inputs to zero and folds those bits out of the APB read mux.
 # Do not drive unreachable bit values at the sequential extraction boundary.
 from audit_snapshot_cells import cells
 cell_map=cells(Path(meta['original_map']).read_text())
 for i in (7,8,10,11):assert cell_map[f'ddr_observer/obs_meta[{i}]~FF'][1]['D']=="1'b0"
 lines.append('initial begin')
 for n,v in states.items():lines.append(ports['\\ddr_observer/'+n]+f"=32'h{v:08x};")
 lines.append('#2000;')
 for n in ('i_arstn','i_pll_locked','pll_locked','user_pll_locked','process_pll_locked','\\startup_reset/released'):
  lines.append(ports[n]+'=1;')
 lines.append('#40000;')
 expected={0:0x54493630,0x70:0x44434f31,0x74:states['core_count'],0x78:states['tdqss_count'],0x7c:states['tac_count'],0x80:states['twd_count'],
           0x84:states['ticks'],0x88:0x56781234,0x8c:0x789a3456,0x90:0x3ab25f,0x94:0x89ab4567}
 for offset,value in expected.items():lines.append(f"jexpect(32'h{0xf8100000+offset:08x},32'h{value:08x});")
 for offset in range(0x70,0x98,4):lines.append(f"jreject(1,32'h{0xf8100000+offset:08x},2);")
 for address,size in ((0xf8100098,2),(0xf8100071,2),(0xf8100070,1),(0xf8100170,2),(0xf8110070,2)):
  lines.append(f"jreject(0,32'h{address:08x},{size});")
 lines.append("jreject(1,32'hf8100008,2);")
 lines.append(f'{ports["i_arstn"]}=0;#2000;{ports["i_arstn"]}=1;#40000;')
 lines.append('jexpect(32\'hf8100070,32\'h44434f31);$display("PASS MAPPED_DCO1_JTAG reads=12 rejected=16 reset=1");$finish;end')
 lines.append('initial begin #1000000;$fatal(1,"mapped JTAG simulation budget");end\nendmodule')
 (r/'tb.sv').write_text('\n'.join(lines))
 for n in ['mapped_control.v',*sorted(f.name for f in source.glob('efx_*.v'))]:shutil.copy2(source/n,r/n)
 inputs=dict(original_map=meta['original_map'],original_map_sha256=meta['original_map_sha256'],
             source_startup_result_sha256=sha(source/'result.json'),source_extraction_sha256=sha(source/'extraction.json'),
             files={f.name:sha(f) for f in r.iterdir()},runner_sha256=sha(__file__),hardware_access=False)
 (r/'inputs.json').write_text(json.dumps(inputs,indent=2));os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 run([Q/'vlib.exe','work'],r,'vlib.log',15)
 run([Q/'vlog.exe','-sv','-timescale','1ns/1ps',*sorted(r.glob('efx_*.v')),'mapped_control.v','tb.sv'],r,'compile.log',40)
 text,seconds=run([Q/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',90)
 assert 'PASS MAPPED_DCO1_JTAG reads=12 rejected=16 reset=1' in text
 result=dict(status='DDR_OBSERVER_MAPPED_JTAG_APB_PASS',inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),seconds=seconds,hardware_access=False)
 (r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
