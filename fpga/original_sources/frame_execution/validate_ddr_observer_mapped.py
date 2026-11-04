"""Simulate the complete public observer extracted verbatim from the real map."""
import argparse,json,os,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,sha,run
from ddr_cal_observer import audit_delta
from audit_ddr_observer import mapped
from prepare_ddr_observer_board import PROOF
Q=Path('E:/quatra/questa_fse/win64')
def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();root=a.root.resolve();audit_delta(board);net=board/'outflow/ti60f225_oob.map.v';m=mapped(net)
 old=ROOT/'tmp/frame-execution/snapshot-control-startup2/extractor.py';code=old.read_text()
 code=code.replace("source=ROOT/'tmp/frame-execution/connected-board-control-snapshot2/outflow/ti60f225_oob.map.v'",f'source=Path({str(net)!r})')
 code=code.replace("out=ROOT/'tmp/frame-execution/snapshot-control-startup2'",f'out=Path({str(root)!r})')
 code=code.replace('02cb1d5e1f6f0a290fc2c98b7d38aac14dbc2ad116a76064a12244bf88b19ef4',sha(net))
 start=code.index('cuts=set(');end=code.index('assert all(n in decl',start)
 cuts=('process_clk_150','ddr_core_clk200','sdram_clk','rx_cal_clk','tx_cal_clk_90edge','w_arstn','ddr_reset_n','ddr_cke','pll_shift_ena','pll_shift','pll_shift_sel')
 # The APB mux is flattened into the CPU response mux. Extract every actual
 # observer state cell, then expose its Q nets through explicit test-only aliases.
 # The RTL APB mux is tested separately; do not call that mux mapped coverage.
 states=('core_count','tdqss_count','tac_count','twd_count','ticks','reset_falls','reset_rises','last_high','last_low','event_saturated','age_saturated','obs_sync','system_falls','reset_age')
 outputs=tuple('probe_'+s for s in states)
 aliases=''.join('wire '+('[31:0]' if s in states[:5] else '[19:0]' if s=='obs_sync' else '' if s in ('event_saturated','age_saturated') else '[15:0]')+' probe_'+s+';\nassign probe_'+s+' = \\ddr_observer/'+s+' ;\n' for s in states)
 code=code.replace('decl={}','text += '+repr(aliases)+'\ndecl={}')
 start=code.index('cuts=set(');end=code.index('assert all(n in decl',start)
 code=code[:start]+f'cuts=set({cuts!r})\noutputs={outputs!r}\n'+code[end:]
 start=code.index('def owned(n):');end=code.index('selected=set()',start)
 code=code[:start]+"def owned(n):\n    return n.startswith('\\\\ddr_observer/')\n"+code[end:]
 code=code.replace('for c in selected:\n    used|=deps[c]\n    for n,d in drivers.items():\n        if c in d:used.add(n)',
                   'for c in selected:used|=deps[c]\nfor n,d in drivers.items():\n    if d & selected:used.add(n)')
 code=code.replace('Exact mapped CPU plus APB control and combinational fan-in; external state cut at FF/RAM/primary/protected boundaries; no board/PHY proof','Actual mapped public DDR observer, external clock/status/address stimuli; no PHY/controller model and no calibration claim')
 exec(compile(code,str(old),'exec'),{'__name__':'_extract','print':lambda *a,**k:None})
 (root/'extractor.py').write_text(code);ex=json.loads((root/'extraction.json').read_text())
 assert set(ex['inputs'])==set(cuts),('Observer cone is not closed',set(ex['inputs'])-set(cuts))
 tb=(PROOF/'tb.sv').read_text()
 actual=dict(zip(cuts,('clk','core','tdqss','tac','twd','sysn','ddrn','cke','ena','code','sel')))
 replacement=''
 for s in states:
  width='[31:0]' if s in states[:5] else '[19:0]' if s=='obs_sync' else '' if s in ('event_saturated','age_saturated') else '[15:0]'
  replacement+=f'wire {width} probe_{s};\n'
 replacement+='mapped_control dut('+','.join(f'.{port}({actual[name]})' for name,port in ex['inputs'].items())+','+','.join(f'.{n}({n})' for n in outputs)+');\n'
 # Same read mux from frozen source, only Q references renamed for this TB.
 rtl=(board/'rtl/ti60f225_oob_top.v').read_text()
 mux=rtl[rtl.index(' // Each APB word is coherent'):];mux=mux[:mux.index('endmodule')]
 mux=mux.replace('read_data','mapped_read_data').replace('read_address','address').replace('system_reset_n','sysn')
 for s in sorted(states,key=len,reverse=True):mux=re.sub(r'\b'+s+r'\b','probe_'+s,mux)
 replacement+='reg [31:0] mapped_read_data; assign data=mapped_read_data;\n'+mux
 anchor='ddr_cal_observer dut(clk,core,tdqss,tac,twd,sysn,ddrn,cke,ena,code,sel,address,data);'
 assert tb.count(anchor)==1;tb=tb.replace(anchor,replacement)
 (root/'tb.sv').write_text(tb);shutil.copy2(PROOF/'cnn_apb_control.sv',root/'cnn_apb_control.sv')
 models=[]
 for kind in ex['cells']:
  if kind=='assign':continue
  f=EF/'sim_models/verilog'/(kind.lower()+'.v');shutil.copy2(f,root/f.name);models.append(root/f.name)
 inputs=dict(original_map=str(net),original_map_sha256=sha(net),files={f.name:sha(f) for f in root.iterdir() if f.is_file()},sync_audit=m,scope=ex['scope'],hardware_access=False)
 (root/'inputs.json').write_text(json.dumps(inputs,indent=2));os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 run([Q/'vlib.exe','work'],root,'vlib.log',15)
 run([Q/'vlog.exe','-sv','-timescale','1ns/1ps',*models,'mapped_control.v','cnn_apb_control.sv','tb.sv'],root,'compile.log',30)
 log,seconds=run([Q/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',90)
 assert 'PASS DDR_OBSERVER clocks=4 phases=8 stop=1 resets=2 readonly_words=10' in log
 for n,h in inputs['files'].items():assert sha(root/n)==h
 result=dict(status='DDR_OBSERVER_ACTUAL_MAP_SHORT_PASS',scope='All mapped observer state with frozen RTL read mux; mapped APB integration is a separate control-startup test',seconds=seconds,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),hardware_access=False,board_release=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
