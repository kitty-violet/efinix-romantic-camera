"""Bounded strict DDR turnaround matrix, with unchanged failed-board controller.

No whole-board run, forced calibration, PHY editing or relaxed device checks.
Pin-level first-failure evidence is independent of the Micron error message.
"""
import argparse,json,os,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,sha

PIN_CHECK = '''
integer turn_ck=0,turn_last_write=-1000000,turn_write_bank=-1,turn_errors=0;
integer turn_required;realtime turn_write_time;
always @(negedge clk)begin
 turn_ck=turn_ck+1;
 if(cal_done && memory_reset===1 && cke===1 && cs===0)begin
  if({ras,cas,we}===3'b100)begin
   turn_last_write=turn_ck;turn_write_time=$realtime;turn_write_bank=ba;
  end
  if({ras,cas,we}===3'b101)begin
   turn_required=ddr3md.write_latency+ddr3md.burst_length/2+ddr3md.TWTR_TCK-ddr3md.additive_latency;
   if(turn_ck-turn_last_write<turn_required)begin
    turn_errors=turn_errors+1;
    $display("TURN_PIN_FAIL case=%0d wb=%0d rb=%0d write_ns=%0.3f read_ns=%0.3f delta_ck=%0d required_ck=%0d WL=%0d BL=%0d AL=%0d",
     u_top.genblk1.memcheck_0.case_number,turn_write_bank,ba,turn_write_time/10.0,$realtime/10.0,
     turn_ck-turn_last_write,turn_required,ddr3md.write_latency,ddr3md.burst_length,ddr3md.additive_latency);$fflush();
    #1;$fatal(1,"TURN real pin WRITE-to-READ interval too short");
   end
  end
 end
end
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
 p.add_argument('--traffic',choices=('matrix','dense','combined'),default='matrix')
 p.add_argument('--layouts',type=int,choices=(1,3),default=1)
 p.add_argument('--serial-turns',action='store_true')
 p.add_argument('--rom-candidate',type=Path);p.add_argument('--axi-bridge',type=Path);a=p.parse_args()
 expected=(336 if a.traffic!='dense' else 0)+(32*a.layouts if a.traffic!='matrix' else 0)+256*a.serial_turns
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 base=ROOT/'tmp/frame-execution/ddr516-concurrent1'
 old=json.loads((base/'concurrent_result.json').read_text());meta=json.loads((base/'inputs.json').read_text())
 assert old['status']=='DDR516_REAL_CONCURRENT_PASS' and old['inputs_sha256']==sha(base/'inputs.json')
 board=ROOT/'tmp/frame-execution/board-full-raw-credit1'
 bound=['ddr516_reference.sv','ddr516_axi_address_bridge.sv','ddr516_v4_controller.sv','ddr516_rx_retime.sv','ddr3_controller.bin','ddr3_controller.vh']
 candidate=None;guard=None
 if a.rom_candidate:
  guard=json.loads((a.rom_candidate/'write_read_guard.json').read_text())
  candidate=a.rom_candidate/'ip/ddr516_reference'
  assert guard['status']=='WR2RD_GUARD_STATIC_AUDIT_PASS'
  assert guard['old_rom_sha256']==sha(board/'ddr3_controller.bin')
  assert guard['new_rom_sha256']==sha(candidate/'ddr3_controller.bin')
  assert guard['new_header_sha256']==sha(candidate/'ddr3_controller.vh')
 root.mkdir();records=[]
 for row in meta['inputs']:
  src=Path(row['snapshot']);assert sha(src)==row['sha256'];dst=root/src.name;shutil.copy2(src,dst)
  if src.name in bound:shutil.copy2(board/src.name,dst)
  if a.axi_bridge and src.name=='ddr516_axi_address_bridge.sv':shutil.copy2(a.axi_bridge,dst)
  if candidate and src.name in ('ddr3_controller.bin','ddr3_controller.vh'):shutil.copy2(candidate/src.name,dst)
  if src.name=='memory_checker_axi.v':
   s=(HERE/'tb_ddr516_turnaround.sv').read_text()
   if a.traffic!='matrix':s=s.replace('MATRIX=1,DENSE=0','MATRIX='+str(int(a.traffic=='combined'))+',DENSE=1')
   s=s.replace('LAYOUTS=1',f'LAYOUTS={a.layouts}')
   if a.serial_turns:s=s.replace('SERIAL=0','SERIAL=1')
   dst.write_text(s)
  if src.name=='example_top.v':
   s=dst.read_text();old_count='measured_writes!=768 || measured_reads!=896 || measured_aw!=6 || measured_ar!=7'
   assert s.count(old_count)==1
   s=s.replace(old_count,'measured_writes!=memcheck_0.total_w || measured_reads!=memcheck_0.total_r || measured_aw!=memcheck_0.total_aw || measured_ar!=memcheck_0.total_ar')
   # The frozen board has no optional read-pipeline debug output. Remove only
   # the old fixture's progress observer; functional/calibration checks remain.
   observer='assign diag_read_pipe=inst_ddr3_axi.controller.diag_read_pipe;'
   assert s.count(observer)==1
   s=s.replace(observer,'')
   s,count=re.subn(r'always @\(diag_read_pipe\)begin.*?\$fflush\(\);\s*end', '',s,flags=re.S)
   assert count==1, 'Optional diagnostic observer template changed'
   # Exact same wrapper and address bridge as the failed whole-board snapshot.
   dst.write_text(s)
  if src.name=='tb.sv':
   s=dst.read_text();assert s.count('endmodule')==1
   s=s.replace('.MEM_BITS(14)','.MEM_BITS(16)')
   # Dense queue coverage may run longer after calibration; bounded at 20 ms.
   if a.traffic!='matrix':s=s.replace('#90000000;','#200000000;').replace('9000us timeout','20000us timeout')
   s=s.replace('$fdisplay(cmd_fd,"%0.3f,%b,%d,%h",','if(cal_done)$fdisplay(cmd_fd,"%0.3f,%b,%d,%h",')
   dst.write_text(s.replace('endmodule',PIN_CHECK+'\nendmodule'))
  origin=candidate/src.name if candidate and src.name in ('ddr3_controller.bin','ddr3_controller.vh') else board/src.name if src.name in bound else src
  if a.axi_bridge and src.name=='ddr516_axi_address_bridge.sv':origin=a.axi_bridge.resolve()
  records.append(dict(source=str(origin),snapshot=str(dst),sha256=sha(dst),base_sha256=row['sha256']))
 for name in bound:
  expected_source=candidate/name if candidate and name in ('ddr3_controller.bin','ddr3_controller.vh') else board/name
  if a.axi_bridge and name=='ddr516_axi_address_bridge.sv':expected_source=a.axi_bridge.resolve()
  assert sha(root/name)==sha(expected_source),f'Controller binding mismatch: {name}'
 data=dict(inputs=records,runner_sha256=sha(__file__),stimulus_sha256=sha(HERE/'tb_ddr516_turnaround.sv'),
  failed_board=str(board),bound_files=bound,full_board=False,calibration_forced=False,cases=expected,traffic=a.traffic,layouts=a.layouts,serial_turns=a.serial_turns,
  rom_candidate=str(a.rom_candidate.resolve()) if a.rom_candidate else None,rom_guard=guard,
  axi_bridge_candidate=str(a.axi_bridge.resolve()) if a.axi_bridge else None)
 (root/'inputs.json').write_text(json.dumps(data,indent=2))
 (root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
 sim=Path('E:/quatra/questa_fse/win64');env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 sources=[root/n for n in ['ddr3.v','efx_iddr.v','efx_oddr.v','efx_srl8.v','memory_checker_axi.v','ddr516_reference.sv','ddr516_axi_address_bridge.sv','ddr516_v4_controller.sv','example_top.v','ddr516_rx_retime.sv','EFX_GPIO_V3.v','EFX_PLL_V3_FAST_PROTOCOL.v','tb.sv']]
 commands=[('vlib',[sim/'vlib.exe','work']),('compile',[sim/'vlog.exe','-sv','-timescale','1ns/1ps','+define+den2048Mb+sg125+x16','+incdir+'+str(root),*sources]),
  ('simulation',[sim/'vsim.exe','-c','-voptargs=-O5','work.efx_ddr3_soft_controller_tb','-do','run.do'])]
 start=time.monotonic();stages=[];body=''
 for name,command in commands:
  with (root/(name+'.log')).open('wb') as out:
   proc=subprocess.Popen(list(map(str,command)),cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT)
   (root/'active_process.json').write_text(json.dumps(dict(pid=proc.pid,command=list(map(str,command)),stage=name,started=time.time())))
   ret=proc.wait()
  body=(root/(name+'.log')).read_text(errors='replace');stages.append(dict(stage=name,returncode=ret))
  if ret or re.search(r'\*\* (?:Fatal|Error)|\bERROR:',body):break
 errors=[line for line in body.splitlines() if re.search(r'\*\* (?:Fatal|Error)|\bERROR:|TURN_PIN_FAIL',line)]
 passed=len(stages)==3 and all(s['returncode']==0 for s in stages) and not errors and f'TURN_ALL_PASS cases={expected} ' in body and 'DDR3 Controller PASS' in body
 for row in records:assert sha(row['snapshot'])==row['sha256']
 result=dict(status='DDR516_TURNAROUND_PASS' if passed else 'DDR516_TURNAROUND_FAIL',seconds=time.monotonic()-start,stages=stages,errors=errors,
  inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/(stages[-1]['stage']+'.log')),passed_cases=len(re.findall(r'^# TURN_PASS ',body,re.M)),full_board=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 if not passed:raise SystemExit(1)
if __name__=='__main__':main()
