"""Quiesce both request and acceptance at an idle boundary, then test serial."""
import json,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once
import validate_control_ddr as base
import continuous_control_flow as flow

original=base.driver_run
def driver_run(source,code,root,args):
 if source.name=='check-mapped-jtag-regression.py':
  old=r"  force dut.\cpu/boot_valid =1'b0;"
  # This generated Python file contains a raw Verilog test template.
  if old not in code:old=old.replace('\\','\\\\')
  slash='\\\\' if '\\\\cpu' in old else '\\'
  new=r'''
  begin : quiet_entry
   integer settle;
   settle=0;
   @(negedge CLOCK);
   while((dut.BScpu/state_oh !==10'h001 || dut.BScpu/sequencer/waiting !==0) && settle<1024)begin
    @(negedge CLOCK);settle=settle+1;
   end
   if(settle==1024)$fatal(1,"quiet entry never drained");
  end
  force dut.BScpu/boot_ready =1'b0;
  force dut.BScpu/boot_valid =1'b0;'''.replace('BS',slash)
  code=once(code,old,new)
  release='  release dut.'+slash+'cpu/boot_valid ;'
  code=once(code,release,release+'\n  release dut.'+slash+'cpu/boot_ready ;')
  # A test pause must not manufacture a timeout in the independent producer.
  gate='  accepts=dut.'+slash+'cpu/observation_accepts ;replies=dut.'+slash+'cpu/observation_replies ;'
  # Only low4 fault bits are retained consumers in this mapped cone. The
  # optimized high4 are not extracted: checking the entire declared8-bit
  # net manufactured z0!=0. Functional fault/failed/capture checks remain.
  code=once(code,gate,gate+'\n  if(dut.'+slash+'cpu/boot_fault [3:0] !==0 || dut.'+slash+'cpu/boot_failed !==0)$fatal(1,"quiet entry already failed");')
  code=once(code,'  release dut.'+slash+'cpu/boot_valid ;',
   '  if(dut.'+slash+'cpu/boot_fault [3:0] !==0 || dut.'+slash+'cpu/boot_failed !==0)$fatal(1,"quiet pause created fault");\n  release dut.'+slash+'cpu/boot_valid ;')
  receipt=dict(status='TEST_QUIET_HANDSHAKE_DRAIN_AND_BOTH_SIDES_PAUSED',previous_failure='serial1 one-sided forced request produces fault8; serial2 added full-width assertion falsely read omitted high4 fault bits as z. trace2 shows low4=0 and normal polling until quiesced',
   checks_preserved=['three quiet windows of4096cycles','paired command reads','snapshot causes','reset recovery','accepted/reply counts1,2,4'],
   hardware_source_changed=False,evidence={str(Path(__file__)):sha(__file__),str(ROOT/'tmp/frame-execution/continuous-control-evidence1-serial/new/tb.sv'):sha(ROOT/'tmp/frame-execution/continuous-control-evidence1-serial/new/tb.sv'),str(ROOT/'tmp/frame-execution/continuous-control-evidence1-serial/new/simulation.log'):sha(ROOT/'tmp/frame-execution/continuous-control-evidence1-serial/new/simulation.log'),str(ROOT/'tmp/frame-execution/continuous-serial-trace2/inputs.json'):sha(ROOT/'tmp/frame-execution/continuous-serial-trace2/inputs.json'),str(ROOT/'tmp/frame-execution/continuous-serial-trace2/simulation.log'):sha(ROOT/'tmp/frame-execution/continuous-serial-trace2/simulation.log')})
  out=root.parent/(root.name+'-quiet-stimulus.json');assert not out.exists();out.write_text(json.dumps(receipt,indent=2))
 return original(source,code,root,args)
base.driver_run=driver_run
if __name__=='__main__':flow.main()
