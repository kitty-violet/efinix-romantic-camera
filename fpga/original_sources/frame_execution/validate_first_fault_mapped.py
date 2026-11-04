"""Reuse the actual full-map startup cut and additionally verify mapped capture."""
import argparse,json,re,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from first_fault_observer import audit_delta

def main():
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    board=a.board.resolve();root=a.root.resolve();audit_delta(board)
    source=HERE/'validate_global_mapped_startup.py';code=source.read_text()
    # Synthesis chooses the public observer/output aliases for these registers.
    # Read the actual retained nets, never synthesize different logic for a TB.
    code=code.replace(r'\cpu/sequencer/state ',r'\cpu/boot_observer_state ')
    # valid is merged with the captured always-ready APB flag. Verify the real
    # serial bit127 (=64+63) still consumes snapshot112 before using that alias.
    net=(board/'outflow/ti60f225_oob.map.v').read_text()
    prefix=r'\cpu/system_hardJtag_debug_logic_jtagBridge/observation/'
    ff=re.search(r'EFX_FF\s+'+re.escape(prefix+'shifter[127]~FF')+r'\s+\((.*?)\) /\*',net,re.S)
    assert ff
    data=re.search(r'\.D\(([^()]+)\)',ff[1])[1]
    driver=next(m for m in re.finditer(r'EFX_LUT4\s+\S+\s+\((.*?)\) /\*',net,re.S) if '.O('+data+')' in m[1])
    assert prefix+'snapshot [112]' in driver[1]
    marker="lines.append('initial begin')"
    extra=r'''
task check_first_fault;
 reg [63:0] saved;
 begin
  if(dut.\cpu/observation_payload [112]!==0)$fatal(1,"normal startup falsely captured");
  @(negedge DUT_CLOCK);force dut.\cpu/state_oh =10'h000;
  @(negedge DUT_CLOCK);
  if(dut.\cpu/observation_payload [112]!==1 || dut.\cpu/observation_payload [61:58]!==4'b0001 ||
     dut.\cpu/observation_payload [41:32]!==0)$fatal(1,"mapped first fault incorrect");
  saved={dut.\cpu/observation_payload [112],dut.\cpu/observation_payload [62:0]};
  release dut.\cpu/state_oh ;
  repeat(100)begin @(negedge DUT_CLOCK);if({dut.\cpu/observation_payload [112],dut.\cpu/observation_payload [62:0]}!==saved)$fatal(1,"mapped record overwritten");end
  DUT_RESET=0;#2000;DUT_RESET=1;#4000;
  check_wait_loop();
  if(dut.\cpu/observation_payload [112]!==0)$fatal(1,"mapped record did not reset");
  $display("PASS ACTUAL_MAPPED_FIRST_FAULT_CAPTURE");
 end
endtask
'''
    assert code.count(marker)==1
    code=code.replace(marker,"lines.append("+repr(extra)+".replace('DUT_CLOCK',clk).replace('DUT_RESET',reset))\n    "+marker)
    marker="lines.extend(['$display(\"PASS GLOBAL_MAPPED_STARTUP resets=3\");$finish;end',"
    assert code.count(marker)==1
    code=code.replace(marker,"lines.extend(['check_first_fault();$display(\"PASS GLOBAL_MAPPED_STARTUP resets=3\");$finish;end',")
    code=code.replace("status='GLOBAL_MAPPED_STARTUP_THREE_RESETS_PASS'","status='FIRST_FAULT_MAP_STARTUP_AND_CAPTURE_PASS'")
    generated=root.parent/(root.name+'-runner.py');assert not generated.exists() and not root.exists()
    generated.write_text(code,encoding='utf-8')
    generated.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(generated),
        wrapper_sha256=sha(__file__),scope='Same exact mapped control/APB cut; three startup resets plus first illegal-state capture, frozen payload and clear on reset.'),indent=2))
    sys.argv=[str(generated),'--board',str(board),'--root',str(root)];runpy.run_path(str(generated),run_name='__main__')
    assert 'PASS ACTUAL_MAPPED_FIRST_FAULT_CAPTURE' in (root/'simulation.log').read_text()
if __name__=='__main__':main()
