"""Use the actual modified public decoder; removed windows must return ERROR."""
import argparse, json, re, shutil
from pathlib import Path
from run_units import ROOT, HERE, MS, run, sha
from prune_unused_sapphire import prune

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--reference-board',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
a=p.parse_args();r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir(parents=True)
reference=a.reference_board/'assembly/EfxSapphireSoc.v';candidate=a.candidate.resolve()
text=candidate.read_text();assert text==prune(reference.read_text())
module=re.search(r'(?ms)^module BmbDecoder_3 \(.*?^endmodule',text).group()
(r/'decoder.v').write_text(module+'\n')
header=module.split(');',1)[0]
ports=re.findall(r'\b(input|output)\s+(?:wire|reg)\s*(\[[^]]+\])?\s*(\w+)',header)
decl=[];links=[]
for direction,width,name in ports:
 decl.append(('reg ' if direction=='input' else 'wire ')+(width or '')+' '+name+('=0' if direction=='input' else '')+';')
 links.append('.'+name+'('+name+')')
tb='module tb_unused_decoder;\n'+'\n'.join(decl)+'\nBmbDecoder_3 dut('+','.join(links)+');\n'
tb+='always #5 io_systemClk=~io_systemClk;\ninteger ticks=0,checks=0;\n'
tb+='always @(posedge io_systemClk)begin ticks=ticks+1;if(ticks>50000)$fatal(1,"decoder watchdog");\n'
tb+='if(!systemCd_logic_outputReset && (io_outputs_2_cmd_valid || io_outputs_3_cmd_valid || io_outputs_4_cmd_valid))$fatal(1,"removed window routed");end\n'
tb+='''task unmapped(input [23:0] address,input bit write_cmd,input [3:0] context_id,input integer stall);
 begin
  @(negedge io_systemClk);io_input_cmd_valid=1;io_input_cmd_payload_last=1;
  io_input_cmd_payload_fragment_address=address;io_input_cmd_payload_fragment_opcode=write_cmd;
  io_input_cmd_payload_fragment_context=context_id;io_input_cmd_payload_fragment_data=32'ha135b240^address;
  io_input_cmd_payload_fragment_mask=4'hf;io_input_cmd_payload_fragment_length=3;io_input_rsp_ready=0;
  do @(posedge io_systemClk);while(!io_input_cmd_ready);
  @(negedge io_systemClk);io_input_cmd_valid=0;
  while(!io_input_rsp_valid)@(negedge io_systemClk);
  repeat(stall+1)begin
   if({io_input_rsp_valid,io_input_rsp_payload_last,io_input_rsp_payload_fragment_opcode,io_input_rsp_payload_fragment_context}!=={3'b111,context_id})$fatal(1,"error response/context/hold");
   @(negedge io_systemClk);
  end
  io_input_rsp_ready=1;@(negedge io_systemClk);io_input_rsp_ready=0;
  repeat(2)@(negedge io_systemClk);if(io_input_rsp_valid)$fatal(1,"duplicate error response");checks=checks+1;
 end
endtask
initial begin
 systemCd_logic_outputReset=1;repeat(4)@(negedge io_systemClk);systemCd_logic_outputReset=0;
 for(integer n=0;n<512;n=n+1)begin
  unmapped(24'h001000+24'(n%64),n[0],n[3:0],n%7);
  unmapped(24'h006000+24'((n*73)%4096),!n[0],4'(n+3),n%11);
  unmapped(24'h00a000+24'(n%256),n[1],4'(n+7),n%5);
 end
 // Withdraw a pending no-hit response by reset, then prove clean recovery.
 io_input_cmd_payload_fragment_address=24'h001000;io_input_cmd_valid=1;
 do @(posedge io_systemClk);while(!io_input_cmd_ready);
 @(negedge io_systemClk);io_input_cmd_valid=0;
 wait(io_input_rsp_valid);@(negedge io_systemClk);systemCd_logic_outputReset=1;
 repeat(3)@(negedge io_systemClk);systemCd_logic_outputReset=0;
 repeat(3)@(negedge io_systemClk);if(io_input_rsp_valid)$fatal(1,"stale response after reset");
 unmapped(24'h006fff,0,4'hd,5);
 $display("PASS UNUSED_PERIPHERAL_DECODER errors=%0d reset_recovery=1",checks);$finish;
end
endmodule
'''
(r/'tb_unused_decoder.sv').write_text(tb)
shutil.copy2(candidate,r/'EfxSapphireSoc.v');shutil.copy2(__file__,r/'frozen_runner.py')
records=[dict(snapshot=str(r/n),sha256=sha(r/n)) for n in ['EfxSapphireSoc.v','decoder.v','tb_unused_decoder.sv','frozen_runner.py']]
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,reference_sha256=sha(reference),transform_sha256=sha(HERE/'prune_unused_sapphire.py')),indent=2))
run([MS/'vlib.exe','work'],r,'vlib.log')
body,_=run([MS/'vlog.exe','-sv',r/'decoder.v',r/'tb_unused_decoder.sv'],r,'compile.log');assert 'Errors: 0, Warnings: 0' in body
body,secs=run([MS/'vsim.exe','-c','work.tb_unused_decoder','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log')
lines=re.findall(r'^# (PASS UNUSED_PERIPHERAL_DECODER.*)$',body,re.M)
assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
for x in records:assert sha(x['snapshot'])==x['sha256']
result=dict(status='UNUSED_PERIPHERAL_DECODER_FOUR_STATE_PASS',line=lines[0],seconds=secs,
 inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),candidate_sha256=sha(candidate),full_board=False)
(r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
