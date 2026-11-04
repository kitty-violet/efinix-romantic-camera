"""Actual external JTAG + fixed control plane + board APB/store/AXI adapter.

Only DDR responses are a stalled testbench memory. No CPU injection, force,
preloaded DDR or direct APB stimulus. Public vendor JTAG/CDC is copied verbatim.
"""
import argparse,json,os,re,shutil,subprocess,time,traceback,zlib
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,sha,run
from validate_real_boot import make_tb

def public_jtag(source):
 text=source.read_text();parts=[]
 for name in ('JtagBridgeNoTap','SystemDebugger','FlowCCUnsafeByToggle','BufferCC'):
  matches=re.findall(r'^module '+name+r'\b.*?^endmodule',text,re.M|re.S)
  assert len(matches)==1,name
  parts.append(matches[0])
 return '\n\n'.join(parts)+'\n'

def testbench():
 t=make_tb().replace('EfxSapphireSoc dut (','cnn_fixed_control_soc dut (.boot_run_allowed(boot_run_allowed),')
 t=t.replace('wire sys_reset;','wire sys_reset,boot_run_allowed,apb_enable;')
 t=t.replace('.enable(enable)','.enable(apb_enable)')
 t=t.replace('cnn_fixed_control_soc dut (','assign enable=apb_enable && boot_run_allowed;\ncnn_fixed_control_soc dut (')
 for before,after in [('dut.system_cores_0_logic_cpu_dBus_cmd_payload_address','dut.boot_address'),
  ('dut.system_cores_0_logic_cpu_dBus_cmd_payload_data','dut.boot_data'),
  ('dut.system_cores_0_logic_cpu_dBus_cmd_valid','dut.boot_valid'),
  ('dut.system_cores_0_dBus_cmd_ready','dut.boot_ready'),
  ('dut.system_cores_0_logic_cpu_dBus_cmd_payload_wr','dut.boot_write')]:t=t.replace(before,after)
 t=t.replace('"jtag_memory_tasks.svh"','"jtag_direct_tasks.svh"')
 t=t.replace('actual_cpu=1','fixed_controller=1')
 t=t.replace('host_halted=0;host_address_valid=0;host_data_valid=0;','')
 t=t.replace('jresume();','')
 t=t.replace('PASS REAL_BOOT_', 'PASS FIXED_SOC_')
 t=t.replace('if(descriptor_probe0[31:0]', '''jreject(1,32'hf8100008,2); // Host cannot bypass the verified run gate.
 jreject(0,32'h10b80000,2); // Removed instruction-injection address is explicit error.
 jreject(1,32'he0000000,2);jreject(0,32'he2000000,2);
 jreject(1,32'he1200001,2);jreject(0,32'he1200000,1);
 if(descriptor_probe0[31:0]''')
 # A common reset must also reset the real APB and coefficient eligibility.
 t=t.replace('.clk(clk),.reset(reset)', '.clk(clk),.reset(reset||sys_reset)')
 return t

DIRECT_TASKS='''
task jtransaction(input bit wr,input [31:0] address,input [31:0] value,input [1:0] size,
 output [31:0] result,output bit error);
 reg [74:0] command;reg [33:0] response;integer attempts;
 begin
  command={size,wr,value,address,8'd0};jscan_start(0);
  for(integer k=0;k<75;k=k+1)begin jtdi=command[k];jclock();end
  jscan_end();response=0;attempts=0;
  while(response[0]!==1'b1 && attempts<100)begin
   repeat(16)@(negedge clk);jscan_start(1);
   for(integer k=0;k<34;k=k+1)begin response[k]=jtdo;jtdi=0;jclock();end
   jscan_end();attempts=attempts+1;
  end
  if(response[0]!==1'b1 || $isunknown(response))$fatal(1,"JTAG no response addr=%h",address);
  result=response[33:2];error=response[1];
 end
endtask
task jwrite(input [31:0] address,input [31:0] value);
 reg [31:0] got;bit error;
 begin jtransaction(1,address,value,2,got,error);if(error)$fatal(1,"JTAG write error %h",address);end
endtask
task jexpect(input [31:0] address,input [31:0] expected);
 reg [31:0] got;bit error;
 begin jtransaction(0,address,0,2,got,error);if(error || got!==expected)$fatal(1,"JTAG read %h got=%h expected=%h",address,got,expected);end
endtask
task jreject(input bit wr,input [31:0] address,input [1:0] size);
 reg [31:0] got;bit error;
 begin jtransaction(wr,address,32'hbadbad00,size,got,error);if(!error)$fatal(1,"JTAG illegal request accepted %h",address);end
endtask
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
 p.add_argument('--board',type=Path,default=ROOT/'tmp/frame-execution/connected-board-fault-boundary1')
 p.add_argument('--coefficient-vectors',type=Path,default=ROOT/'tmp/frame-execution/joint-reorder-vectors1')
 p.add_argument('--accepted-a',action='store_true')
 p.add_argument('--host-codec-smoke',action='store_true')
 p.add_argument('--smoke',action='store_true');a=p.parse_args()
 r=a.root.resolve();board=a.board.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
 v=HERE/'variants/fixed_boot_rom';records=[];sources=[]
 def copy(f,compile=True):
  d=r/f.name;assert not d.exists();shutil.copy2(f,d);records.append(dict(source=str(f),snapshot=str(d),sha256=sha(f)))
  if compile:sources.append(d)
 try:
  for f in (v/'cnn_fixed_control_soc.sv',v/'cnn_boot_sequencer.sv',EF/'sim_models/verilog/efx_ram10.v'):copy(f)
  for n in ('resident_weight_bank64.sv','coefficient_bank_store_fast.sv','cnn_apb_control.sv','sapphire_axi32_to_axi128.sv'):copy(board/'assembly'/n)
  vendor=board/'assembly/EfxSapphireSoc.v';jtag=r/'official_jtag.sv';jtag.write_text(public_jtag(vendor));sources.append(jtag)
  tasks=(HERE/'board_integration/jtag_memory_tasks.svh').read_text().split('task jdebug(')[0]+DIRECT_TASKS
  (r/'jtag_direct_tasks.svh').write_text(tasks)
  vector_path=a.coefficient_vectors.resolve()/'load_words.hex'
  words=[int(x,16) for x in vector_path.read_text().split()];assert len(words)==2560
  packed=b''.join(w.to_bytes(8,'little') for w in words);crc=zlib.crc32(packed)
  if a.accepted_a:
   from accepted_a_assets import accepted_package
   assert packed==accepted_package()[0]
  vector=f"localparam [31:0] EXPECTED_CRC=32'h{crc:08x};\nfunction automatic [31:0] upload_word(input integer i);\nbegin case(i)\n"
  vector+='\n'.join(f"{i}:upload_word=32'h{int.from_bytes(packed[i*4:i*4+4],'little'):08x};" for i in range(5120))
  vector+='\ndefault:upload_word=32\'hx;endcase end endfunction\n';(r/'boot_vectors.vh').write_text(vector)
  tb_text=testbench()
  if a.host_codec_smoke:
   assert a.smoke
   from fixed_jtag_loader import command_scan
   transactions=[]
   for address,value in ((0xe1000000,0x81234567),(0xe1100000,0x89abcdef),(0xe1200000,0x80000001)):
    transactions.extend(((True,address,value,0),(False,address,0,value)))
   packets=[command_scan(w,addr,value)[0] for w,addr,value,_ in transactions]
   body='''
task host_codec(input [76:0] packet,input bit writing,input [31:0] expected);
 reg [33:0] response;integer attempt;
 begin
  jscan_start(packet[1:0]);
  for(integer k=2;k<77;k=k+1)begin jtdi=packet[k];jclock();end
  jscan_end();response=0;attempt=0;
  while(response[0]!==1'b1 && attempt<100)begin
   repeat(16)@(negedge clk);jscan_start(1);
   for(integer k=0;k<34;k=k+1)begin response[k]=jtdo;jtdi=0;jclock();end
   jscan_end();attempt=attempt+1;
  end
  if(response[1:0]!==2'b01 || $isunknown(response))$fatal(1,"host codec invalid response");
  if(!writing && response[33:2]!==expected)$fatal(1,"host codec data mismatch");
 end
endtask
'''
   tb_text=tb_text.replace('task reboot;',body+'\ntask reboot;')
   calls='\n'.join(f" host_codec(77'h{packet:x},1'b{int(row[0])},32'h{row[3]:08x});" for packet,row in zip(packets,transactions))
   calls+='\n $display("PASS PYTHON_HOST_CODEC actual_jtag=1 transfers=6 full_descriptor_payload=0");\n'
   tb_text=tb_text.replace(" reboot();\n jwrite(32'he1000000",' reboot();\n'+calls+" jwrite(32'he1000000",1)
  tb=r/'tb_real_boot.sv';tb.write_text(tb_text);sources.append(tb)
  generated={f.name:sha(f) for f in (jtag,tb,r/'boot_vectors.vh',r/'jtag_direct_tasks.svh')}
  (r/'inputs.json').write_text(json.dumps(dict(inputs=records,generated=generated,runner_sha256=sha(__file__),vendor_source=str(vendor),vendor_sha256=sha(vendor),smoke=a.smoke,coefficient_vectors=str(vector_path),coefficient_vectors_sha256=sha(vector_path),accepted_a=a.accepted_a,coefficient_crc32=f'{crc:08x}'),indent=2))
  run([MS/'vlib.exe','work'],r,'vlib.log',timeout=30)
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*sources],r,'compile.log',timeout=45)
  assert 'Errors: 0' in body
  scratch=r/'temp';scratch.mkdir();env=os.environ.copy();env.update(TEMP=str(scratch),TMP=str(scratch))
  started=time.monotonic()
  with (r/'simulation.log').open('w') as log:
   proc=subprocess.run([str(MS/'vsim.exe'),'-c','work.tb_real_boot','+SMOKE='+str(int(a.smoke)),'-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
  body=(r/'simulation.log').read_text();assert proc.returncode==0 and not re.search(r'(?m)^#?\s*(?:FAIL|\*\* (?:Error|Fatal))',body)
  marker='PASS FIXED_SOC_SMOKE' if a.smoke else 'PASS FIXED_SOC_COMPLETE';assert marker in body
  for x in records:assert sha(x['source'])==sha(x['snapshot'])==x['sha256']
  for n,h in generated.items():assert sha(r/n)==h
  result=dict(status=marker.replace(' ','_'),lines=re.findall(r'^# (PASS .*)$',body,re.M),seconds=time.monotonic()-started,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),actual_external_jtag=True,actual_apb_store_adapter=True,ddr_phy=False,full_board=False,board_release=False,accepted_a=a.accepted_a,coefficient_crc32=f'{crc:08x}')
  (r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (r/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise

if __name__=='__main__':main()
