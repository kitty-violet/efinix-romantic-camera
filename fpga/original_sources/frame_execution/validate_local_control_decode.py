"""Four-state exact-decode and cycle-by-cycle controller equivalence short gate."""
import argparse,json,re,shutil,os
from pathlib import Path
from run_units import ROOT,EF,MS,sha,run
from local_control_decode import BASE,audit_delta,decoder
MS=Path('E:/quatra/questa_fse/win64')

PORTS={'clk':1,'reset':1,'host_valid':1,'boot_valid':1,'host_write':1,'boot_write':1,
 'host_address':32,'boot_address':32,'host_data':32,'boot_data':32,'host_size':2,
 'permit':1,'io_apbSlave_0_PREADY':1,'io_apbSlave_0_PRDATA':32,'io_apbSlave_0_PSLVERROR':1,
 'axiA_awready':1,'axiA_wready':1,'axiA_bvalid':1,'axiA_bid':8,'axiA_bresp':2,
 'axiA_arready':1,'axiA_rvalid':1,'axiA_rdata':32,'axiA_rid':8,'axiA_rresp':2,'axiA_rlast':1}
OUTPUTS={'host_ready':1,'boot_ready':1,'host_rsp_valid':1,'host_rsp_error':1,'host_rsp_data':32,
 'boot_reply':1,'boot_error':1,'boot_run_allowed':1,
 'io_apbSlave_0_PADDR':16,'io_apbSlave_0_PSEL':1,'io_apbSlave_0_PENABLE':1,
 'io_apbSlave_0_PWRITE':1,'io_apbSlave_0_PWDATA':32,
 'axiA_awvalid':1,'axiA_wvalid':1,'axiA_awaddr':32,'axiA_wdata':32,'axiA_awid':8,
 'axiA_awregion':4,'axiA_awlen':8,'axiA_awsize':3,'axiA_awburst':2,'axiA_awlock':1,
 'axiA_awcache':4,'axiA_awqos':4,'axiA_awprot':3,'axiA_wstrb':4,'axiA_wlast':1,
 'axiA_bready':1,'axiA_arvalid':1,'axiA_araddr':32,'axiA_arid':8,'axiA_arregion':4,
 'axiA_arlen':8,'axiA_arsize':3,'axiA_arburst':2,'axiA_arlock':1,'axiA_arcache':4,
 'axiA_arqos':4,'axiA_arprot':3,'axiA_rready':1}

def width(n):return '' if n==1 else f'[{n-1}:0] '

def core(text,name):
    start=text.index(' // Fixed one-hot control;');end=text.index(' // Read-only, configuration-initialized evidence',start)
    ports=[f'input wire {width(w)}{n}' for n,w in PORTS.items()]+[f'output wire {width(w)}{n}' for n,w in OUTPUTS.items()]
    return 'module '+name+'('+','.join(ports)+');\n reg [31:0] reply_data;\n'+text[start:end]+'\nendmodule\n'

def main():
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    board=a.board.resolve();root=a.root.resolve();audit_delta(board)
    assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    (root/'cores.sv').write_text(core((BASE/'assembly/EfxSapphireSoc.v').read_text(),'original')+
        core((board/'assembly/EfxSapphireSoc.v').read_text(),'candidate')+decoder())
    lines=['`timescale 1ns/1ps','module tb;','reg [9:0] value;wire [9:0] matched;',
     'control_exact_state_decode decode(.value(value),.matched(matched));',
     *[f'reg {width(w)}{n}=0;' for n,w in PORTS.items()]]
    for side in ('o','n'):
        lines.extend(f'wire {width(w)}{side}_{n};' for n,w in OUTPUTS.items())
        pins=[f'.{n}({n})' for n in PORTS]+[f'.{n}({side}_{n})' for n in OUTPUTS]
        lines.append(('original' if side=='o' else 'candidate')+' '+side+'('+','.join(pins)+');')
    lines+=['always #5 clk=~clk;','integer i,j,k,checks=0,cycles=0;reg [9:0] original_value;',
      'reg [9:0] expected_value;integer bit_index;\n'
      'task check_decode;begin #1;expected_value=value;'
      'for(bit_index=0;bit_index<10;bit_index=bit_index+1) if(value[bit_index]===1\'bz) expected_value[bit_index]=0;'
      'for(k=0;k<10;k=k+1) if(matched[k] !== (expected_value == (10\'b1<<k))) $fatal(1,"decode value=%h bit=%0d got=%b",value,k,matched[k]);checks=checks+1;end endtask',
      'task compare;begin #1;']
    for name in OUTPUTS:
        lines.append(f'if(o_{name} !== n_{name}) $fatal(1,"output mismatch {name} cycle=%0d old=%h new=%h",cycles,o_{name},n_{name});')
    for name in ('state_oh','host_owner','last_host','writing','word_access','aw_sent','w_sent','address','payload','reply_data','reply_error','fatal_bus_error','bus_age','mailbox_read'):
        lines.append(f'if(o.{name} !== n.{name}) $fatal(1,"internal mismatch {name} cycle=%0d",cycles);')
    lines+=['end endtask',r'''
reg [31:0] rng=32'h69c10a55;
task randomize_inputs;begin
 rng=rng^(rng<<13);rng=rng^(rng>>17);rng=rng^(rng<<5);
 host_valid=rng[0];boot_valid=rng[1];host_write=rng[2];boot_write=rng[3];host_size=rng[5:4];
 host_data=rng;boot_data=~rng;permit=rng[6];
 case(rng[8:7])
 0:begin host_address=32'hf9000f00+{26'd0,rng[12:9],2'b0};boot_address=host_address;end
 1:begin host_address=32'hf8100000+{26'd0,rng[12:9],2'b0};boot_address=host_address;end
 2:begin host_address=32'he1000000+{24'd0,rng[14:9],2'b0};boot_address=host_address;end
 3:begin host_address=rng;boot_address=rng;end
 endcase
 io_apbSlave_0_PREADY=rng[15];io_apbSlave_0_PRDATA=rng;io_apbSlave_0_PSLVERROR=rng[16];
 axiA_awready=rng[17];axiA_wready=rng[18];axiA_bvalid=rng[19];axiA_bid=rng[20]?8'd0:8'd3;axiA_bresp=rng[22:21];
 axiA_arready=rng[23];axiA_rvalid=rng[24];axiA_rdata=rng;axiA_rid=rng[25]?8'd0:8'd4;axiA_rresp=rng[27:26];axiA_rlast=rng[28];
end endtask
initial begin
 reset=1;
 // All binary states and X at every bit: exact equality. Official EFX_LUT4
 // has weak-zero input defaults, so a Z at its pin resolves to 0. Test that
 // separately; driven state FFs cannot produce Z. Do not claim RTL Z equality.
 for(i=0;i<1024;i=i+1)begin
  value=i;check_decode;
  for(j=0;j<10;j=j+1)begin value=i;value[j]=1'bx;check_decode;value[j]=1'bz;check_decode;end
 end
 @(negedge clk);reset=0;
 // Equivalent data, errors, AW/W independent backpressure, arbitration,
 // reset interruption and recovery. No new latency is allowed.
 repeat(40000)begin
  @(negedge clk);randomize_inputs;reset=(cycles%997)==0;compare;
  @(posedge clk);compare;cycles=cycles+1;
 end
 // Every binary illegal/valid state at identical payloads. Check outputs,
 // then release before the next clock and compare recovery/transition.
 for(i=0;i<1024;i=i+1)begin
  @(negedge clk);randomize_inputs;reset=0;
  force o.state_oh=i;force n.state_oh=i;compare;
  release o.state_oh;release n.state_oh;
  @(posedge clk);compare;cycles=cycles+1;
 end
 // Force an APB wait beyond the existing fatal timeout; retain the owner.
 @(negedge clk);reset=1;@(negedge clk);reset=0;
 host_valid=0;boot_valid=1;boot_address=32'hf8100000;boot_write=0;io_apbSlave_0_PREADY=0;
 repeat(8200)begin @(negedge clk);compare;cycles=cycles+1;end
 if(!n.fatal_bus_error || n.state_oh!==10'h010)$fatal(1,"timeout must retain APB owner");
 io_apbSlave_0_PREADY=1;repeat(10)begin @(negedge clk);compare;end
 $display("PASS LOCAL_DECODE checks=%0d differential_cycles=%0d illegal_states=1024 timeout=8200",checks,cycles);$finish;
end
initial begin #1000000;$fatal(1,"bounded test timeout");end
endmodule
''']
    (root/'tb.sv').write_text('\n'.join(lines));primitive=EF/'sim_models/verilog/efx_lut4.v';shutil.copy2(primitive,root/primitive.name)
    files={n:sha(root/n) for n in ('cores.sv','tb.sv','efx_lut4.v')}
    (root/'inputs.json').write_text(json.dumps(dict(board_inputs_sha256=sha(board/'inputs.json'),files=files,runner_sha256=sha(__file__)),indent=2))
    os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
    run([MS/'vlib.exe','work'],root,'vlib.log',60)
    run([MS/'vlog.exe','-sv','efx_lut4.v','cores.sv','tb.sv'],root,'compile.log',60)
    log,seconds=run([MS/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',120)
    found=re.findall(r'^# (PASS LOCAL_DECODE.*)$',log,re.M);assert len(found)==1
    assert all(sha(root/n)==h for n,h in files.items());audit_delta(board)
    result=dict(status='LOCAL_EXACT_DECODE_FOURSTATE_AND_CONTROLLER_EQUIVALENCE_PASS',line=found[0],seconds=seconds,
        inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),hardware_access=False)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
