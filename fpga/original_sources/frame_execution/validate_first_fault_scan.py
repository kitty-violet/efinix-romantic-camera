"""Actual first-fault payload through the retained JTAG snapshot CDC and host decoder."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,sha,run
from first_fault_observer import BASE,MODULE,transform_observer
from read_first_fault import decode

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve()
    assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
    shutil.copy2(MODULE,r/MODULE.name)
    (r/'jtag_bringup_snapshot.sv').write_text(transform_observer((BASE/'assembly/jtag_bringup_snapshot.sv').read_text()))
    tb=r'''`timescale 1ns/1ps
module tb;
reg clk=0,run_clock=1,reset=1,tck=0,request=0,poll=0,shift=0;
reg [9:0] state=1;reg [31:0] address=0,flags=0;reg [12:0] age=0;
wire [191:0] record_data;wire tdo;
always #3.333 if(run_clock)clk=~clk;
control_first_fault capture(clk,reset,state,6'd3,address,flags,age,1'b0,1'b0,record_data);
jtag_bringup_snapshot snapshot(clk,tck,request,poll,shift,record_data,8'h3f,tdo);
task edge_tck;begin #23;tck=1;#29;tck=0;end endtask
task scan(input req,output [255:0] data);begin
 request=req;poll=!req;shift=0;edge_tck();request=0;poll=0;shift=1;
 for(integer b=0;b<256;b=b+1)begin data[b]=tdo;edge_tck();end
 shift=0;edge_tck();if(data[31:0]!==32'h4f465331)$fatal(1,"signature");
end endtask
reg [255:0] data;reg [191:0] frozen;
initial begin
 repeat(10)@(negedge clk);reset=0;
 scan(1,data);scan(0,data);
 if(data[33:32]!==1 || data[127:126]!==2'b01)$fatal(1,"not armed");
 $display("OFSRAW %h",{data,2'b00});
 @(negedge clk);state=2;address=32'hf9000f00;flags=32'h00010000;
 @(negedge clk);state=0;address=32'hf8100040;flags=32'h00200000;
 @(negedge clk);state=1;frozen=record_data;
 scan(1,data);scan(0,data);
 if(data[255:64]!==frozen || data[33:32]!==1)$fatal(1,"capture corrupt");
 $display("OFSRAW %h",{data,2'b00});
 repeat(3)begin
  scan(1,data);scan(0,data);if(data[255:64]!==frozen)$fatal(1,"frozen mismatch");
  $display("OFSRAW %h",{data,2'b00});
 end
 @(negedge clk);run_clock=0;
 scan(1,data);scan(0,data);if(data[33:32]!==3)$fatal(1,"busy when clock stopped");
 $display("OFSRAW %h",{data,2'b00});
 run_clock=1;scan(0,data);scan(0,data);
 if(data[33:32]!==1 || data[255:64]!==frozen)$fatal(1,"clock recovery");
 $display("OFSRAW %h",{data,2'b00});
 @(negedge clk);reset=1;repeat(10)@(negedge clk);reset=0;
 scan(1,data);scan(0,data);
 if(data[127:126]!==2'b01)$fatal(1,"reset clear");
 $display("OFSRAW %h",{data,2'b00});
 $display("PASS FIRST_FAULT_SCAN_CDC reset/stall/freeze");$finish;
end
initial begin #1000000;$fatal(1,"timeout");end
endmodule
'''
    (r/'tb.sv').write_text(tb)
    originals={str(f):sha(f) for f in (MODULE,BASE/'assembly/jtag_bringup_snapshot.sv',HERE/'first_fault_observer.py',HERE/'read_first_fault.py')}
    inputs=dict(original=originals,files={f.name:sha(f) for f in r.iterdir() if f.is_file()},runner_sha256=sha(__file__))
    (r/'inputs.json').write_text(json.dumps(inputs,indent=2))
    run([MS/'vlib.exe','work'],r,'vlib.log',15)
    run([MS/'vlog.exe','-sv','control_first_fault.sv','jtag_bringup_snapshot.sv','tb.sv'],r,'compile.log',20)
    log,seconds=run([MS/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',30)
    assert 'PASS FIRST_FAULT_SCAN_CDC' in log
    packets=[decode(int(v,16)) for v in re.findall(r'OFSRAW ([0-9a-fA-F]+)',log)]
    assert len(packets)==8 and not packets[0]['payload']['captured']
    for packet in packets[1:7]:
        data=packet['payload'];assert data['captured'] and data['reason']['illegal_control_state']
        assert data['control_state']['raw']=='000' and data['previous_control_state']['asserted']==['DECODE']
        assert data['flags']['boot_fault']==8 and data['address']=='f8100040' and data['previous_address']=='f9000f00'
    assert packets[5]['busy'] and not packets[6]['busy'] and not packets[7]['payload']['captured']
    (r/'host_decoded.json').write_text(json.dumps(packets,indent=2))
    result=dict(status='FIRST_FAULT_SCAN_CDC_HOST_PASS',seconds=seconds,packets=len(packets),
                inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),decoded_sha256=sha(r/'host_decoded.json'),hardware_access=False)
    (r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
