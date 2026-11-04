"""Actual Efinity small mapped observer versus RTL, using official primitives."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,sha,run
p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
mapped=ROOT/'tmp/frame-execution/snapshot-sync-map1/outflow/jtag_bringup_snapshot.map.v'
source=HERE/'variants/board_control_snapshot/jtag_bringup_snapshot.sv'
ref=source.read_text().replace('module jtag_bringup_snapshot (','module observer_reference (',1)
(r/'reference.sv').write_text(ref);shutil.copy2(mapped,r/mapped.name)
sources=[r/'reference.sv',r/mapped.name];bindings={str(source):sha(source),str(mapped):sha(mapped)}
for name in ('ff','lut4','add','srl8'):
 f=EF/f'sim_models/verilog/efx_{name}.v';shutil.copy2(f,r/f.name);sources.append(r/f.name);bindings[str(f)]=sha(f)
tb=r'''module tb;
reg clk=0,run_clock=1,tck=0,request_capture=0,poll_capture=0,shift=0;
reg [31:0] tick_count=0;
always #3.333 if(run_clock) clk=~clk;
always @(posedge clk) tick_count<=tick_count+1;
wire [191:0] payload={tick_count,~tick_count,tick_count^32'h98765432,tick_count+32'd17,tick_count-32'd11,tick_count};
reg [7:0] levels=0;
wire actual,reference;
jtag_bringup_snapshot dut(clk,tck,request_capture,poll_capture,shift,payload,levels,actual);
observer_reference golden(clk,tck,request_capture,poll_capture,shift,payload,levels,reference);
integer compared=0;
task edge_tck;begin
 #23;tck=1;#1;
 if(actual!==reference || (actual!==0 && actual!==1))$fatal(1,"mapped/RTL mismatch count=%d",compared);
 compared=compared+1;#28;tck=0;
end endtask
task scan(input request,output [255:0] data);begin
 request_capture=request;poll_capture=!request;shift=0;edge_tck();request_capture=0;poll_capture=0;shift=1;
 for(integer b=0;b<256;b=b+1)begin data[b]=actual;edge_tck();end
 shift=0;edge_tck();
 if(data[31:0]!==32'h4f425331)$fatal(1,"signature");
end endtask
reg [255:0] data;
initial begin
 repeat(20)@(negedge clk);
 scan(1,data);scan(0,data);
 if(data[33:32]!==1)$fatal(1,"initial fresh");
 repeat(80)begin levels=levels+1;scan(1,data);scan(0,data);end
 @(negedge clk);run_clock=0;
 scan(1,data);scan(0,data);if(data[33:32]!==3)$fatal(1,"stopped busy");
 repeat(3)scan(1,data);
 run_clock=1;scan(0,data);scan(0,data);if(data[33:32]!==1)$fatal(1,"recovery");
 $display("PASS MAPPED_OBSERVER compared=%0d",compared);$finish;
end
initial begin #10000000;$fatal(1,"timeout");end
endmodule
'''
(r/'tb.sv').write_text(tb);sources.append(r/'tb.sv')
inputs=dict(original=bindings,files={f.name:sha(f) for f in r.iterdir() if f.is_file()},runner_sha256=sha(__file__))
(r/'inputs.json').write_text(json.dumps(inputs,indent=2))
run([MS/'vlib.exe','work'],r,'vlib.log',20)
run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*sources],r,'compile.log',30)
body,seconds=run([MS/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',45)
m=re.search(r'PASS MAPPED_OBSERVER compared=(\d+)',body);assert m and int(m[1])>40000
for path,h in bindings.items():assert sha(path)==h
result=dict(status='MAPPED_OBSERVER_RTL_EQUIVALENCE_PASS',compared=int(m[1]),seconds=seconds,
 inputs_sha256=sha(r/'inputs.json'),simulation_sha256=sha(r/'simulation.log'),hardware_access=False)
(r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
