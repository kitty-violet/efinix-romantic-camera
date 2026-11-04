"""Small map-only probe: does an explicit LUT keep independent control cones?"""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,EF,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
source='''module lut_binding_probe(input wire clk,input wire [8:0] data,output reg [5:0] q);
reg [8:0] d;always @(posedge clk)d<=data;
wire [5:0] f;
assign f[0]=d[0] && d[2];assign f[1]=d[0] ^ d[1];
EFX_LUT4 #(.LUTMASK(16'h8888)) plain_a(.I0(d[3]),.I1(d[5]),.I2(1'b0),.I3(1'b0),.O(f[2]));
EFX_LUT4 #(.LUTMASK(16'h6666)) plain_b(.I0(d[3]),.I1(d[4]),.I2(1'b0),.I3(1'b0),.O(f[3]));
(* syn_keep="true" *) wire [1:0] kept;
(* syn_preserve="true" *) EFX_LUT4 #(.LUTMASK(16'h8888)) kept_a(.I0(d[6]),.I1(d[8]),.I2(1'b0),.I3(1'b0),.O(kept[0]));
(* syn_preserve="true" *) EFX_LUT4 #(.LUTMASK(16'h6666)) kept_b(.I0(d[6]),.I1(d[7]),.I2(1'b0),.I3(1'b0),.O(kept[1]));
assign f[5:4]=kept;
always @(posedge clk)q<=f;
endmodule
'''
(r/'lut_binding_probe.sv').write_text(source)
project=(ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml').read_text().replace('frame_execution_meter','lut_binding_probe')
project=project.replace('</efx:synthesis>','<efx:param name="pack-luts-to-comb4" value="2" value_type="e_integer" /></efx:synthesis>')
(r/'lut_binding_probe.xml').write_text(project);(r/'lut_binding_probe.sdc').write_text('create_clock -name clk -period 6.666 [get_ports {clk}]\n')
shutil.copy2(__file__,r/'runner.py')
(r/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat lut_binding_probe.xml --flow map --timeout 60\nexit /b %errorlevel%\n')
body,sec=run(['cmd.exe','/d','/c','run.cmd'],r,'map.log',timeout=90)
assert re.search(r'map\s*:\s*PASS',body)
net=r/'outflow/lut_binding_probe.map.v';report=r/'outflow/lut_binding_probe.map.rpt'
data=dict(status='LUT_PACKING_MAP_PROBE_ONLY',seconds=sec,map_sha256=sha(net),report_sha256=sha(report),resources=dict(re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',report.read_text(),re.M)))
(r/'result.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
