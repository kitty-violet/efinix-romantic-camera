"""Map the exact observation chain first, without waiting for a board build."""
import argparse,json,re,os
from pathlib import Path
from run_units import ROOT,HERE,EF,sha,run

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
top=HERE/'variants/board_bringup_status/ti60f225_oob_top.v'
source=top.read_text();start=source.index('(* async_reg="true", syn_srlstyle=')
end=source.index('wire [31:0] committed_frames',start)
block=source[start:end]
name='bringup_sync_probe'
(r/(name+'.sv')).write_text('module '+name+'(input wire process_clk_150,input wire [31:0] bringup_ddr_raw,output wire [31:0] status);\n'+block+'assign status=bringup_ddr_sync;\nendmodule\n')
xml=(ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml').read_text().replace('frame_execution_meter',name)
(r/(name+'.xml')).write_text(xml)
(r/(name+'.sdc')).write_text('create_clock -name process_clk_150 -period 6.6667 [get_ports {process_clk_150}]\n')
(r/'inputs.json').write_text(json.dumps(dict(top=str(top),top_sha256=sha(top),block=block,source_sha256=sha(r/(name+'.sv'))),indent=2))
(r/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\ncall efx_run.bat '+name+'.xml --flow map --timeout 60\nexit /b %errorlevel%\n')
body,seconds=run(['cmd.exe','/d','/c','run.cmd'],r,'map.log',timeout=80)
assert re.search(r'map\s*:\s*PASS',body)
net=r/('outflow/'+name+'.map.v');mapped=net.read_text()
ffs=re.findall(r'^\s*EFX_FF\s+',mapped,re.M)
assert len(ffs)==64 and 'EFX_SRL8' not in mapped
cells=re.findall(r'EFX_FF\s+(\S+)\s+\((.*?)\) /\*',mapped,re.S)
for k in range(32):
 for data,q in ((f'bringup_ddr_raw[{k}]',f'bringup_ddr_meta[{k}]'),(f'bringup_ddr_meta[{k}]',f'status[{k}]')):
  assert sum('.D('+data+')' in ports and '.Q('+q+')' in ports for _,ports in cells)==1
os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
ms=Path('E:/quatra/questa_fse/win64')
(r/'tb.sv').write_text('''module tb;
reg clk=0;always #5 clk=~clk;reg [31:0] d=0;wire [31:0] q;reg [31:0] previous=0;integer k;
bringup_sync_probe dut(.process_clk_150(clk),.bringup_ddr_raw(d),.status(q));
initial begin
for(k=0;k<200;k=k+1)begin
 @(negedge clk);d=$random;@(posedge clk);#1;if(q!==previous)$fatal(1,"two cycle sample");previous=d;
end
$display("PASS BRINGUP_SYNC_200");$finish;end
endmodule
''')
run([ms/'vlib.exe','work'],r,'vlib.log')
run([ms/'vlog.exe','-sv','-timescale','1ns/1ps',r/(name+'.sv'),r/'tb.sv'],r,'compile.log')
log,simseconds=run([ms/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'sim.log',timeout=40)
assert 'PASS BRINGUP_SYNC_200' in log and 'Errors: 0, Warnings: 0' in log
assert top.read_text()[start:end]==block
result=dict(status='BRINGUP_SYNC_64_FF_NO_SRL_PASS',map_seconds=seconds,simulation_seconds=simseconds,
 top_sha256=sha(top),inputs_sha256=sha(r/'inputs.json'),map_sha256=sha(net),ff=64,srl=0,board_release=False)
(r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
