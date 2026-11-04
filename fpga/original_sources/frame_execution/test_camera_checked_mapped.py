"""Pin-level tests of the actual full-board mapped camera cells and ROM bits."""
import argparse,concurrent.futures,json,os,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from camera_checked_board import BOARD,audit_delta
from native_local_drivers import graph,render_token
from probe_full_native_split_map import params
from windows_process_tree import stop_launched_tree

def extract(board):
 text=(board/'outflow/ti60f225_oob.map.v').read_text();g=graph(text);ps=params(text)
 names=json.loads((board/'result.json').read_text())['new_cells'];selected={n:g[n] for n in names}
 output_pins={'EFX_LUT4':['O'],'EFX_FF':['Q'],'EFX_ADD':['O','CO'],'EFX_RAM10':['RDATA']}
 driven=set();used=set()
 for kind,pins in selected.values():
  for pin,ts in pins.items():
   if pin in output_pins[kind]:driven.update(ts)
   used.update(t for t in ts if t[0]!='literal')
 nets={t:f'n{i}' for i,t in enumerate(sorted(used))}
 outside={t for t in used-driven if t[0]!='literal'}
 allowed={'CLK_5M':'clk','process_clk_150':'clk','i_mipi_rx_pclk':'clk',
  'w_arstn':'reset_n','w_i2c_arstn':'reset_n','w_mipi_rx_pclk_arstn':'reset_n',
  'io_cam_scl_IN':'scl_i','io_cam_sda_IN':'sda_i','w_mipi_rx_de':"1'b0",'w_mipi_rx_vs':"1'b0",'camera_boundary_error':"1'b0",'n75722_2':"1'b0",'n75725_2':"1'b0"}
 lines=['module camera_checked_config #(parameter PRESCALE=32)(input clk,reset_n,scl_i,sda_i,output scl_o,scl_oen,sda_o,sda_oen,config_ok,config_failed,output [63:0] diagnostic);',
  'wire '+','.join(nets.values())+';']
 for t in sorted(outside):
  if t[1] in ('raw_stream_errors','inst_efx_csi2_rx/u_efx_csi2_rx/csi2_rx_top_inst/genblk3.byte2pixel_inst/vc_sync'):v="1'b0"
  else:assert t[1] in allowed,t;v=allowed[t[1]]
  lines.append(f'assign {nets[t]}={v};')
 for p,n in [('scl_oen','camera_new_scl_oen'),('sda_oen','camera_new_sda_oen'),('config_ok','camera_config_ok'),('config_failed','camera_config_failed')]:
  lines.append(f'assign {p}={nets[("net",n,None)]};')
 lines+=['assign scl_o=0;assign sda_o=0;']
 for i in range(64):
  t=('net','camera_config_diagnostic',i)
  if t not in nets:assert i==59
  lines.append(f'assign diagnostic[{i}]='+nets.get(t,"1'b0")+';')
 def value(v):return str(v) if not isinstance(v,tuple) else f"{v[0]}'h{v[1]:x}"
 for n,(kind,pins) in selected.items():
  def expr(ts):
   vals=[t[1] if t[0]=='literal' else nets[t] for t in ts]
   return vals[0] if len(vals)==1 else '{'+','.join(vals)+'}' if vals else ''
  lines.append(kind+' \\'+n+' ('+','.join('.'+p+'('+expr(ts)+')' for p,ts in pins.items())+');')
  for k,v in ps[n].items():lines.append('defparam \\'+n+' .'+k+'='+value(v)+';')
 lines.append('endmodule')
 return '\n'.join(lines),dict(cells=len(selected),kinds=sorted(set(k for k,p in selected.values())),external_inputs=[list(t) for t in sorted(outside)])

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--modes',default='0,1,2,3,4,5,6,7,8,9');p.add_argument('--questa',action='store_true');p.add_argument('--budget',type=int,default=100);a=p.parse_args()
 assert 30<=a.budget<=600
 out=a.out.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists();audit_delta(BOARD);out.mkdir()
 body,proof=extract(BOARD);(out/'mapped_camera.v').write_text(body)
 tb=(HERE/'tb_camera_checked_config.sv').read_text()
 tb=tb.replace('mem[address]=shift;writes=writes+1;','mem[address]=shift;writes=writes+1; if(writes%32==0) $display("PROGRESS writes=%0d sim_ns=%0t",writes,$time);')
 (out/'tb.sv').write_text(tb)
 shutil.copy2(ROOT/'tmp/board-test/camera-checked-unit2/expected.hex',out/'expected.hex')
 models=[]
 for kind in proof['kinds']:
  f=EF/'sim_models/verilog'/(kind.lower()+'.v');assert f.exists() and 'pragma protect' not in f.read_text()
  shutil.copy2(f,out/f.name);models.append(f.name)
 compiler=ROOT/'tools/iverilog/ucrt64/bin/iverilog.exe';env=os.environ.copy();env['PATH']=str(compiler.parent)+os.pathsep+env.get('PATH','');env['TEMP']=env['TMP']=str(out)
 cmd=[str(compiler),'-g2012','-s','tb_camera_checked_config','-o','sim.vvp','mapped_camera.v','tb.sv',*models]
 Q=Path('E:/quatra/questa_fse/win64')
 if a.questa:
  env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
  r=subprocess.run([str(Q/'vlib.exe'),'work'],cwd=out,env=env,capture_output=True,text=True,timeout=20);assert r.returncode==0
  cmd=[str(Q/'vlog.exe'),'-sv','-timescale','1ns/1ps','mapped_camera.v','tb.sv',*models]
 r=subprocess.run(cmd,cwd=out,env=env,capture_output=True,text=True,timeout=30);(out/'compile.log').write_text(r.stdout+r.stderr);assert r.returncode==0,(r.returncode,r.stdout,r.stderr)
 files={f.name:sha(f) for f in out.iterdir() if f.suffix in ('.v','.sv','.hex')}
 meta=dict(map_sha256=sha(BOARD/'outflow/ti60f225_oob.map.v'),map_result_sha256=sha(BOARD/'result.json'),files=files,proof=proof,runner_sha256=sha(__file__),hardware_access=False)
 (out/'inputs.json').write_text(json.dumps(meta,indent=2))
 def case(mode):
  cmd=[str(compiler.with_name('vvp.exe')),'sim.vvp',f'+MODE={mode}']
  if a.questa:cmd=[str(Q/'vsim.exe'),'-c','work.tb_camera_checked_config',f'+MODE={mode}','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0','-l',f'transcript{mode}.log']
  start=time.monotonic();timed_out=False
  with (out/f'mode{mode}.log').open('w') as log:
   proc=subprocess.Popen(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
   try:code=proc.wait(timeout=a.budget)
   except subprocess.TimeoutExpired:
    timed_out=True;code=-1
    (out/f'mode{mode}_cleanup.json').write_text(json.dumps(stop_launched_tree(proc),indent=2))
  output=(out/f'mode{mode}.log').read_text(errors='replace')
  return dict(mode=mode,passed=not timed_out and code==0 and 'PASS CHECKED_CAMERA' in output,seconds=time.monotonic()-start,budget_seconds=a.budget,timed_out=timed_out,log_sha256=sha(out/f'mode{mode}.log'),output=output)
 # This installed uncounted node-locked license permits one Questa session.
 with concurrent.futures.ThreadPoolExecutor(max_workers=1 if a.questa else 2) as pool:cases=list(pool.map(case,map(int,a.modes.split(','))))
 result=dict(status='CAMERA_ACTUAL_MAP_PIN_TESTS_PASS' if all(c['passed'] for c in cases) else 'FAIL',cases=cases,inputs_sha256=sha(out/'inputs.json'),map_sha256=meta['map_sha256'],hardware_access=False)
 for n,h in files.items():assert sha(out/n)==h
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);assert all(c['passed'] for c in cases)
if __name__=='__main__':main()
