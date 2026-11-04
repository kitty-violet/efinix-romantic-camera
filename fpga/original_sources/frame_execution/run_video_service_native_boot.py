"""Exercise the real retained native command mux with the replacement boot.

Uses official EFX primitive models and the LOAD/LIVE/STOP protocol test. This
checks the nontrivial graft boundary, rather than testing only source RTL.
"""
import argparse,collections,json,re,shutil,time
from pathlib import Path
from run_units import ROOT,HERE,MS,EF,sha,run
from video_service_graph import load
from native_local_drivers import render_token,lexical_tokens
from camera_joint_native import OUT

def main():
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 src=a.native.resolve();out=a.root.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists();out.mkdir(parents=True)
 proof=json.loads((src/'graft.json').read_text());D=load();g,drivers,ps,raw=[D[k] for k in ('g','drivers','ps','raw')]
 assert proof['source_sha256']==D['sha256']
 boundary={tuple(r['token']) for r in proof['exports']['boot']}
 boundary.add(('net','cpu/choose_host',None));boundary.add(('net','cpu/host_write',None));boundary.update(('net','cpu/host_data',i) for i in range(32));boundary.update(('net','cpu/host_address',i) for i in range(32))
 roots=[('net','cpu/n736',i) for i in range(32)]+[('net','cpu/n769',i) for i in range(32)]+[('net','cpu/n732',None)]
 selected=set();todo=list(roots);needed=set(boundary)
 while todo:
  t=todo.pop();needed.add(t)
  if t[0]!='net' or t in boundary:continue
  n=drivers.get(t);assert n is not None,('unresolved leaf',t)
  if n in selected:continue
  assert n not in proof['removed'],('removed mux source',n,t)
  k,pins=g[n];assert k in ('EFX_LUT4','EFX_ADD'),('non-combinational command mux',n,k)
  selected.add(n)
  for pin,ts in pins.items():
   needed.update(ts)
   if pin not in OUT[k]:todo.extend(ts)
 # Preserve the source port list used by the existing bus-level test.
 original=(HERE/'variants/video_service/cnn_preview_boot.sv').read_text()
 header=original[:original.index(');')+2]
 rows=proof['exports']['boot'];native=(src/'cnn_preview_boot.sv').read_text().replace('module cnn_preview_boot #(','module service_boot_rtl #(',1)
 (out/'service_boot_rtl.sv').write_text(native)
 declarations={}
 for t in needed:
  if t[0]!='net':continue
  name,bit=t[1:];declarations.setdefault(name,[]).append(bit)
 code=header+'\n'
 for n,bits in sorted(declarations.items()):
  assert not (None in bits and len(set(bits))>1),(n,bits)
  code+='wire '+(f'[{max(bits)}:0] ' if bits[0] is not None else '')+'\\'+n+' ;\n'
 code+='wire ref_valid,ref_write;wire[31:0] ref_address,ref_data;\n'
 code+='service_boot_rtl #(.RUN_TIMEOUT_CYCLES(RUN_TIMEOUT_CYCLES),.NO_PROGRESS_CYCLES(NO_PROGRESS_CYCLES),.BUS_TIMEOUT_CYCLES(BUS_TIMEOUT_CYCLES)) core(\n'
 ports=['clk','reset','button_event','payload_changed','work_drained','prepared','draining','rejected_buttons','cmd_ready','rsp_valid','rsp_error','rsp_data','run_allowed','terminal','failed','fault','observed_crc','observer_state','native_state','frame_budget']
 code+=',\n'.join('.'+n+'('+n+')' for n in ports)+',\n.cmd_valid(ref_valid),.cmd_write(ref_write),.cmd_address(ref_address),.cmd_data(ref_data),\n.service_native_state({'+', '.join(render_token(tuple(r['token'])) for r in reversed(rows))+'}));\n'
 code+="assign \\cpu/choose_host =1'b0;assign \\cpu/host_write =1'b0;assign \\cpu/host_data =32'h0;assign \\cpu/host_address =32'h0;\n"
 for n in sorted(selected):
  k,pins,_=raw[n];code+=k+' \\'+n+' ('+', '.join('.'+p+'('+e+')' for p,e in pins.items())+');\n'
  for key,(w,val) in ps[n].items():code+=f"defparam \\{n} .{key}={w}'h{val:x};\n"
 code+='always @* begin cmd_valid=ref_valid;cmd_address=\\cpu/n736 ;cmd_data=\\cpu/n769 ;cmd_write=\\cpu/n732 ;end\n'
 code+='always @(negedge clk)if(!reset && ref_valid && (cmd_address!==ref_address || cmd_data!==ref_data || cmd_write!==ref_write))$fatal(1,"native command mux mismatch state=%d address=%h/%h data=%h/%h write=%b/%b",observer_state,cmd_address,ref_address,cmd_data,ref_data,cmd_write,ref_write);\nendmodule\n'
 (out/'cnn_preview_boot.sv').write_text(code)
 tb=(HERE/'variants/video_service/tb_video_boot_live.sv').read_text().replace('dut.state','dut.core.state').replace('dut.waiting','dut.core.waiting')
 (out/'tb_video_boot_live.sv').write_text(tb)
 snapshots=[out/'service_boot_rtl.sv',out/'cnn_preview_boot.sv',out/'tb_video_boot_live.sv']
 for fn in ['efx_lut4.v','efx_add.v']:
  p=EF/'sim_models/verilog'/fn;shutil.copy2(p,out/p.name);snapshots.append(out/p.name)
 inputs={str(p):sha(p) for p in [src/'graft.json',src/'cnn_preview_boot.sv',HERE/'variants/video_service/tb_video_boot_live.sv',Path(__file__),*snapshots]}
 (out/'inputs.json').write_text(json.dumps(dict(files=inputs,selected_native_cells=sorted(selected)),indent=2))
 result=dict(status='RUNNING',native_cells=len(selected),hardware_access=False,board_release=False);start=time.monotonic()
 try:
  run([MS/'vlib.exe','work'],out,'vlib.log',30)
  run([MS/'vlog.exe','-sv',*[str(p) for p in snapshots]],out,'compile.log',45)
  log,_=run([MS/'vsim.exe','-c','work.tb_video_boot_live','-do','run -all; quit -code 0'],out,'simulate.log',45)
  assert 'PASS VIDEO_BOOT_LIVE' in log
  assert all(sha(p)==h for p,h in inputs.items())
  result['status']='RETAINED_NATIVE_COMMAND_MUX_LOAD_LIVE_STOP_PASS'
 except Exception as e:result.update(status='FAIL',error=str(e));raise
 finally:
  result['seconds']=time.monotonic()-start;result['inputs_sha256']=sha(out/'inputs.json')
  (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
