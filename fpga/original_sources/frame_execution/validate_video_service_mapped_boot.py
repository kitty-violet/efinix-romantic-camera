"""Actual full-map boot state cone vs source (retained command mux separately checked).

Inputs are cut at the same bus and user-control boundaries. The full real
timeouts remain in the mapped cone; this bounded test covers load, live/stop,
rearm and payload invalidation. Parameter-scaled timeout tests are separate.
"""
import argparse,json,re,shutil,time,pickle
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,sha,run
from native_local_drivers import CELL,render_token
from audit_continuous_vectors import expand_graph
from continuous_control_native import connections,net
from camera_joint_native import OUT

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();out=a.root.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists();out.mkdir()
 started=time.monotonic();record=dict(status='RUNNING',hardware_access=False,board_release=False);inputs={}
 try:
  src=board/'outflow/ti60f225_oob.map.v';text=src.read_text();inputs={str(src):sha(src),str(Path(__file__)):sha(__file__)}
  cache=board/'service-validation-graph.pkl'
  cache_key=(sha(src),sha(HERE/'audit_continuous_vectors.py'),sha(HERE/'native_local_drivers.py'))
  if cache.exists():
   key,g=pickle.loads(cache.read_bytes());assert key==cache_key
  else:
   g=expand_graph(text);cache.write_bytes(pickle.dumps((cache_key,g),protocol=4))
  drivers,loads=connections(g)
  cuts={net('process_clk_150'):'clk',net('ceg_net4871'):'!reset',net('w_arstn'):'!reset',
   net('cpu/boot_ready'):'cmd_ready',net('n78042_2'):'rsp_valid',net('cpu/reply_error'):'rsp_error',net('cpu/fatal_bus_error'):"1'b0",
   net('service_button_event'):'button_event',
   # The mapper folded the three wrapper expressions into LUTs. Drive their
   # real external boundaries instead of inventing disconnected signal names.
   # LIVE_A disables automatic stop; a capture in flight prevents drain.
   net('cap_active'):'!work_drained',net('cap_done'):"1'b0",net('cap_failed'):"1'b0",
   net('service_front_valid'):"1'b0",net('video_processing_pipeline/pipeline/compute_active'):"1'b0",
   net('video_processing_pipeline/pipeline/prod_frame_valid'):"1'b0",
   net('cpu/ctrl_state_match',0):"1'b1",net('cpu/choose_host'):'payload_changed',net('cpu/host_write'):'payload_changed'}
  cuts.update({net('cpu/reply_data',i):f'rsp_data[{i}]' for i in range(32)})
  cuts.update({net('cpu/host_data',i):"1'b0" for i in range(32)})
  cuts.update({net('cpu/host_address',i):f"1'b{(0xe1000000>>i)&1}" for i in range(32)})
  cuts.update({net(n,i):"1'b0" for n,w in [('service_mode',2),('service_capture_gate/state',2),('video_processing_pipeline/frame_states',9)] for i in range(w)})
  outputs={'cmd_valid':('service_boot_valid',1),
   'prepared':('service_prepared',1),'run_allowed':('cpu/permit',1),
   'terminal':('cpu/boot_terminal',1),'failed':('cpu/boot_failed',1),'fault':('cpu/boot_fault',4),'observer_state':('cpu/boot_observer_state',6),
   'frame_budget':('service_frame_budget',10),'rejected_buttons':('service_rejected_buttons',32)}
  assert "assign rejected_buttons={16'd0,rejected_count};" in (board/'cnn_preview_boot.sv').read_text()
  # This logical port's upper half is a declared constant, eliminated by map.
  # Do not invent 16 undriven mapped registers for it.
  output_bits={port:[('literal',"1'b0",None) if port=='rejected_buttons' and i>=16 else net(n,i if w>1 else None) for i in range(w)] for port,(n,w) in outputs.items()}
  roots=[t for ts in output_bits.values() for t in ts]
  owned={n for n,(k,_) in g.items() if k in ('EFX_FF','EFX_RAM10') and n.startswith(('service_boot/','cpu/sequencer/','cpu/boot_observer_state[','cpu/boot_fault[','cpu/permit~','cpu/boot_terminal~','cpu/boot_failed~','preview_frame_budget[','service_rejected_buttons[','service_frame_budget[','service_prepared~'))}
  selected=set();todo=list(roots);needed=set(cuts);unknown=[]
  while todo:
   t=todo.pop();needed.add(t)
   if t[0]=='literal' or t in cuts:continue
   cell=drivers.get(t)
   if cell is None:unknown.append(('undriven',t));continue
   if cell in selected:continue
   kind,pins=g[cell]
   if kind not in ('EFX_LUT4','EFX_ADD') and cell not in owned:unknown.append((cell,t));continue
   selected.add(cell)
   for pin,ts in pins.items():
    needed.update(ts)
    if pin not in OUT[kind]:todo.extend(ts)
  (out/'unknown-cuts.json').write_text(json.dumps(unknown,indent=2))
  assert not unknown,('unexpected cut',unknown[:12])
  ports=['input wire clk,reset,button_event,payload_changed,work_drained,cmd_ready,rsp_valid,rsp_error,input wire[31:0] rsp_data']
  ports += ['output wire '+(f'[{w-1}:0] ' if w>1 else '')+n for n,(_,w) in outputs.items()]
  decl='\n'.join(re.findall(r'^\s*wire\b[^;]*;',text,re.M))
  assigns='\n'.join(f'assign {render_token(t)}={v};' for t,v in cuts.items() if t in needed)
  for port,ts in output_bits.items():assigns+='\nassign '+port+'={'+','.join(render_token(t) for t in reversed(ts))+'};'
  cells='\n'.join(m[0] for m in CELL.finditer(text) if m[2].lstrip('\\') in selected)
  params='\n'.join(m[0] for m in re.finditer(r'^\s*defparam\s+(\S+?)\s*\.\w+\s*=[^;]*;',text,re.M) if m[1].lstrip('\\') in selected)
  (out/'mapped_boot.sv').write_text('module mapped_boot('+','.join(ports)+');\n'+decl+'\n'+assigns+'\n'+cells+'\n'+params+'\nendmodule\n')
  source=board/'cnn_preview_boot.sv';shutil.copy2(source,out/'cnn_preview_boot.sv');inputs[str(source)]=sha(source)
  tb=(HERE/'variants/video_service/tb_video_boot_live.sv').read_text()
  tb=tb.replace(' wire [206:0] native_state;', ' wire [193:0] service_native_state;\n wire [206:0] native_state;')
  tb=tb.replace('cnn_preview_boot #(.RUN_TIMEOUT_CYCLES(8192),.NO_PROGRESS_CYCLES(4096),.BUS_TIMEOUT_CYCLES(64)) dut(.*);','cnn_preview_boot dut(.*);')
  tail=tb.index('  @(negedge clk);reset=1;repeat(4)@(negedge clk);reset=0;await_ready();\n  freeze_progress=1;')
  end=tb.index('  $display("PASS VIDEO_BOOT_LIVE',tail)
  tb=tb[:tail]+tb[end:]
  tb=tb.replace(' no_progress_timeout=1',' no_progress_timeout=source_scaled_test_only')
  extra='\ninteger comparisons=0;\n'
  for port,(_,w) in outputs.items():extra+='wire '+(f'[{w-1}:0] ' if w>1 else '')+'m_'+port+';\n'
  inports='clk reset button_event payload_changed work_drained cmd_ready rsp_valid rsp_error rsp_data'.split()
  extra+='mapped_boot mapped('+','.join('.'+n+'('+n+')' for n in inports)+','+','.join('.'+n+'(m_'+n+')' for n in outputs)+');\n'
  extra+='always @(negedge clk)begin #2;if(!reset)begin\n'
  for port in outputs:
   cond='cmd_valid && ' if port in ('cmd_write','cmd_address','cmd_data') else ''
   lhs=port+'[3:0]' if port=='fault' else port
   extra+=f'if({cond}{lhs}!==m_{port})$fatal(1,"mapped {port} differs state=%0d source=%h mapped=%h",observer_state,{port},m_{port});\n'
  extra+='comparisons=comparisons+1;end end\n'
  tb=tb.replace(' reg [31:0] rng=',extra+' reg [31:0] rng=',1).replace('$finish;','$display("PASS VIDEO_SERVICE_ACTUAL_MAP_BOOT comparisons=%0d",comparisons);$finish;',1)
  (out/'tb.sv').write_text(tb)
  for kind in {g[n][0] for n in selected}:shutil.copy2(EF/'sim_models/verilog'/(kind.lower()+'.v'),out/(kind.lower()+'.v'))
  inputs.update({str(f):sha(f) for f in out.iterdir() if f.suffix in ('.sv','.v')})
  (out/'inputs.json').write_text(json.dumps(dict(files=inputs,selected_cells=sorted(selected),cuts={render_token(k):v for k,v in cuts.items()}),indent=2))
  run([MS/'vlib.exe','work'],out,'vlib.log',20)
  run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*out.glob('*.v'),*out.glob('*.sv')],out,'compile.log',50)
  body,_=run([MS/'vsim.exe','-c','work.tb_video_boot_live','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],out,'simulate.log',240)
  assert 'PASS VIDEO_SERVICE_ACTUAL_MAP_BOOT' in body
  assert all(sha(f)==h for f,h in inputs.items()),'inputs changed'
  record.update(status='VIDEO_SERVICE_ACTUAL_MAP_BOOT_PASS',selected_cells=len(selected),markers=re.findall(r'^# PASS.*$',body,re.M))
 except Exception as e:record.update(status='FAIL',error=repr(e));raise
 finally:
  record.update(seconds=time.monotonic()-started,inputs=inputs)
  (out/'result.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k!='inputs'},indent=2))
if __name__=='__main__':main()
