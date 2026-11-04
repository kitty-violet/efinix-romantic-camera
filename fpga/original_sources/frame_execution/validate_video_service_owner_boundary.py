"""Actual mapped ownership cone, including real graft terminal connections.

The fault input is injected before the retained capture-failure inverter and
at the retained CNN error aggregate, not at the owner's compute_failed port.
"""
import argparse,json,pickle,re,shutil,time
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,sha,run
from audit_continuous_vectors import expand_graph
from continuous_control_native import connections,net
from native_local_drivers import CELL,render_token
from camera_joint_native import OUT
from video_service_graph import load

P='video_processing_pipeline/pipeline/'
def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();out=a.root.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists();out.mkdir()
 start=time.monotonic();result=dict(status='RUNNING',hardware_access=False,board_release=False)
 try:
  src=board/'outflow/ti60f225_oob.map.v';text=src.read_text();native=(board/'native_source.v').read_text()
  cache=board/'service-validation-graph.pkl';key=(sha(src),sha(HERE/'audit_continuous_vectors.py'),sha(HERE/'native_local_drivers.py'))
  if cache.exists():old,g=pickle.loads(cache.read_bytes());assert old==key
  else:g=expand_graph(text);cache.write_bytes(pickle.dumps((key,g),protocol=4))
  drivers,loads=connections(g)
  # Locate wrapper handshakes through retained native consumers, so renamed
  # mapped aliases cannot turn into independent, unconnected test inputs.
  graft=json.loads((Path(json.loads((board/'inputs.json').read_text())['native'])/'graft.json').read_text())
  base=load();assert base['sha256']==graft['source_sha256']
  old={n:(k,{p:list(v) for p,v in pins.items()}) for n,(k,pins) in base['g'].items() if n not in graft['removed']}
  for edit in graft['changed_loads']:
   n,p=edit['cell'],edit['pin'];a0,b0=tuple(edit['from_token']),tuple(edit['to_token'])
   old[n][1][p]=[b0 if t==a0 else t for t in old[n][1][p]]
  od,ol=connections(old)
  assert 'LUT__243923_2' in old and old['LUT__243923_2'][1]['O']==[net('cap_failed')], 'native capture-failure driver was removed'
  def actual(t):
   found={g[n][1][pin][old[n][1][pin].index(t)] for n,pin in ol.get(t,[]) if n in g}
   n=od.get(t)
   if n in g:
    k,pins=old[n]
    for pin in OUT[k]:
     if t in pins.get(pin,[]):found.add(g[n][1][pin][pins[pin].index(t)])
   assert len(found)==1,(t,found)
   return next(iter(found))
  cuts={net('process_clk_150'):'clk',
   actual(net('u_ddr3_top/controller/u_efx_ddr3_soft_controller/genblk1.inst_ddr3_axi/inst_efx_ddr3/top_mc/clk_rstn')):'!reset',
   actual(net('camera_burst_capture/start_fire')):'capture_valid',
   actual(net('cap_done')):'capture_done',
   g['LUT__243923_2'][1]['I0'][0]:'!capture_failed',
   actual(net('video_processing_pipeline/compute_start')):'compute_ready',
   actual(net(P+'compute_active')):'native_active',
   actual(net(P+'prod_frame_valid')):'compute_done',
   actual(net(P+'cnn/dp_error')):'compute_failed',
   actual(net(P+'display/start_fire')):'display_ready',
   actual(net('video_processing_pipeline/display_done')):'display_done',
   net('service_mode',0):'bypass_compute',net('service_mode',1):"1'b0"}
  # Full-map Boolean folding can bypass the named handshakes above. These
  # external producer registers model a loaded, idle core and enabled board;
  # they are not owner state. All error injection paths use the same value.
  external={
   'cal_done_sync':"1'b1",'ceg_net4871':'!reset',
   'control_cancel':"1'b0",'control_enable':"1'b1",
   'cpu/fatal_bus_error':"1'b0",'cpu/permit':'compute_ready',
   'load_complete':"1'b1",'loading':"1'b0",'store_error':"1'b0",
   'service_boot/stop_command':"1'b0",'startup_reset/released':'!reset',
   'DDR3_PLL_RSTN':'!reset','i_pll_locked':"1'b1",'pll_locked':"1'b1",
   'process_pll_locked':"1'b1",'user_pll_locked':"1'b1",
   P+'stopped':"1'b0",P+'cnn/core/address_fault':"1'b0",
   P+'cnn/core/header/bias_pending':"1'b0",P+'cnn/core/header/mask_pending':"1'b0",
   P+'cnn/core/header/status':"1'b0",P+'cnn/core/stopped':"1'b0",
   P+'cnn/core/stopping':"1'b0",P+'cnn/core/store/verified':"1'b1",
   P+'cnn/datapath_fault_q':"1'b0",P+'cnn/fetch/active':"1'b0",
   P+'cnn/fetch_error':"1'b0",P+'cnn/issue_busy':"1'b0",P+'cnn/issue_error':'compute_failed'}
  cuts.update({net(n):v for n,v in external.items()})
  for n,w,value in [('cpu/boot_observer_state',6,47),('service_frame_budget',10,1023),
   ('service_admission/run_admitted',10,0),(P+'cnn/core/dma/state',3,0),
   (P+'cnn/core/dut/state',4,0),(P+'cnn/core/header/fetch_state',2,0),(P+'cnn/fetch/state',3,0)]:
   cuts.update({net(n,i):f"1'b{(value>>i)&1}" for i in range(w)})
  outputs={'states':('video_processing_pipeline/frame_states',9),'error':('bringup_pipeline_status',1),
   'committed_frames':('committed_frames',32),'dropped_results':('dropped_results',32),
   'repeated_reads':('service_repeated',32),'front_valid':('service_front_valid',1)}
  roots={name:([net(n,0)] if name=='error' else [net(n,i if w>1 else None) for i in range(w)]) for name,(n,w) in outputs.items()}
  owner_text=(board/'frame_pair_owner.sv').read_text()
  for port,counter in [('dropped_results','dropped_count'),('repeated_reads','repeated_count')]:
   assert f"assign {port}={{16'd0,{counter}}};" in owner_text
   roots[port][16:]=[('literal',"1'b0",None)]*16
  exported={tuple(x['token']) for x in graft['exports']['owner']}
  owned={n for n,(k,pins) in g.items() if k=='EFX_FF' and (n.startswith(('service_owner/','service_front_','service_repeated[')) or pins['Q'][0] in exported)}
  selected=set();needed=set();unknown=[];todo=[q for ts in roots.values() for q in ts]
  while todo:
   t=todo.pop();needed.add(t)
   if t in cuts or t[0]=='literal':continue
   n=drivers.get(t)
   if n is None:unknown.append(('undriven',t));continue
   if n in selected:continue
   k,pins=g[n]
   if k not in ('EFX_LUT4','EFX_ADD') and n not in owned:unknown.append((n,t));continue
   selected.add(n)
   for pin,ts in pins.items():
    needed.update(ts)
    if pin not in OUT[k]:todo.extend(ts)
  (out/'unknown-cuts.json').write_text(json.dumps(sorted(set(map(str,unknown))),indent=2));assert not unknown,unknown[:12]
  assert 'LUT__243923_2' in selected,'failure inverter must be exercised'
  inputs='clk reset bypass_compute capture_valid capture_done capture_failed compute_ready compute_done compute_failed display_ready display_done native_active'.split()
  ports=['input wire '+','.join(inputs)]+['output wire '+(f'[{w-1}:0] ' if w>1 else '')+'out_'+n for n,(_,w) in outputs.items()]
  decl='\n'.join(re.findall(r'^\s*wire\b[^;]*;',text,re.M))
  assignments='\n'.join(f'assign {render_token(t)}={expr};' for t,expr in cuts.items() if t in needed)
  decl+='\nwire[7:0]led;'
  for port,ts in roots.items():assignments+='\nassign out_'+port+'={'+','.join(render_token(t) for t in reversed(ts))+'};'
  cells='\n'.join(m[0] for m in CELL.finditer(text) if m[2].lstrip('\\') in selected)
  params='\n'.join(m[0] for m in re.finditer(r'^\s*defparam\s+(\S+?)\s*\.\w+\s*=[^;]*;',text,re.M) if m[1].lstrip('\\') in selected)
  (out/'mapped_owner.sv').write_text('module mapped_owner('+','.join(ports)+');\n'+decl+'\n'+assignments+'\n'+cells+'\n'+params+'\nendmodule\n')
  shutil.copy2(board/'frame_pair_owner.sv',out/'frame_pair_owner.sv')
  tb=(HERE/'variants/video_service/tb_video_pair_owner.sv').read_text()
  tb=tb.replace('frame_pair_owner dut(.*);','frame_pair_owner dut(.*, .service_native_state());')
  extra='\nreg native_active=0;\nalways @(posedge clk)if(reset)native_active<=0;else if(compute_ready)native_active<=1;else if(compute_done||compute_failed)native_active<=0;\ninteger comparisons=0;\n'
  for n,(_,w) in outputs.items():extra+='wire '+(f'[{w-1}:0] ' if w>1 else '')+'m_'+n+';\n'
  extra+='mapped_owner mapped('+','.join('.'+n+'('+n+')' for n in inputs)+','+','.join('.out_'+n+'(m_'+n+')' for n in outputs)+');\n'
  extra+='always @(negedge clk)begin #1;if(!reset)begin\n'
  for n in outputs:extra+=f'if({n}!==m_{n})$fatal(1,"actual mapped owner {n}: source=%h mapped=%h",{n},m_{n});\n'
  extra+='comparisons=comparisons+1;end end\n'
  tb=tb.replace(' task capture',extra+' task capture',1).replace('$finish;','$display("PASS ACTUAL_MAP_OWNER_BOUNDARY comparisons=%0d capture_failure=1 compute_failure=1 repeated_front=1",comparisons);$finish;',1)
  (out/'tb.sv').write_text(tb)
  for k in {g[n][0] for n in selected}:shutil.copy2(EF/'sim_models/verilog'/(k.lower()+'.v'),out/(k.lower()+'.v'))
  files={str(f):sha(f) for f in out.iterdir() if f.suffix in ('.v','.sv')}
  (out/'inputs.json').write_text(json.dumps(dict(map_sha256=sha(src),checker_sha256=sha(__file__),files=files,selected_cells=sorted(selected),cuts={str(k):v for k,v in cuts.items()}),indent=2))
  run([MS/'vlib.exe','work'],out,'vlib.log',20)
  run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*out.glob('*.v'),*out.glob('*.sv')],out,'compile.log',50)
  body,_=run([MS/'vsim.exe','-c','work.tb_video_pair_owner','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],out,'simulate.log',120)
  assert 'PASS ACTUAL_MAP_OWNER_BOUNDARY' in body
  assert all(sha(f)==h for f,h in files.items())
  result.update(status='ACTUAL_MAP_OWNER_FAILURE_BOUNDARY_PASS',selected_cells=len(selected),markers=re.findall(r'^# PASS.*$',body,re.M),inputs_sha256=sha(out/'inputs.json'))
 except Exception as e:result.update(status='FAIL',error=repr(e));raise
 finally:result['seconds']=time.monotonic()-start;(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
