"""Affected cones extracted from the complete mapped board, official models only."""
import argparse,itertools,json,os,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,sha
from native_local_drivers import graph
from probe_full_native_split_map import params
from camera_recovery_board import BOARD,PARENT

def net(n,i=None):return ('net',n,i)
def lit(v):return ('literal',f"1'b{v}",None)
OUT={'EFX_FF':['Q'],'EFX_LUT4':['O'],'EFX_ADD':['O','CO']}

class Cones:
 def __init__(self,board):
  self.text=(board/'outflow/ti60f225_oob.map.v').read_text();self.g=graph(self.text);self.ps=params(self.text)
  self.drivers={t:n for n,(k,p) in self.g.items() for o in OUT.get(k,[]) for t in p.get(o,[])}
  self.added=set(json.loads((board/'result.json').read_text())['added_cells'])
 def extract(self,module,ports,inputs,outputs):
  chosen=set()
  def visit(t):
   if t[0]=='literal' or t in inputs:return
   assert t in self.drivers,('unknown external',t)
   n=self.drivers[t];assert n in self.added,('escaped affected cone',n,t)
   if n in chosen:return
   chosen.add(n);k,p=self.g[n]
   for pin,ts in p.items():
    if pin not in OUT[k]:
     for x in ts:visit(x)
  for t in outputs.values():visit(t)
  used=set(inputs)|set(outputs.values())
  for n in chosen:
   for ts in self.g[n][1].values():used.update(ts)
  names={t:f'n{i}' for i,t in enumerate(sorted((t for t in used if t[0]!='literal'),key=str))}
  def val(t):return t[1] if t[0]=='literal' else names[t]
  def expr(ts):return val(ts[0]) if len(ts)==1 else '{'+','.join(map(val,ts))+'}' if ts else ''
  lines=[f'module {module}({ports});','wire '+','.join(names.values())+';']
  lines += [f'assign {names[t]}={v};' for t,v in inputs.items()]
  lines += [f'assign {p}={val(t)};' for p,t in outputs.items()]
  for n in sorted(chosen):
   k,p=self.g[n];lines.append(k+' \\'+n+' ('+','.join('.'+pin+'('+expr(ts)+')' for pin,ts in p.items())+');')
   for key,v in self.ps[n].items():
    value=f"{v[0]}'h{v[1]:x}" if isinstance(v,tuple) else str(v)
    lines.append('defparam \\'+n+' .'+key+'='+value+';')
  lines.append('endmodule')
  return '\n'.join(lines),dict(cells=sorted(chosen),kinds=sorted({self.g[n][0] for n in chosen}),external_inputs=[dict(token=t,port=v) for t,v in inputs.items()])

def generate(board):
 c=Cones(board);g=c.g
 outputs={p:net(n) for p,n in [('frame_start','camera_frame_start'),('frame_end','camera_frame_end'),('pixel_valid','camera_pixel_valid'),('frame_active','camera_frame_active'),('polarity_known','camera_polarity_known'),('active_video_vsync_level','camera_active_vsync_level'),('boundary_error','camera_boundary_error')]}
 inputs={net(n):p for n,p in [('i_mipi_rx_pclk','clk'),('raw_frontend/ingress/n10','rst_n'),('w_mipi_rx_vs','vsync'),('n1345','data_enable')]}
 boundary,bp=c.extract('video_frame_boundary_detector','input clk,rst_n,vsync,data_enable,output '+','.join(outputs),inputs,outputs)
 obs_in={net(n):p for n,p in [('process_clk_150','system_clk'),('w_arstn','system_reset_n'),('i_mipi_rx_pclk','pixel_clk'),('raw_frontend/ingress/n10','pixel_reset_n'),('w_mipi_rx_vs','vs'),('w_mipi_rx_de','de'),('camera_polarity_known','known'),('camera_active_vsync_level','level'),('camera_frame_active','frame_active'),('camera_frame_start','fs'),('camera_frame_end','fe'),('camera_pixel_valid','pv'),('camera_boundary_error','boundary_error'),('raw_frontend/ingress/crossing/wr_guard_done','packer_ready'),('raw_frontend/ingress/crossing/fifo_full',"1'b0"),('video_processing_pipeline/pipeline/capture/active','capture_active')]}
 obs_in[g['camera_observer/hs_d~FF'][1]['D'][0]]='hs'
 obs_in[net('raw_stream_errors',7)]='length_error'
 obs_in.update({net('raw_frontend/ingress/packer/captured_words',i):f'packer_words[{i}]' for i in range(19)})
 obs_in.update({net('video_processing_pipeline/frame_states',i):f'owner_states[{i}]' for i in range(9)})
 diag=[]
 for i in range(111):
  t=net('camera_stream_faults',i)
  if t not in c.drivers:
   if i in (19,20,51,52,60) or 95<=i<=100:t=lit(0)
   elif 61<=i<=63:t=lit(1)
   elif i==101:t=net('video_processing_pipeline/pipeline/capture/active')
   elif 102<=i<=110:t=net('video_processing_pipeline/frame_states',i-102)
   else:raise AssertionError(('unexpected optimized output',i))
  diag.append(t)
 # Prove all 111 mapped page selectors, including all constant and aliased
 # bits, against the frozen original snapshot sources. No LUT ID assumptions.
 old=graph((PARENT/'outflow/ti60f225_oob.map.v').read_text());rename=json.loads((board/'public_connectivity_audit.json').read_text())['renamed_nets']
 slots=json.loads((board/'camera_graft.json').read_text())['diagnostic_slots'];prefix='cpu/system_hardJtag_debug_logic_jtagBridge/observation/'
 select=net(prefix+'request_sync');truth=0
 def ev(t,values):
  if t[0]=='literal':assert t[1] in ("1'b0","1'b1");return int(t[1][-1])
  if t in values:return values[t]
  n=c.drivers[t];k,p=g[n];assert k=='EFX_LUT4' and n in c.added
  idx=sum(ev(p.get('I'+str(i),[lit(0)])[0],values)<<i for i in range(4))
  return c.ps[n]['LUTMASK'][1]>>idx&1
 for i,slot in enumerate(slots):
  n=prefix+f'held_payload[{832+slot}]~FF';original=old[n][1]['D'][0]
  original=(original[0],rename.get(original[1],original[1]),original[2])
  leaves=list(dict.fromkeys(t for t in [select,diag[i],original] if t[0]!='literal'))
  for bits in itertools.product((0,1),repeat=len(leaves)):
   v=dict(zip(leaves,bits));expected=ev(diag[i] if v[select] else original,v)
   assert ev(g[n][1]['D'][0],v)==expected,(i,v);truth+=1
 # Mapping shares combinational terms between boundary and observer. Keep
 # that entire real cone, rather than inventing independent internal inputs.
 for n in ('camera_polarity_known','camera_active_vsync_level','camera_frame_active','camera_frame_start','camera_frame_end','camera_pixel_valid','camera_boundary_error'):
  obs_in.pop(net(n))
 obs_in[net('n1345')]='qualified_de'
 observer,op=c.extract('camera_recovery_bundle','input system_clk,system_reset_n,pixel_clk,pixel_reset_n,vs,hs,de,qualified_de,packer_ready,length_error,input [19:0] packer_words,input capture_active,input [8:0] owner_states,output [110:0] diagnostic',obs_in,{f'diagnostic[{i}]':t for i,t in enumerate(diag)})
 return boundary,observer,dict(boundary=bp,observer=op,page_mux_bits=111,page_mux_truth_checks=truth,map_sha256=sha(board/'outflow/ti60f225_oob.map.v'))

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists();out.mkdir()
 started=time.monotonic();b,o,proof=generate(BOARD)
 (out/'boundary.v').write_text(b);(out/'observer.v').write_text(o)
 files=[ROOT/'tmp/board-test/tb_camera_boundary_recovery.sv',ROOT/'tmp/board-test/tb_camera_recovery_bundle.sv',PARENT/'assembly/raw32_validated_packer.sv']
 for f in files:shutil.copy2(f,out/f.name)
 for name in ('video_frame_boundary_detector.sv','camera_stream_fault_observer.sv'):
  body=(BOARD/name).read_text().replace('module video_frame_boundary_detector','module reference_boundary').replace('module camera_stream_fault_observer','module reference_observer')
  (out/('reference_'+name)).write_text(body)
 models=[]
 for kind in sorted(set(proof['boundary']['kinds']+proof['observer']['kinds'])):
  f=EF/'sim_models/verilog'/(kind.lower()+'.v');assert 'pragma protect' not in f.read_text();shutil.copy2(f,out/f.name);models.append(f.name)
 inputs={f.name:sha(f) for f in out.iterdir() if f.suffix in ('.v','.sv')};env=os.environ.copy();env['TEMP']=env['TMP']=str(out);cases=[]
 def run(label,args,budget=20):
  r=subprocess.run([str(MS/args[0]),*args[1:]],cwd=out,env=env,capture_output=True,text=True,timeout=budget)
  log=r.stdout+r.stderr;(out/(label+'.log')).write_text(log);assert r.returncode==0 and '** Error:' not in log and '** Fatal:' not in log,(label,r.returncode,log[-2400:]);return log
 run('library',['vlib.exe','work']);run('compile',['vlog.exe','-sv','-timescale','1ns/1ps','boundary.v','observer.v',*[f.name for f in files],*[f.name for f in out.glob('reference_*.sv')],*models])
 for active,case in itertools.product((0,1),range(10)):
  label=f'a{active}-c{case}';log=run(label,['vsim.exe','-c','work.tb_camera_boundary_recovery',f'-gACTIVE={active}',f'-gCASE={case}','-do','run -all; quit -code 0'])
  assert 'PASS CAMERA_RECOVERY' in log;cases.append(dict(active=active,case=case,log_sha256=sha(out/(label+'.log'))))
 log=run('observer',['vsim.exe','-c','work.tb_camera_recovery_bundle','-do','run -all; quit -code 0']);assert 'PASS CAMERA_MAPPED_BUNDLE' in log
 assert all(sha(out/n)==h for n,h in inputs.items())
 result=dict(status='CAMERA_RECOVERY_ACTUAL_MAP_SHORT_PASS',map_sha256=proof['map_sha256'],proof=proof,inputs=inputs,runner_sha256=sha(__file__),cases=cases,observer_log_sha256=sha(out/'observer.log'),seconds=time.monotonic()-started,hardware_access=False)
 (out/'result.json').write_text(json.dumps(result,indent=2));print(result['status'],result['seconds'],proof['page_mux_truth_checks'])
if __name__=='__main__':main()
