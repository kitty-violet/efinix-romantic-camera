"""Cut actual mapped observation inputs at the same held/raw sample boundary.

The existing old native observer is the serial-format reference. Service-page
content is tested separately; here each actual held-D input gets that cell's
reference probe, preserving shared drivers and constants explicitly.
"""
import argparse,json,pickle,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from video_service_graph import load
from video_service_serializer_native import P,B,net
from native_local_drivers import CELL,render_token
from continuous_control_native import connections
from camera_joint_native import OUT

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();ref=a.reference.resolve();root=a.root.resolve()
 assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 meta=json.loads((ref/'inputs.json').read_text())
 for f,h in meta['files'].items():assert sha(f)==h,f
 src=board/'outflow/ti60f225_oob.map.v';key,g=pickle.loads((board/'service-validation-graph.pkl').read_bytes());assert key[0]==sha(src)
 D=load();old=D['g'];drivers,_=connections(g)
 def mapped(t):
  n=D['drivers'].get(t)
  if n is None:return t
  k,pins=old[n];assert g[n][0]==k
  options=[g[n][1][p][ts.index(t)] for p,ts in pins.items() if p in OUT[k] and t in ts]
  assert len(options)==1,(t,n,options)
  return options[0]
 special={net('process_clk_150'):'system_clk',net('jtag_inst1_TCK'):'tck',net(P+'n2987'):'capture',net(B+'n8'):'shift',net(B+'jtag_wrapper_header',1):'poll'}
 sample_names=sorted(n for n in old if n.startswith(P+'held_payload['))
 sample_names += [P+f'raw_meta[{i}]~FF' for i in range(8)]
 probes=sorted({old[n][1]['D'][0] for n in sample_names}-set(special),key=str);probes=[t for t in probes if t[0]=='net']
 cuts={mapped(t):v for t,v in special.items()}
 for n in sample_names:
  ot=old[n][1]['D'][0];nt=g[n][1]['D'][0]
  if nt[0]=='literal':assert nt==ot,(n,nt,ot);continue
  value=render_token(ot) if ot[0]=='literal' else f'probes[{probes.index(ot)}]'
  assert nt not in cuts or cuts[nt]==value,(n,nt,cuts.get(nt),value)
  cuts[nt]=value
 owned={n for n,(k,_) in g.items() if n.startswith((P,'service_serializer/')) and k not in ('EFX_LUT4','EFX_ADD')}
 roots=[t for n in owned for pin,ts in g[n][1].items() if pin in OUT[g[n][0]] for t in ts]+[net('service_serial_tdo')]
 selected=set();todo=roots[:];unknown=[]
 while todo:
  t=todo.pop()
  if t[0]=='literal' or t in cuts:continue
  n=drivers.get(t)
  if n is None:unknown.append(t);continue
  if n in selected:continue
  k,pins=g[n];assert k in ('EFX_LUT4','EFX_ADD') or n in owned,(n,t)
  selected.add(n)
  for pin,ts in pins.items():
   if pin not in OUT[k]:todo.extend(ts)
 assert not unknown,unknown[:10]
 text=src.read_text();decl='\n'.join(re.findall(r'^\s*wire\b[^;]*;',text,re.M))
 assigns='\n'.join('assign '+render_token(t)+'='+v+';' for t,v in cuts.items())
 seq=render_token(net(P+'sequence_number'));req=render_token(mapped(net(P+'request_toggle')))
 outputs=f'assign tdo=service_serial_tdo;\nassign sequence_number={seq};\nassign busy={req}^{seq}[0];\nassign valid={render_token(mapped(net(P+"valid")))};'
 cells='\n'.join(m[0] for m in CELL.finditer(text) if m[2].lstrip('\\') in selected)
 pars='\n'.join(m[0] for m in re.finditer(r'^\s*defparam\s+(\S+?)\s*\.\w+\s*=[^;]*;',text,re.M) if m[1].lstrip('\\') in selected)
 ports=f'input wire system_clk,tck,capture,poll,shift,input wire[{len(probes)-1}:0]probes,output wire tdo,busy,valid,output wire[7:0]sequence_number'
 root.mkdir();(root/'new.sv').write_text(f'module observer_new({ports});\n'+decl+'\n'+assigns+'\n'+outputs+'\n'+cells+'\n'+pars+'\nendmodule\n')
 for name in ('old.sv','tb.sv'):shutil.copy2(ref/name,root/name)
 for k in {g[n][0] for n in selected}|{'EFX_FF','EFX_LUT4','EFX_ADD','EFX_SRL8'}:shutil.copy2(EF/'sim_models/verilog'/(k.lower()+'.v'),root/(k.lower()+'.v'))
 files={str(f):sha(f) for f in root.iterdir()};files.update({str(src):sha(src),str(Path(__file__)):sha(__file__)})
 (root/'inputs.json').write_text(json.dumps(dict(files=files,reference_inputs_sha256=sha(ref/'inputs.json'),selected=len(selected),held_raw_input_cells=len(sample_names),board_release=False),indent=2))
 print('ACTUAL_MAP_SERIAL_COMPARISON_PREPARED',len(selected),len(probes))
if __name__=='__main__':main()
