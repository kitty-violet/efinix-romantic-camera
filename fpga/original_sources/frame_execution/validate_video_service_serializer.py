"""Actual original native observer versus indexed-serializer graft, no board."""
import argparse,json,re,shutil,time
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,sha,run
from video_service_graph import load,SOURCE
from video_service_serializer_native import extension,P,B,net
from native_local_drivers import CELL,render_token,rewrite_expression
from camera_joint_native import OUT

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
 assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir();started=time.monotonic()
 D=load();g=D['g'];text=SOURCE.read_text();removed=set();edits={};supplied={}
 extra,proof=extension(D,removed,edits,supplied)
 special={net('process_clk_150'):'system_clk',net('jtag_inst1_TCK'):'tck',net(P+'n2987'):'capture',
          net(B+'n8'):'shift',net(B+'jtag_wrapper_header',1):'poll'}
 probes=set(pins['D'][0] for n,(k,pins) in g.items() if n.startswith(P+'held_payload['))
 probes.update(g[P+f'raw_meta[{i}]~FF'][1]['D'][0] for i in range(8))
 probes=sorted((t for t in probes-set(special) if t[0]=='net'),key=str);cuts=dict(special)
 cuts.update({t:f'probes[{i}]' for i,t in enumerate(probes)})
 owned={n for n,(k,_) in g.items() if n.startswith(P) and k not in ('EFX_LUT4','EFX_ADD')}
 owned.add(B+'diagnostic_tdo~FF');selected=set();unknown=[]
 todo=[t for n in owned for pin,ts in g[n][1].items() if pin in OUT[g[n][0]] for t in ts]
 while todo:
  t=todo.pop()
  if t[0]=='literal' or t in cuts:continue
  n=D['drivers'].get(t)
  if n is None:unknown.append(t);continue
  if n in selected:continue
  k,pins=g[n];assert k in ('EFX_LUT4','EFX_ADD') or n in owned,(n,t)
  selected.add(n)
  for pin,ts in pins.items():
   if pin not in OUT[k]:todo.extend(ts)
 assert not unknown,unknown[:10]
 assert removed<=selected,sorted(removed-selected)[:10]
 declarations='\n'.join(re.findall(r'^\s*wire\b[^;]*;',text,re.M))
 assignments='\n'.join('assign '+render_token(t)+'='+expr+';' for t,expr in cuts.items())
 outputs=f'assign tdo={render_token(net(B+"diagnostic_tdo"))};\nassign sequence_number={render_token(net(P+"sequence_number"))};\nassign busy={render_token(net(P+"request_toggle"))}^{render_token(net(P+"sequence_number",0))};\nassign valid={render_token(net(P+"valid"))};'
 for mode in ('old','new'):
  keep=selected if mode=='old' else selected-removed
  def cell(m):
   n=m[2].lstrip('\\')
   if n not in keep:return ''
   if mode=='old' or n not in edits:return m[0]
   pins=re.sub(r'\.(\w+)\(([^()]*)\)',lambda z:'.'+z[1]+'('+rewrite_expression(z[2],edits[n].get(z[1],{}))+')',m[3])
   return m[0][:m.start(3)-m.start()]+pins+m[0][m.end(3)-m.start():]
  body='\n'.join(cell(m) for m in CELL.finditer(text) if m[2].lstrip('\\') in keep)
  pars='\n'.join(m[0] for m in re.finditer(r'^\s*defparam\s+(\S+?)\s*\.\w+\s*=[^;]*;',text,re.M) if m[1].lstrip('\\') in keep)
  prefix='wire service_serial_tdo,service_snapshot_publish,service_snapshot_publish_allowed;\n' if mode=='new' else ''
  addition=extra.replace(prefix.strip(),'') if mode=='new' else ''
  if mode=='new':addition+='\nassign '+render_token(net(B+'diagnostic_tdo'))+'=service_serial_tdo;'
  ports=f'input wire system_clk,tck,capture,poll,shift,input wire[{len(probes)-1}:0]probes,output wire tdo,busy,valid,output wire[7:0]sequence_number'
  (root/(mode+'.sv')).write_text(f'module observer_{mode}({ports});\n'+declarations+'\n'+prefix+assignments+'\n'+outputs+'\n'+body+'\n'+pars+'\n'+addition+'\nendmodule\n')
 source=HERE/'variants/video_service/video_snapshot_serializer.sv';shutil.copy2(source,root/source.name)
 for k in {g[n][0] for n in selected}:shutil.copy2(EF/'sim_models/verilog'/(k.lower()+'.v'),root/(k.lower()+'.v'))
 tb=(HERE/'variants/video_service/tb_video_snapshot_serializer.sv').read_text().replace('PROBE_BITS',str(len(probes)))
 (root/'tb.sv').write_text(tb)
 files={str(f):sha(f) for f in root.iterdir() if f.suffix in ('.v','.sv')}
 files.update({str(f):sha(f) for f in [SOURCE,Path(__file__),HERE/'video_service_serializer_native.py',source,HERE/'variants/video_service/tb_video_snapshot_serializer.sv']})
 (root/'inputs.json').write_text(json.dumps(dict(files=files,selected=len(selected),removed=len(removed),truth=proof),indent=2))
 result=dict(status='RUNNING',hardware_access=False)
 try:
  run([MS/'vlib.exe','work'],root,'vlib.log',20)
  run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*root.glob('*.v'),*root.glob('*.sv')],root,'compile.log',60)
  body,_=run([MS/'vsim.exe','-c','work.tb_video_snapshot_serializer','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],root,'simulate.log',240)
  assert 'PASS VIDEO_SERVICE_INDEXED_SERIAL_NATIVE' in body
  assert not re.search(r'^# \*\* (?:Error|Fatal)',body,re.M)
  assert all(sha(f)==h for f,h in files.items())
  result.update(status='VIDEO_SERVICE_INDEXED_SERIAL_NATIVE_PASS',markers=re.findall(r'^# PASS.*$',body,re.M))
 except Exception as e:result.update(status='FAIL',error=repr(e));raise
 finally:
  result.update(seconds=time.monotonic()-started,inputs_sha256=sha(root/'inputs.json'),board_release=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
