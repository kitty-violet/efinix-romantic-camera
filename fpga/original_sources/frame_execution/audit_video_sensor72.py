"""Audit all retained mapped primitives and extract the actual camera cone."""
import argparse,copy,json,pickle,re,shutil,time
from pathlib import Path
from native_local_drivers import graph,render_token
from probe_full_native_split_map import params,lut_function
from probe_native_open_ports import compare_boundary
from audit_native_roundtrip import tokens
from audit_local_reset_implementation import norm
from run_units import ROOT,HERE,EF,sha


def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();prepared=a.prepared.resolve();assert prepared.is_relative_to(ROOT/'tmp') and not prepared.exists()
 meta=json.loads((board/'inputs.json').read_text());native=Path(meta['native']);proof=json.loads((native/'result.json').read_text())
 parent=Path(meta['parent']);mapped=board/'outflow/ti60f225_oob.map.v';parentmap=parent/'outflow/ti60f225_oob.map.v'
 assert proof['candidate_native_sha256']==sha(board/'native_source.v')
 result=json.loads((board/'result.json').read_text());assert result['status']=='VIDEO_SERVICE_COMPLETE_NATIVE_SYNTHESIS_PROBE_PASS'
 for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
 key,src,old,sp,op=pickle.loads((parent/'retained-audit-parse.pkl').read_bytes());assert key[1]==sha(parentmap)
 cache=board/'sensor72-map-parse.pkl'
 if cache.exists():
  h,g,ps=pickle.loads(cache.read_bytes());assert h==sha(mapped)
 else:
  text=mapped.read_text(encoding='utf8');g=graph(text);ps=params(text)
  cache.write_bytes(pickle.dumps((sha(mapped),g,ps)))
 expected=copy.deepcopy(old);ep=copy.deepcopy(op)
 for patch in proof['native_init_patches']:
  key=patch['parameter'];target='camera_checked/config_rom/ROM__D$2'
  assert ep[target][key]==(256,int(patch['old'],2));ep[target][key]=(256,int(patch['new'],2))
 for bit in range(1,5):
  n=f'camera_config_diagnostic[{bit+16}]~FF'
  assert expected[n][1]['D']==[('net','camera_checked/n229',bit)]
  expected[n][1]['D']=[('net',f'sensor72_expected_{bit}',None)]
  name=f'sensor72_expected_xor{bit}'
  expected[name]=('EFX_LUT4',{'I0':[('net','camera_checked/n229',bit)],'I1':[('net','n80871_2',None)],'I2':[('net','sensor72_index10',None)],'O':[('net',f'sensor72_expected_{bit}',None)]})
  ep[name]={'LUTMASK':(16,0x6a6a)}
 expected['sensor72_index_decode']=('EFX_LUT4',{**{f'I{i}':[('net','camera_config_diagnostic',48+i)] for i in range(4)},'O':[('net','sensor72_index10',None)]})
 ep['sensor72_index_decode']={'LUTMASK':(16,0x0400)}
 assert expected.keys()==g.keys(),('cell_sets',sorted(expected.keys()-g.keys())[:20],sorted(g.keys()-expected.keys())[:20])
 forward={};reverse={}
 for name,(kind,pins) in expected.items():
  nk,np=g[name];assert nk==kind,(name,kind,nk)
  if kind!='EFX_LUT4':assert pins.keys()==np.keys() and ep[name]==ps[name],('params/ports',name)
  for pin in pins.keys()&np.keys():
   xs,ys=pins[pin],np[pin];assert len(xs)==len(ys),(name,pin)
   for x,y in zip(xs,ys):
    if x[0]=='literal' or y[0]=='literal':assert x==y,(name,pin,x,y);continue
    assert x not in forward or forward[x]==y,('forward',name,pin,x,y,forward.get(x))
    assert y not in reverse or reverse[y]==x,('reverse',name,pin,x,y,reverse.get(y))
    forward[x]=y;reverse[y]=x
 def rendered(pins,rename):
  return {p:('' if not vs else render_token(rename.get(vs[0],vs[0])) if len(vs)==1 else '{'+','.join(render_token(rename.get(t,t)) for t in vs)+'}') for p,vs in pins.items()}
 for name,(kind,pins) in expected.items():
  x,y=rendered(pins,forward),rendered(g[name][1],{})
  if kind=='EFX_LUT4':assert lut_function(x,ep[name]['LUTMASK'][1])==lut_function(y,ps[name]['LUTMASK'][1]),('lut',name)
  else:assert x==y,('cell',name)
 renamed={tokens(norm(render_token(x)))[0]:tokens(norm(render_token(y)))[0] for x,y in forward.items()}
 boundary=compare_boundary(parentmap.read_text(),mapped.read_text(),renamed)
 # Validate the existing qualifier against its actual primitive connections.
 q0=old['LUT__241150'][1];assert q0=={'I0':[('net','camera_config_diagnostic',60)],'I1':[('net','camera_config_diagnostic',61)],'I2':[('net','camera_config_diagnostic',63)],'I3':[('net','camera_config_diagnostic',62)],'O':[('net','n80869_2',None)]},q0
 assert op['LUT__241150']['LUTMASK']==(16,0x0100)
 assert old['LUT__241151'][1]=={**{f'I{i}':[('net','camera_config_diagnostic',52+i)] for i in range(4)},'O':[('net','n80870_2',None)]}
 assert op['LUT__241151']['LUTMASK']==(16,1)
 assert old['LUT__241152'][1]=={'I0':[('net','n80869_2',None)],'I1':[('net','n80870_2',None)],'O':[('net','n80871_2',None)]}
 assert op['LUT__241152']['LUTMASK']==(16,0x8888)
 # Every unmodified source file and the original constraints remain bound.
 for row in meta['inputs']:
  rel=Path(row['snapshot']).relative_to(board)
  if rel.as_posix()!='native_source.v':assert sha(parent/rel)==sha(board/rel)
 report=dict(status='SENSOR72_ALL_RETAINED_MAP_PASS',retained=len(old),new_luts=5,expected_ff_d_changes=4,
   other_parameters_connections_equivalent=True,qualifier_verified=True,protected_boundary=boundary,
   map_sha256=sha(mapped),vdb_sha256=sha(board/'outflow/ti60f225_oob.vdb'),source_sha256=sha(board/'native_source.v'),
   evidence={str(f):sha(f) for f in (Path(__file__),native/'result.json',board/'inputs.json',board/'result.json',parentmap,mapped)},hardware_access=False,board_release=False)
 (board/'sensor72-map-audit.json').write_text(json.dumps(report,indent=2))
 print(report['status'],len(old),flush=True)
 # Extract the fully mapped I2C cone, without overriding any parameters.
 outputs={'EFX_LUT4':['O'],'EFX_FF':['Q'],'EFX_ADD':['O','CO'],'EFX_RAM10':['RDATA']}
 drivers={t:n for n,(k,pins) in g.items() for pin in outputs.get(k,[]) for t in pins.get(pin,[])}
 boundary_names={'CLK_5M':'clk','w_i2c_arstn':'reset_n','io_cam_scl_IN':'scl_i','io_cam_sda_IN':'sda_i'}
 roots={p:('net',n,None) for p,n in [('scl_oen','camera_new_scl_oen'),('sda_oen','camera_new_sda_oen'),('config_ok','camera_config_ok'),('config_failed','camera_config_failed')]}
 for i in range(64):
  t=('net','camera_config_diagnostic',i)
  if t not in drivers:assert i==59
  else:roots[f'diagnostic[{i}]']=t
 stack=list(roots.values());selected=set();outside=set()
 while stack:
  t=stack.pop()
  if t[0]=='literal':continue
  if t[1] in boundary_names:outside.add(t);continue
  assert t in drivers,('outside',t)
  n=drivers[t]
  if n in selected:continue
  selected.add(n);k,pins=g[n];assert k in outputs
  stack.extend(t for pin,ts in pins.items() if pin not in outputs[k] for t in ts)
 used={t for n in selected for ts in g[n][1].values() for t in ts if t[0]!='literal'}
 nets={t:f'n{i}' for i,t in enumerate(sorted(used))}
 lines=['module camera_checked_config #(parameter PRESCALE=32)(input clk,reset_n,scl_i,sda_i,output scl_o,scl_oen,sda_o,sda_oen,config_ok,config_failed,output [63:0] diagnostic);','wire '+','.join(nets.values())+';']
 for t in sorted(outside):lines.append(f'assign {nets[t]}={boundary_names[t[1]]};')
 for port,t in roots.items():lines.append(f'assign {port}={nets[t]};')
 lines+=['assign diagnostic[59]=0; assign scl_o=0; assign sda_o=0;']
 for n in sorted(selected):
  k,pins=g[n]
  def expr(ts):
   v=[t[1] if t[0]=='literal' else nets[t] for t in ts]
   return v[0] if len(v)==1 else '{'+','.join(v)+'}' if v else ''
  lines.append(k+' \\'+n+' ('+','.join('.'+pin+'('+expr(ts)+')' for pin,ts in pins.items())+');')
  for key,value in ps[n].items():
   v=str(value) if not isinstance(value,tuple) else f"{value[0]}'h{value[1]:x}"
   lines.append('defparam \\'+n+' .'+key+'='+v+';')
 lines.append('endmodule');prepared.mkdir();(prepared/'mapped_camera.v').write_text('\n'.join(lines))
 rom=(native/'candidate_i2c_master_reg_rom.v').read_text();clean=re.sub(r'/\*.*?\*/|//[^\n]*','',rom,flags=re.S)
 rows=re.findall(r"ROM\[8'h([0-9a-f]+)\]\s*=\s*\{16'h([0-9a-f]+),\s*8'h([0-9a-f]+),\s*1'b([01])\}",clean,re.I)
 assert len(rows)==164 and [int(x[0],16) for x in rows]==list(range(164))
 (prepared/'expected.hex').write_text(''.join(f'60{r}{v}\n' for _,r,v,_ in rows))
 tb=(HERE/'tb_camera_checked_config.sv').read_text()
 tb=tb.replace("if(mode==3 && address==16'h3107)","if(mode==10 && address==16'h3e09) send_byte=8'h2a;\n          if(mode==3 && address==16'h3107)")
 tb=tb.replace('mode>=7','(mode>=7 && mode<=9)')
 tb=tb.replace('if(mode==5 && diagnostic',"if(mode==10 && (diagnostic[59:56]!=5 || diagnostic[47:32]!=16'h3e09 || diagnostic[31:24]!=8'h2a || diagnostic[23:16]!=8'h34)) $fatal(1,\"OLD_GAIN_NOT_REJECTED\");\n      if(mode==5 && diagnostic")
 (prepared/'tb.sv').write_text(tb)
 kinds=sorted({g[n][0] for n in selected})
 for kind in kinds:shutil.copy2(EF/'sim_models/verilog'/(kind.lower()+'.v'),prepared/(kind.lower()+'.v'))
 inputs=dict(map_sha256=sha(mapped),board=str(board),audit_sha256=sha(board/'sensor72-map-audit.json'),
   files={str(f):sha(f) for f in prepared.iterdir() if f.is_file()},cells=len(selected),kinds=kinds,
   actual_models='unmodified official Efinity2026.1',hardware_access=False,runner_sha256=sha(__file__))
 (prepared/'inputs.json').write_text(json.dumps(inputs,indent=2));print('ACTUAL_SENSOR72_CONE_PREPARED',len(selected),flush=True)


if __name__=='__main__':main()
