"""Offline candidate: sensor writes and verification bytes share one BRAM.

No board access or implementation release. The existing ROM waits, address
checks, masks, ACK checks, error handling and clocks are retained. A separate
verification bank stores bytes derived from the final write to each register.
"""
import argparse,copy,json,pickle,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from prepare_video_service_color_exposure import read_rom
from native_local_drivers import CELL

PARENT=ROOT/'tmp/frame-execution/connected-board-video-service-probe60-input-isp-explicit'
MAP=PARENT/'outflow/ti60f225_oob.map.v'
REGS=[0x3018,0x3031,0x3037,0x3208,0x3209,0x320a,0x320b,0x0100,0x3221,0x3e08,0x3e09]
MASKS=[0xe0,0x0f,0x60]+[0xff]*4+[1]+[0xff]*3
RAM='camera_checked/config_rom/ROM__D$2'

def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def prepare(root,gain):
 assert not root.exists() and root.is_relative_to(ROOT/'tmp/frame-execution')
 assert gain in (0x2a,0x34)
 assert sha(MAP)=='21bfdc8a33626feed493cddc74fd73f7c191c7ad33346c2c520f6a6582407c26'
 key,_,old,_,op=pickle.loads((PARENT/'retained-audit-parse.pkl').read_bytes());assert key[1]==sha(MAP)
 original=MAP.read_text(encoding='utf8');words,blocks=read_rom(original)
 newwords=list(words)
 for i,reg,val in [(116,0x3e01,0x46),(117,0x3e02,0),(162,0x3e09,gain)]:
  assert words[i]>>9==reg;newwords[i]=reg<<9|val<<1
 final={w>>9:(w>>1)&255 for w in newwords}
 defaults=dict(zip(REGS,[0x60,0x0a,0x20,7,0x80,4,0x38,1,0x66,0x80,0x2a]))
 # Several checked registers retain sensor defaults rather than receiving a
 # write. Preserve those explicit expectations; do not invent ROM writes.
 values=[final.get(r,defaults[r]) for r in REGS]
 assert values[:10]==[0x60,0x0a,0x20,7,0x80,4,0x38,1,0x66,0x80]
 g=copy.deepcopy(old);ps=copy.deepcopy(op)
 lo,hi,block,width,blob=next(b for b in blocks if b[3]==20)
 updated=blob
 for i,w in enumerate(newwords):updated=(updated&~(0xfffff<<(i*20)))|((w&0xfffff)<<(i*20))
 for i,v in enumerate(values):
  assert (blob>>((256+i)*20))&0xfffff==0
  updated=(updated&~(0xfffff<<((256+i)*20)))|(v<<1<<((256+i)*20))
 changed=block
 for i in range(40):
  a=(blob>>(256*i))&((1<<256)-1);b=(updated>>(256*i))&((1<<256)-1)
  if a==b:continue
  param='INIT_'+format(i,'X');ps[RAM][param]=(256,b)
  changed,n=re.subn('('+param+r"\s*=\s*256'b)"+f'{a:0256b}',lambda m:m[1]+f'{b:0256b}',changed);assert n==2
 pin=g[RAM][1]['RADDR'];assert pin[0]==('literal',"1'b0",None) and len(pin)==9
 pin[0]=('net','n80869_2',None)
 changed,n=re.subn(r"(\.RADDR\(\{)1'b0,",r'\g<1>n80869_2,',changed);assert n==1
 assert op['LUT__241150']['LUTMASK']==(16,0x0100)
 assert old['LUT__241150'][1]=={'I0':[('net','camera_config_diagnostic',60)],'I1':[('net','camera_config_diagnostic',61)],'I2':[('net','camera_config_diagnostic',63)],'I3':[('net','camera_config_diagnostic',62)],'O':[('net','n80869_2',None)]}
 body=original[:lo]+changed+original[hi:];edits=[];extra='\nwire '+','.join(f'sensor_rom_expected_{b}' for b in range(8))+';\n'
 for b in range(8):
  n=f'camera_config_diagnostic[{16+b}]~FF';before=g[n][1]['D'][0];assert before==('net','camera_checked/n229',b)
  assert g[n][1]['CE']==[('net','camera_checked/n1950',None)] and g[n][1]['CLK']==[('net','CLK_5M',None)]
  new=f'sensor_rom_expected_{b}';g[n][1]['D']=[('net',new,None)]
  pattern=rf'(EFX_FF\s+\\{re.escape(n)}\s+\(\.D\()([^)]*)(\))'
  m=re.search(pattern,body);assert m and m[2].strip()==rf'\camera_checked/n229 [{b}]'
  edits.append((pattern,m[2],new));body,count=re.subn(pattern,lambda m:m[1]+new+m[3],body);assert count==1
  name=f'sensor_rom_expected_mux{b}'
  g[name]=('EFX_LUT4',{'I0':[before],'I1':[('net','camera_checked/rom_data',b+1)],'I2':[('net','n80869_2',None)],'O':[('net',new,None)]});ps[name]={'LUTMASK':(16,0xcaca)}
  extra+=rf'EFX_LUT4 {name}(.I0(\camera_checked/n229 [{b}]),.I1(\camera_checked/rom_data [{b+1}]),.I2(n80869_2),.O({new}));'+'\n'
  extra+=f"defparam {name}.LUTMASK=16'hcaca;\n"
 assert len(re.findall(r'^endmodule\b',body,re.M))==1
 body=re.sub(r'^endmodule\b',lambda m:extra+'endmodule',body,flags=re.M)
 restored=body.replace(extra,'')
 for pattern,before,new in edits:restored,count=re.subn(pattern,lambda m:m[1]+before+m[3],restored);assert count==1
 assert restored[:lo]+block+restored[lo+len(changed):]==original
 # Count the exact graph changes independently; no new register or clock.
 assert set(g)-set(old)=={f'sensor_rom_expected_mux{b}' for b in range(8)}
 assert {n for n in old if old[n]!=g[n]}=={RAM}|{f'camera_config_diagnostic[{16+b}]~FF' for b in range(8)}
 for i in range(16):assert (0xcaca>>i)&1==((i>>1)&1 if i&4 else i&1)
 # Extract the actual primitive circuit, unchanged official models.
 outputs={'EFX_LUT4':['O'],'EFX_FF':['Q'],'EFX_ADD':['O','CO'],'EFX_RAM10':['RDATA']}
 drivers={t:n for n,(k,pins) in g.items() for pin in outputs.get(k,[]) for t in pins.get(pin,[])}
 boundary={'CLK_5M':'clk','w_i2c_arstn':'reset_n','io_cam_scl_IN':'scl_i','io_cam_sda_IN':'sda_i'}
 roots={p:('net',n,None) for p,n in [('scl_oen','camera_new_scl_oen'),('sda_oen','camera_new_sda_oen'),('config_ok','camera_config_ok'),('config_failed','camera_config_failed')]}
 roots.update({f'diagnostic[{i}]':('net','camera_config_diagnostic',i) for i in range(64) if i!=59})
 stack=list(roots.values());selected=set();outside=set()
 while stack:
  t=stack.pop()
  if t[0]=='literal':continue
  if t[1] in boundary:outside.add(t);continue
  assert t in drivers,('undriven',t)
  n=drivers[t]
  if n in selected:continue
  selected.add(n);k,pins=g[n];assert k in outputs
  stack.extend(t for pin,ts in pins.items() if pin not in outputs[k] for t in ts)
 nets={t:f'n{i}' for i,t in enumerate(sorted({t for n in selected for ts in g[n][1].values() for t in ts if t[0]!='literal'}))}
 lines=['module camera_checked_config #(parameter PRESCALE=32)(input clk,reset_n,scl_i,sda_i,output scl_o,scl_oen,sda_o,sda_oen,config_ok,config_failed,output [63:0] diagnostic);','wire '+','.join(nets.values())+';']
 for t in sorted(outside):lines.append(f'assign {nets[t]}={boundary[t[1]]};')
 for port,t in roots.items():lines.append(f'assign {port}={nets[t]};')
 lines+=['assign diagnostic[59]=0; assign scl_o=0; assign sda_o=0;']
 def expr(ts):
  v=[t[1] if t[0]=='literal' else nets[t] for t in ts];return v[0] if len(v)==1 else '{'+','.join(v)+'}' if v else ''
 for n in sorted(selected):
  k,pins=g[n];lines.append(k+' \\'+n+' ('+','.join('.'+p+'('+expr(ts)+')' for p,ts in pins.items())+');')
  for p,v in ps[n].items():lines.append('defparam \\'+n+' .'+p+'='+(str(v) if not isinstance(v,tuple) else f"{v[0]}'h{v[1]:x}")+';')
 lines.append('endmodule');root.mkdir()
 (root/'native_source.v').write_text(body,encoding='utf8');(root/'mapped_camera.v').write_text('\n'.join(lines),encoding='utf8')
 (root/'expected.hex').write_text(''.join(f'60{w>>9:04x}{(w>>1)&255:02x}\n' for w in newwords))
 tb=(HERE/'tb_camera_checked_config.sv').read_text()
 tb=tb.replace("if(mode==3 && address==16'h3107)","if(mode==10 && address==16'h3e09) send_byte=8'h"+('2a' if gain==0x34 else '34')+";\n          if(mode==3 && address==16'h3107)")
 tb=tb.replace('mode>=7','(mode>=7 && mode<=9)')
 tb=tb.replace('if(mode==5 && diagnostic',f'if(mode==10 && (diagnostic[59:56]!=5 || diagnostic[47:32]!=16\'h3e09 || diagnostic[23:16]!=8\'h{gain:02x})) $fatal(1,"GAIN_MISMATCH_NOT_REJECTED");\n      if(mode==5 && diagnostic')
 (root/'tb.sv').write_text(tb)
 kinds=sorted({g[n][0] for n in selected})
 for k in kinds:shutil.copy2(EF/'sim_models/verilog'/(k.lower()+'.v'),root/(k.lower()+'.v'))
 report=dict(status='SENSOR_ROM_VERIFY_CANDIDATE_NOT_SYNTHESIZED',original_map_sha256=sha(MAP),gain_register=gain,exposure_half_lines=1120,
  verification_bank_base=256,verification_entries=[dict(address=r,expected=v,mask=m) for r,v,m in zip(REGS,values,MASKS)],
  written_register_expectations_from_final_write=True,unwritten_registers_keep_existing_default_expectations=[r for r in REGS if r not in final],new_luts=8,new_flip_flops=0,new_clocks=0,changed_ff_d_pins=8,changed_ram_address_pins=1,
  all_unlisted_source_byte_equal=True,cells=len(selected),kinds=kinds,hardware_access=False,board_release=False,
  source_sha256=sha(root/'native_source.v'),script_sha256=sha(__file__),files={str(f):sha(f) for f in root.iterdir() if f.is_file()})
 save(root/'inputs.json',report);print(json.dumps({k:report[k] for k in ('status','gain_register','cells','new_luts')}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gain-register',type=lambda v:int(v,0),default=0x34);a=p.parse_args();prepare(a.root.resolve(),a.gain_register)
