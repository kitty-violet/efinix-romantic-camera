"""Pin-level test of the current mapped controller with an official ROM overlay."""
import argparse,collections,json,os,re,shutil,subprocess,time
from pathlib import Path
from native_local_drivers import graph
from probe_full_native_split_map import params
from run_units import ROOT,HERE,EF,sha
from windows_process_tree_safe import stop_launched_tree


def extract(mapped, candidate):
 text=mapped.read_text(encoding='utf8');g=graph(text);ps=params(text)
 outputs={'EFX_LUT4':['O'],'EFX_FF':['Q'],'EFX_ADD':['O','CO'],'EFX_RAM10':['RDATA']}
 drivers={t:n for n,(k,p) in g.items() for pin in outputs.get(k,[]) for t in p.get(pin,[])}
 boundary={'CLK_5M':'clk','w_i2c_arstn':'reset_n','io_cam_scl_IN':'scl_i','io_cam_sda_IN':'sda_i'}
 roots={p:('net',n,None) for p,n in [('scl_oen','camera_new_scl_oen'),('sda_oen','camera_new_sda_oen'),('config_ok','camera_config_ok'),('config_failed','camera_config_failed')]}
 for i in range(64):
  t=('net','camera_config_diagnostic',i)
  if t not in drivers:assert i==59
  else:roots[f'diagnostic[{i}]']=t
 stack=list(roots.values());selected=set();outside=set()
 while stack:
  t=stack.pop()
  if t[0]=='literal':continue
  if t[1] in boundary:outside.add(t);continue
  assert t in drivers,('unbound_actual_camera_input',t)
  n=drivers[t]
  if n in selected:continue
  selected.add(n);k,p=g[n];assert k in outputs,(n,k)
  stack.extend(t for pin,ts in p.items() if pin not in outputs[k] for t in ts)
 used={t for n in selected for ts in g[n][1].values() for t in ts if t[0]!='literal'}
 nets={t:f'n{i}' for i,t in enumerate(sorted(used))}
 lines=['module camera_checked_config #(parameter PRESCALE=32)(input clk,reset_n,scl_i,sda_i,output scl_o,scl_oen,sda_o,sda_oen,config_ok,config_failed,output [63:0] diagnostic);','wire '+','.join(nets.values())+';']
 for t in sorted(outside):lines.append(f'assign {nets[t]}={boundary[t[1]]};')
 for port,t in roots.items():lines.append(f'assign {port}={nets[t]};')
 lines += ['assign diagnostic[59]=0; assign scl_o=0; assign sda_o=0;']
 patches=json.loads((candidate/'result.json').read_text(encoding='utf8'))['native_init_patches'];changed=[]
 target='camera_checked/config_rom/ROM__D$2'
 for n in sorted(selected):
  k,p=g[n]
  def expr(ts):
   v=[t[1] if t[0]=='literal' else nets[t] for t in ts]
   return v[0] if len(v)==1 else '{'+','.join(v)+'}' if v else ''
  lines.append(k+' \\'+n+' ('+','.join('.'+pin+'('+expr(ts)+')' for pin,ts in p.items())+');')
  for key,value in ps[n].items():
   if n==target:
    for patch in patches:
     if key==patch['parameter']:
      assert value==(256,int(patch['old'],2));value=(256,int(patch['new'],2));changed.append(key)
   v=str(value) if not isinstance(value,tuple) else f"{value[0]}'h{value[1]:x}"
   lines.append('defparam \\'+n+' .'+key+'='+v+';')
 assert changed==[p['parameter'] for p in patches]
 lines.append('endmodule')
 return '\n'.join(lines),dict(cells=len(selected),kinds=sorted({g[n][0] for n in selected}),external_inputs=sorted(t[1] for t in outside),only_parameters_changed=changed)


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
 p.add_argument('--stage',choices=['prepare','simulate'],required=True);a=p.parse_args()
 root=a.root.resolve();candidate=a.candidate.resolve();assert root.is_relative_to(ROOT/'tmp')
 if a.stage=='prepare':
  assert not root.exists();root.mkdir()
  mapped=ROOT/'tmp/frame-execution/video-input-bit2/outflow/ti60f225_oob.map.v'
  body,proof=extract(mapped,candidate);(root/'mapped_camera.v').write_text(body)
  rom=(candidate/'candidate_i2c_master_reg_rom.v').read_text(encoding='utf8')
  clean=re.sub(r'/\*.*?\*/|//[^\n]*','',rom,flags=re.S)
  rows=re.findall(r"ROM\[8'h([0-9a-f]+)\]\s*=\s*\{16'h([0-9a-f]+),\s*8'h([0-9a-f]+),\s*1'b([01])\}",clean,re.I)
  assert len(rows)==164 and [int(x[0],16) for x in rows]==list(range(164))
  (root/'expected.hex').write_text(''.join(f'60{r}{v}\n' for _,r,v,_ in rows))
  shutil.copy2(HERE/'tb_camera_checked_config.sv',root/'tb.sv')
  for kind in proof['kinds']:shutil.copy2(EF/'sim_models/verilog'/(kind.lower()+'.v'),root/(kind.lower()+'.v'))
  # Icarus can evaluate initially-zero LUT pins before the continuous lutrom
  # assignment. Re-evaluate when lutrom initializes; no truth-table change.
  official=root/'efx_lut4.v';source=official.read_text()
  old='always @(I0_net or I1_net or I2_net or I3_net)';assert source.count(old)==1
  adapted=source.replace(old,'always @(I0_net or I1_net or I2_net or I3_net or lutrom)')
  (root/'efx_lut4_icarus.v').write_text(adapted)
  masks=sorted(set(re.findall(r"\.LUTMASK=16'h([0-9a-f]+);",body)))
  checks=['`timescale 1ns/1ps','module tb_lut_semantics; reg [3:0] x=0; integer code,i; reg [1:0] digit;',
   'function automatic refbit(input [15:0] mask,input [3:0] pins); integer k,j; reg match,found,value; begin found=0;value=0; for(k=0;k<16;k=k+1)begin match=1;for(j=0;j<4;j=j+1)if(pins[j]!==1\'bx && pins[j]!==((k>>j)&1))match=0; if(match)begin if(!found)begin value=mask[k];found=1;end else if(value!==mask[k])value=1\'bx;end end refbit=value;end endfunction']
  for i,mask in enumerate(masks):
   checks.append(f"wire y{i}; EFX_LUT4 #(.LUTMASK(16'h{mask})) l{i}(.I0(x[0]),.I1(x[1]),.I2(x[2]),.I3(x[3]),.O(y{i}));")
  checks.append('task check; begin #1;')
  for i,mask in enumerate(masks):
   checks.append(f"if(y{i}!==refbit(16'h{mask},{{l{i}.I3_net,l{i}.I2_net,l{i}.I1_net,l{i}.I0_net}})) $fatal(1,\"LUT_TRUTH mask={mask} x=%b got=%b\",x,y{i});")
  checks.append('end endtask initial begin check();for(code=0;code<256;code=code+1)begin for(i=0;i<4;i=i+1)begin digit=(code>>(2*i))&3;case(digit)0:x[i]=0;1:x[i]=1;2:x[i]=1\'bx;3:x[i]=1\'bz;endcase end check();end $display("PASS LUT_INITIAL_AND_FOUR_STATE masks=%0d cases_per_mask=257",'+str(len(masks))+');$finish;end endmodule')
  (root/'tb_lut_semantics.sv').write_text('\n'.join(checks))
  proof['simulation_only_lut_adaptation']=dict(original_sha256=sha(official),adapted_sha256=sha(root/'efx_lut4_icarus.v'),change='add lutrom initialization to sensitivity; truth function unchanged',unique_actual_masks=len(masks),hardware_modified=False)
  files={str(f):sha(f) for f in root.iterdir() if f.suffix in ('.v','.sv','.hex')}
  meta=dict(base_map_sha256=sha(mapped),candidate_sha256=sha(candidate/'result.json'),files=files,proof=proof,runner_sha256=sha(__file__),hardware_access=False)
  (root/'inputs.json').write_text(json.dumps(meta,indent=2));print('ACTUAL_CAMERA_EXTRACTED_START_FRESH_SIMULATION_PROCESS',flush=True);return
 # Preserve managed TMP/TEMP. Explicit stdin PIPE is required by the installed
 # Icarus driver in this host; inherit/DEVNULL failed in the bounded A/B probe.
 meta=json.loads((root/'inputs.json').read_text());proof=meta['proof'];files=meta['files']
 assert meta['runner_sha256']==sha(__file__) and meta['candidate_sha256']==sha(candidate/'result.json')
 assert not (root/'result.json').exists() and not (root/'compile.log').exists()
 for f,h in files.items():assert sha(f)==h
 Q=ROOT/'tools/iverilog/ucrt64/bin';env=os.environ.copy();env['PATH']=str(Q)+';'+env.get('PATH','')
 def run(args,label,limit):
  start=time.monotonic()
  proc=subprocess.Popen(list(map(str,args)),cwd=root,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  try:s,_=proc.communicate(timeout=limit)
  except subprocess.TimeoutExpired:
   (root/(label+'.timeout.json')).write_text(json.dumps(stop_launched_tree(proc),indent=2));raise
  (root/label).write_text(s);rc=proc.returncode
  assert rc==0 and not re.search(r'\*\* (?:Error|Fatal)',s),(label,rc,s[-1800:])
  return s,dict(log_sha256=sha(root/label),seconds=time.monotonic()-start,exit_code=rc)
 run([Q/'iverilog.exe','-g2012','-s','tb_lut_semantics','-o','lut.vvp','tb_lut_semantics.sv','efx_lut4_icarus.v'],'lut-compile.log',45)
 text,lut=run([Q/'vvp.exe','lut.vvp'],'lut-truth.log',20);assert 'PASS LUT_INITIAL_AND_FOUR_STATE' in text;print(text,flush=True)
 run([Q/'iverilog.exe','-g2012','-s','tb_camera_checked_config','-o','sim.vvp','mapped_camera.v','tb.sv',*[('efx_lut4_icarus.v' if k=='EFX_LUT4' else k.lower()+'.v') for k in proof['kinds']]],'compile.log',45)
 cases=[]
 for mode in (0,4,6,8):
  s,r=run([Q/'vvp.exe','sim.vvp',f'+MODE={mode}'],f'mode{mode}.log',120)
  assert 'PASS CHECKED_CAMERA' in s;sline=next(x for x in s.splitlines() if 'PASS CHECKED_CAMERA' in x)
  cases.append(dict(mode=mode,pass_line=sline,**r));print(sline,flush=True)
 for f,h in files.items():assert sha(f)==h
 result=dict(status='ACTUAL_MAPPED_EXPOSURE_CONTROLLER_PIN_TESTS_PASS',cases=cases,lut_semantics=lut,inputs_sha256=sha(root/'inputs.json'),hardware_access=False,board_release=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(result['status'],flush=True)


if __name__=='__main__':main()
