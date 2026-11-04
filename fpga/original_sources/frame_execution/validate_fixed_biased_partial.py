"""Short independent INT32-bias accumulation and official-DSP quantization gates."""
import argparse,json,os,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--kind',choices=['sum','rq'],required=True);p.add_argument('--split',action='store_true');p.add_argument('--rtl',type=Path);p.add_argument('--testbench',type=Path);p.add_argument('--dot-width',type=int,choices=[19,20,21],default=21);p.add_argument('--registered-scale-class',action='store_true');p.add_argument('--registered-bias-class',action='store_true');a=p.parse_args()
assert not a.registered_bias_class or a.registered_scale_class
assert not a.registered_scale_class or (a.kind=='rq' and a.split and a.rtl)
assert a.dot_width==21 or (a.kind=='sum' and a.split and a.rtl)
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
name='partial' if a.kind=='sum' else 'rq'
style='split' if a.split else 'biased'
top='tb_fixed_'+style+'_'+name;marker='FIXED_'+style.upper()+'_'+name.upper()
folder='fixed_split_bias' if a.split else 'fixed_biased_partial'
v=HERE/'variants'/folder
files=[v/('fixed_partial32.sv' if a.kind=='sum' else 'fixed_rq32_mixed.sv'),HERE/'variants'/(folder+'_test')/(top+'.sv')]
if a.testbench:
 candidate=a.testbench.resolve();assert candidate.is_relative_to(HERE/'variants') and candidate.name==top+'.sv'
 files[1]=candidate
if a.rtl:
 candidate=a.rtl.resolve();assert candidate.is_relative_to(HERE/'variants') and candidate.name==files[0].name
 files[0]=candidate
if a.kind=='rq':files += [EF/'sim_models/verilog'/n for n in ('efx_srl8.v','efx_dsp48.v')]
records=[]
for f in files:
 d=r/f.name
 if f.name==top+'.sv' and a.dot_width<21:
  tb=f.read_text();assert 'fixed_partial32 dut(.*);' in tb
  tb=tb.replace('fixed_partial32 dut(.*);',f'fixed_partial32 #(.DOT_WIDTH({a.dot_width})) dut(.*);')
  tb=tb.replace('1048575',str((1<<(a.dot_width-1))-1)).replace('1048576',str(1<<(a.dot_width-1)))
  tb=tb.replace("21'($random(seed))",f"21'($signed({a.dot_width}'($random(seed))))")
  insert_at='  out_ready=1;phase(450,0,1);'
  assert tb.count(insert_at)==1
  rejected=f'''  allow_error=1;in_first=1;in_last=1;in_parts=1;in_x=0;in_dots=0;in_dots[0+:21]=21'd{1<<(a.dot_width-1)};
  in_valid=1;@(negedge clk);in_valid=0;
  if(!error || out_valid)$fatal(1,"DOT_RANGE_NOT_REJECTED");
  cancel=1;@(negedge clk);cancel=0;allow_error=0;
  $display("DOT_RANGE_REJECTED width={a.dot_width}");
'''
  tb=tb.replace(insert_at,rejected+insert_at);d.write_text(tb)
 elif f.name==top+'.sv' and a.registered_scale_class:
  tb=f.read_text();assert tb.count('fixed_rq32_mixed dut(.*);')==1
  params='.REGISTER_SCALE_CLASS(1)'+(',.REGISTER_BIAS_CLASS(1)' if a.registered_bias_class else '')
  tb=tb.replace('fixed_rq32_mixed dut(.*);','fixed_rq32_mixed #('+params+') dut(.*);')
  # Production presents the parameter RAM view before any MAC result.
  # Exercise that contract explicitly, including each interrupted correction.
  for token in ('   in_valid=7;out_ready=0;','   payload(0,450);in_valid=1;'):
   assert tb.count(token)==1;tb=tb.replace(token,'   @(negedge clk);\n'+token)
  d.write_text(tb)
 else:shutil.copy2(f,d)
 records.append(dict(source=str(f),snapshot=str(d),sha256=sha(d),source_sha256=sha(f)))
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2)+'\n')
run([MS/'vlib.exe','work'],r,'vlib.log');body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[x['snapshot'] for x in records]],r,'compile.log');assert 'Errors: 0, Warnings: 0' in body
scratch=r/'temp';scratch.mkdir();env=os.environ.copy();env.update(TEMP=str(scratch),TMP=str(scratch),TMPDIR=str(scratch))
t=time.monotonic();log=r/'simulation.log'
with log.open('w',encoding='utf-8') as f:
 proc=subprocess.run(list(map(str,[MS/'vsim.exe','-c','work.'+top,'-do','onerror {quit -code 1}; run -all; quit -code 0'])),cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
body=log.read_text();assert proc.returncode==0 and not re.search(r'(?m)^#?\s*(FAIL|\*\* (Error|Fatal))',body),body[-2000:]
lines=re.findall(r'^# (PASS '+marker+r'.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
for x in records:assert sha(x['source'])==x['source_sha256'] and sha(x['snapshot'])==x['sha256']
if a.dot_width<21:assert f'DOT_RANGE_REJECTED width={a.dot_width}' in body
result=dict(status=marker+'_PASS',line=lines[0],dot_width=a.dot_width,registered_scale_class=a.registered_scale_class,registered_bias_class=a.registered_bias_class,seconds=time.monotonic()-t,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(log))
(r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
