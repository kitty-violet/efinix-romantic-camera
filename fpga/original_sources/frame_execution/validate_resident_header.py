"""Official RAM check of full resident headers and production image addressing."""
import argparse,json,os,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,sha,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--rtl',type=Path);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
v=HERE/'variants/fixed_resident_header_test'
files=[v/'resident_header_table.sv',v/'tb_resident_header_table.sv',EF/'sim_models/verilog/efx_ram10.v']
if a.rtl:
 assert a.rtl.resolve().is_relative_to(HERE/'variants') and a.rtl.name=='resident_header_table.sv'
 files[0]=a.rtl.resolve()
records=[]
for f in files:
 d=r/f.name;shutil.copy2(f,d);records.append(dict(source=str(f),snapshot=str(d),sha256=sha(f)))
descriptor=ROOT/'tmp/frame-execution/paired-ddr-descriptors1/compute.bin'
data=descriptor.read_bytes();layouts=set()
for at in range(0,len(data),64):
 x=int.from_bytes(data[at:at+64],'little');g=(x>>49)&7;b=(x>>236)&32767;s=(x>>251)&32767;rel=(x>>266)&32767
 assert g in (1,2,4) and b%32==s%32==0 and 12608<=b and b+g*32<=15040 and 15040<=s and s+g*32<=17472 and 17472<=rel and rel+g<=17600
 layouts.add((b,s,rel,g))
assert len(data)//64==8789
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__),descriptor_sha256=sha(descriptor),descriptors=8789,layouts=sorted(layouts)),indent=2)+'\n')
run([MS/'vlib.exe','work'],r,'vlib.log');body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[x['snapshot'] for x in records]],r,'compile.log');assert 'Errors: 0, Warnings: 0' in body
scratch=r/'temp';scratch.mkdir();env=os.environ.copy();env.update(TEMP=str(scratch),TMP=str(scratch),TMPDIR=str(scratch))
results=[]
for seed in (71,391,1027):
 t=time.monotonic();log=r/f'seed{seed}.log'
 with log.open('w',encoding='utf-8') as f:
  proc=subprocess.run(list(map(str,[MS/'vsim.exe','-c','work.tb_resident_header_table',f'-gSEED={seed}','-do','onerror {quit -code 1}; run -all; quit -code 0'])),cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=90)
 body=log.read_text();assert proc.returncode==0 and not re.search(r'(?m)^#?\s*(FAIL|\*\* (Error|Fatal))',body),body[-2500:]
 lines=re.findall(r'^# (PASS RESIDENT_HEADER .*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
 results.append(dict(seed=seed,line=lines[0],seconds=time.monotonic()-t,log_sha256=sha(log)))
for x in records:assert sha(x['source'])==sha(x['snapshot'])==x['sha256']
result=dict(status='RESIDENT_HEADER_OFFICIAL_RAM_PASS',results=results,descriptors=8789,layouts=len(layouts),inputs_sha256=sha(r/'inputs.json'),full_cnn=False)
(r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
