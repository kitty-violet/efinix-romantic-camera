"""Four-state issuer test: all original program fields and two dependency streams."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import HERE,ROOT,MS,sha,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--vectors',type=Path,required=True);p.add_argument('--descriptors',type=Path,required=True);p.add_argument('--issuer',type=Path);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
records=[]
for n in ('frame_program_issue.sv','tb_frame_program_issue.sv'):
 src=(a.issuer.resolve() if a.issuer and n=='frame_program_issue.sv' else HERE/'variants/row_pair_arena'/n)
 assert src.is_relative_to(HERE/'variants') and src.name==n
 dst=root/n;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
manifest=json.loads((a.vectors/'frame_manifest.json').read_text())
for n in ('program.hex','dma_commands.hex'):assert sha(a.vectors/n)==manifest['files'][n]
descriptor_meta=json.loads((a.descriptors/'result.json').read_text())
descriptor_records=[]
for name,width,count in [('compute',64,8789),('dma',16,10440)]:
 src=a.descriptors/(name+'.bin');data=src.read_bytes()
 assert sha(src)==descriptor_meta['files'][name+'.bin'] and len(data)==width*count
 dst=root/(name+'.hex')
 dst.write_text(''.join(format(int.from_bytes(data[i:i+width],'little'),'0'+str(width*2)+'x')+'\n' for i in range(0,len(data),width)))
 descriptor_records.append(dict(source=str(src.resolve()),sha256=sha(src),hex_snapshot=str(dst),hex_sha256=sha(dst)))
# Independent expected fields: preserve the original 32-field scoreboard;
# only replace physical placement using the separately proven row layout.
program=Path(descriptor_meta['program_path'])
pr=json.loads((program/'result.json').read_text())
assert sha(program/'program.json')==pr['program_sha256']==descriptor_meta['program_sha256']
rows=json.loads((program/'program.json').read_text())
words=[int(x,16) for x in (a.vectors/'program.hex').read_text().splitlines()]
expected=[];phases=[]
for c,word in zip(rows,words):
 f=[(word>>(32*i))&0xffffffff for i in range(32)];phase=0
 def pk(ds,k,n):return sum(x[k]<<(n*i) for i,x in enumerate(ds))
 if c['kind']=='compute':
  for key,bf,b0,b1 in [('inputs',10,11,12),('outputs',13,14,15),('skip',20,21,22)]:
   f[bf]=pk(c[key],'bank',3);b=pk(c[key],'base',10);f[b0]=b&0xffffffff;f[b1]=b>>32
  phase=c['inputs'][0]['phase']|(pk(c['outputs'],'phase',1)<<1)
 else:f[4]=c['row']['bank'];f[5]=c['row']['base']
 expected.append(sum(x<<(32*i) for i,x in enumerate(f)));phases.append(phase)
(root/'expected_program.hex').write_text(''.join(f'{x:0256x}\n' for x in expected))
(root/'expected_phase.hex').write_text(''.join(f'{x:02x}\n' for x in phases))
assert len((root/'expected_program.hex').read_text().splitlines())==19229
assert len((root/'expected_phase.hex').read_text().splitlines())==19229

(root/'inputs.json').write_text(json.dumps(dict(inputs=records,vectors=str(a.vectors.resolve()),vector_manifest_sha256=sha(a.vectors/'frame_manifest.json'),descriptors=str(a.descriptors.resolve()),descriptor_records=descriptor_records),indent=2))
run([MS/'vlib.exe','work'],root,'vlib.log')
body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
body,secs=run([MS/'vsim.exe','-c','work.tb_frame_program_issue','+VECTORS='+a.vectors.resolve().as_posix(),'+DESCRIPTORS='+root.as_posix(),'-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=180)
lines=re.findall(r'^# (PASS (?:ISSUE_NORMAL|PROGRAM_ISSUE).*)$',body,re.M)
assert len(lines)==3 and 'Errors: 0, Warnings: 0' in body
for r in records:assert sha(r['source'])==sha(r['snapshot'])==r['sha256']
result=dict(status='ROW_PAIR_PROGRAM_ISSUE_FOURSTATE_PASS',lines=lines,seconds=secs,inputs_sha256=sha(root/'inputs.json'),simulation_sha256=sha(root/'simulation.log'),compute_engines='protocol models with independent original descriptor scoreboard',full_system=False)
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
