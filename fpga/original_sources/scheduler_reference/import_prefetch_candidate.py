"""Verify handed-off program-only candidate, importing only into this workflow."""
from pathlib import Path
import importlib.util,json,hashlib,shutil,sys,zlib
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parent;A=R.parents[1]/'dma-prefetch-audit'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text('utf8'))
def main():
 expected='6d3570453d9b6bf414f2e059ae9d30e3e28e4f46965245fb1454dc188f612020'
 program=A/'allocation-candidate/program.json';assert sha(program)==expected
 snapshot=read(A/'snapshot.json');diffs={}
 for ver,allowed in [(2,{'cnn_frame_datapath.sv','row_pair_compute_cluster.sv'}),(3,{'cnn_frame_datapath.sv','row_pair_compute_cluster.sv','fixed_weight_loader.sv','fixed_ring_weights9.sv','fixed_production_mac.sv'})]:
  different=[]
  for name,h in snapshot['files'].items():
   p=Path(name)
   if p.parts[0]!='rtl':continue
   assert sha(A/p)==h,(p,'handoff source changed')
   target=R/f'experiments/control-boundary{ver}'/p.name
   if sha(target)!=h:different.append(p.name)
  assert set(different)==allowed,(ver,different);diffs[ver]=different
 base=R/'data/descriptors4';oldhash=['ec56be4d5511cd54d97a10f5c7574a163737582f0ba2bc904773ffac3ebcd0ce','519c04cdd5ca699d389a6d71e95087e3dd6da17764735146f5a5799c39744471']
 assert all(sha(base/f'slot{i}/compute.bin')==oldhash[0]for i in range(5))
 assert sha(base/'dma.bin')==oldhash[1]
 assert sha(A/'schedule/program.json')==sha(R/'data/schedule4/program.json')
 v=read(A/'candidate-vectors/result.json')
 for n,h in v['files'].items():assert sha(A/'candidate-vectors'/n)==h['sha256']
 for name in ['trace1','candidate-trace1']:
  f=read(A/name/'result.json');assert f['status']=='FULL_COMPUTE_FRAME_PASS_NOT_BOARD_RELEASE'
  assert all(sha(Path(p))==h for p,h in f['inputs'].items())
 # Repeat actual descriptor repacking and peer-dependency replays locally.
 sys.path.insert(0,str(A));spec=importlib.util.spec_from_file_location('prefetch_export',A/'export_candidate.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 units=mod.accesses.__globals__['units'];peak=[0]*6
 def bounded_units(d,start,count):
  assert 0<=d['bank']<6 and 0<=d['base']<1024
  us=units(d,start,count)
  for u,_ in us:
   assert d['bank']*2048<=u<(d['bank']+1)*2048,(d,u)
   peak[d['bank']]=max(peak[d['bank']],(u-d['bank']*2048)//2+1)
  return us
 mod.accesses.__globals__['units']=bounded_units
 cmds=read(program);reference=read(R/'data/schedule4/program.json')
 for c in cmds:
  if c['kind']=='load'and c['row']['groups']==4:assert c['row']['base']+c['row']['length']<=1024
 out,proof=mod.emit(cmds,reference,[(base/'slot3/compute.bin').read_bytes(),(base/'dma.bin').read_bytes()])
 assert proof['counts']==[6546,8277]and max(peak)==1024
 for b,n in zip(out,['compute.bin','dma.bin']):assert b==(A/'candidate-vectors'/n).read_bytes()
 # Static access identities are rechecked independently of the abstract timing model.
 owner=[None]*12288;checked=0
 for i,c in enumerate(cmds):
  rr,ww,pad=mod.accesses(c)
  for u,identity in rr:
   assert owner[u]==(c['tile'],*identity),(i,u,owner[u],identity);checked+=1
  for u in pad:owner[u]=('padding',i)
  for u,identity in ww:owner[u]=(c['tile'],*identity)
 assert checked==7040479
 dest=R/'data/descriptors5';dest.mkdir(exist_ok=False)
 for i in range(5):
  (dest/f'slot{i}').mkdir();(dest/f'slot{i}/compute.bin').write_bytes(out[0])
 (dest/'dma.bin').write_bytes(out[1]);shutil.copy2(base/'schema.json',dest/'schema.json')
 d=read(base/'result.json');d.update(schedule_sha256=expected,replays=proof['replays'],source_handoff=str(A),source_handoff_sha256=sha(A/'snapshot.json'),SRAM_high_words=peak,static_read_identity_checks=checked,unchanged_fields_checked=proof['unchanged_fields_checked'],rtl_decoder_verified=False)
 d.pop('seconds',None)
 for item in d['styles']:item.update(compute_sha256=sha(dest/f'slot{item["slot"]}/compute.bin'),compute_crc32=f'{zlib.crc32(out[0]):08x}')
 d.update(dma_sha256=sha(dest/'dma.bin'),dma_crc32=f'{zlib.crc32(out[1]):08x}')
 (dest/'result.json').write_text(json.dumps(d,indent=2),'utf8')
 vec=R/'vectors/frame-prefetch1';vec.mkdir(exist_ok=False)
 oldvec=R/'vectors/frame-dma-reordered4'
 for n,h in v['files'].items():
  if n not in ('compute.bin','dma.bin','crc.txt'):assert sha(A/'candidate-vectors'/n)==sha(oldvec/n),(n,'Golden or PIXEL coefficients changed')
  shutil.copy2(A/'candidate-vectors'/n,vec/n)
 (vec/'result.json').write_text(json.dumps(dict(v,source_handoff=str(A),source_receipt_sha256=sha(A/'candidate-vectors/result.json')),indent=2),'utf8')
 report=dict(status='PREFETCH_PROGRAM_IMPORTED_STATIC_PASS_NEEDS_CURRENT_RTL',source=str(A),program_sha256=expected,rtl_differences_from_handoff=diffs,bank_high_words=peak,semantic_reads=checked,descriptor_proof=proof,only_two_programs_imported=True,weights_replaced=False,hardware_access=False)
 (R/'evidence/prefetch-import.json').write_text(json.dumps(report,indent=2),'utf8');print(report['status'],peak,checked)
if __name__=='__main__':main()
