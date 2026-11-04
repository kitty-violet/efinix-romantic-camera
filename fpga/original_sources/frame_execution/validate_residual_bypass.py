"""Freeze sources; golden numeric/credit/flush verification of actual RAM pool."""
import argparse,json,re,shutil,sys,traceback
from pathlib import Path
from run_units import HERE,ROOT,EF,MS,sha,run
def sources():
 fw=HERE.parent/'feature_window';selection=json.loads((fw/'shared_pool_selected_sources.json').read_text())
 paths=[]
 for row in selection['sources']:
  f=ROOT/row['path'];assert sha(f).upper()==row['sha256']
  if row['role'] in ('front','array'):continue
  paths.append(HERE.parent/'pw_array_feed_compact'/f.name if row['role']=='requant' else f)
 return paths+[HERE/n for n in ('banked_result_fifo.sv','shared_mac_stream_front.sv','shared_pool_banked_fifo.sv')]
def selected():
 return [HERE/'variants/residual_bypass'/f.name if f.name in ('shared_pool_banked_fifo.sv','shared_rq_oc8_cluster.sv') else f for f in sources()]
def direct_vectors(dest):
 sys.path.insert(0,str(ROOT/'network_ab'))
 from contract_reference import requantize
 mults=[0,1,65536,131071,-1,-65536,-131072,56234]
 values=[];expected=[];masks=[]
 for mi,m in enumerate(mults):
  for v in range(512):
   lanes=[((v+l*71)%512)-256 for l in range(8)]
   mask=255 if v%3 else (v*17)&255
   values.append(sum((x&511)<<(9*l) for l,x in enumerate(lanes)))
   expected.append(sum((requantize(x,0,m)[0]&255)<<(8*l) for l,x in enumerate(lanes) if mask>>l&1))
   masks.append(mask)
 for name,vals,width in [('value',values,18),('expected',expected,16),('mask',masks,2),('multiplier',[m&262143 for m in mults],5)]:
  (dest/('bypass_'+name+'.hex')).write_text(''.join(f'{x:0{width}x}\n' for x in vals))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seeds',nargs='+',type=int,default=[71]);p.add_argument('--kind',choices=['normal','bypass'],required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 fw=HERE.parent/'feature_window'
 top='tb_banked_pool' if a.kind=='normal' else 'tb_residual_bypass'
 paths=[EF/'sim_models/verilog'/n for n in ('efx_dsp24.v','efx_ram10.v')]+selected()+[HERE/'variants/residual_bypass'/(top+'.sv')]
 records=[]
 try:
  for f in paths+[fw/'shared_array_reference.py',Path(__file__),HERE/'run_units.py']:
   dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  run([sys.executable,'-B',fw/'shared_array_reference.py','--output-dir',root/'vectors'],root,'vectors.log')
  direct_vectors(root/'vectors')
  (root/'inputs.json').write_text(json.dumps(dict(inputs=records,fronts=32,clusters=4,credits_per_bank=16,vectors={f.name:sha(f) for f in (root/'vectors').iterdir()}),indent=2))
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[root/f.name for f in paths]],root,'compile.log')
  assert 'Errors: 0, Warnings: 0' in body
  rows=[]
  for seed in a.seeds:
   body,secs=run([MS/'vsim.exe','-c','work.'+top,'-gRANDOM_SEED='+str(seed),
    '+VECTORS='+(root/'vectors').as_posix(),'-do','onerror {quit -code 1}; run -all; quit -code 0'],root,f'seed{seed}.log')
   lines=re.findall(r'^# (PASS (?:BANKED_POOL|RESIDUAL_BYPASS).*)$',body,re.M)
   assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
   rows.append(dict(seed=seed,line=lines[0],seconds=secs,log_sha256=sha(root/f'seed{seed}.log')))
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='LOCAL_RESIDUAL_BYPASS_POOL_'+a.kind.upper()+'_PASS',results=rows,inputs_sha256=sha(root/'inputs.json'),full_frame=False,real_feed=False,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
