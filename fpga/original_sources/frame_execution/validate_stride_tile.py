"""Real resident + arena + MAC/RQ + write ACK; synthetic original-tensor oracle."""
import argparse,json,re,shutil,sys,traceback
from pathlib import Path
from run_units import HERE,ROOT,EF,MS,sha,run
from validate_stride_tap_scatter import vectors as taps
from validate_mixed_pool import selected as pool_sources
from cluster_stride_vectors import generate,parameters,build_layout,requantize
from stride_tap_layout import source_coordinate
def selected():
 return pool_sources(True)+[HERE/n for n in ('resident_weight_bank64.sv','tensor_arena12.sv','phase_resident_reader.sv','next_header_prefetch.sv','resident_read_mux2.sv','column_write4.sv','stride_tile_cluster.sv')]+[HERE/'variants/stride_writeback/stride_tap_feed24.sv',HERE.parent/'full_frame_closure/coefficient_bank_store_fast.sv']
def vectors(root):
 root.mkdir();oracle=taps(root/'taps');generate(root/'coeff','dense')
 params=parameters('dense');ops={x['name']:x for x in build_layout()['operators']};words=[];offsets=[]
 def signed(v):return (v&255)-256 if v&128 else v&255
 for ci,(sw,sh,ox,oy,w,g,base) in enumerate(oracle['shapes']):
  name='down1.dw' if g==1 else 'down2.dw';op=ops[name];par=params[name]
  offsets.append([op[k]['offset'] for k in ('weights','bias','multiplier','relu')]+[op['index']])
  for r in range(3):
   for x in range(w):
    for group in range(g):
     out=0
     for lane in range(8):
      oc=group*8+lane;dot=0
      for tap in range(9):
       sx,sy=source_coordinate(sw,sh,ox+x,oy+r,tap)
       dot+=signed(sx*37+sy*71+group*19+lane*23+ci*11)*par.weights[oc][tap]
      q,overflow=requantize(dot,par.bias[oc],par.multiplier[oc],relu=par.relu);assert not overflow
      out|=(q&255)<<(8*lane)
     words.append(out)
 (root/'result.hex').write_text(''.join(f'{x:016x}\n' for x in words))
 for i,n in enumerate(('wo','bo','mo','ro','lid')):(root/(n+'.hex')).write_text(''.join(f'{v[i]:08x}\n' for v in offsets))
 return oracle
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,default=71);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 try:
  oracle=vectors(root/'vectors');records=[]
  paths=[EF/'sim_models/verilog'/n for n in ('efx_dsp24.v','efx_ram10.v','efx_dpram10.v')]+selected()+[HERE/'tb_stride_tile_cluster.sv']
  for f in paths:
   dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  dependencies=[Path(__file__),HERE/'stride_tap_layout.py',HERE/'validate_stride_tap_scatter.py',HERE.parent/'pw_array_feed/feed_reference.py',ROOT/'network_ab/coefficient_layout.py',ROOT/'network_ab/contract_reference.py',ROOT/'network_ab/model_b_contract.json']
  (root/'inputs.json').write_text(json.dumps(dict(inputs=records,oracle=oracle,dependencies={str(f):sha(f) for f in dependencies},vectors={str(f.relative_to(root/'vectors')):sha(f) for f in (root/'vectors').rglob('*') if f.is_file()}),indent=2))
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
  checker=HERE.parent/'coefficient_store/validate_readback.py';dump=root/'readback.hex'
  do='onerror {quit -code 1}\nrun 200 us\nif {[examine -radix unsigned sim:/tb_stride_tile_cluster/verify_checkpoint] != 1} {error "missing real readback"}\n'
  do+='puts [exec {'+Path(sys.executable).as_posix()+'} -B {'+checker.as_posix()+'} --image-dir {'+(root/'vectors/coeff').as_posix()+'} --readback {'+dump.as_posix()+'}]\n'
  do+='force -deposit sim:/tb_stride_tile_cluster/software_verified 1\nrun -all\nquit -code 0\n';(root/'run.do').write_text(do)
  body,secs=run([MS/'vsim.exe','-c','work.tb_stride_tile_cluster','-gRANDOM_SEED='+str(a.seed),'+VECTORS='+(root/'vectors').as_posix(),'+DUMP='+dump.as_posix(),'-do','do run.do'],root,'simulation.log')
  lines=re.findall(r'^# (PASS STRIDE_TILE_CLUSTER.*)$',body,re.M);assert len(lines)==1
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='LOCAL_REAL_RESIDENT_STRIDE_WRITE_ACK_PASS',seed=a.seed,line=lines[0],rows=re.findall(r'^# STRIDE_TILE (.*)$',body,re.M),seconds=secs,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),weights_source='SYNTHETIC_DIAGNOSTIC_REAL_STORE',producer_connected=False,result_writer_connected=True,last_write_ack_in_timing=True,full_frame=False,measured_frame_ms=None,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
