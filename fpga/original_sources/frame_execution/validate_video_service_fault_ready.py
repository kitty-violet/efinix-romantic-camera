"""Actual source/mapped EFX LUT handshake cut, exhaustive known-control states."""
import argparse,json,os,pickle,shutil,time
from pathlib import Path
from run_units import ROOT,EF,sha
from map_video_service_safe import run
def main():
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--board',type=Path);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 n=a.native.resolve();o=a.root.resolve();assert not o.exists() and o.is_relative_to(ROOT/'tmp')
 proof=json.loads((n/'fault-ready-independent-audit.json').read_text());assert proof['status']=='FAULT_READY_SOURCE_INDEPENDENT_PASS'
 for f,h in proof['inputs'].items():assert sha(f)==h
 assert proof['source_sha256']==sha(n/'native_source.v')
 parent=ROOT/'tmp/frame-execution/video-service-native50-weight-zero-factored'
 oldsha,old,op=pickle.loads((parent/'zero-source-parse.pkl').read_bytes());assert oldsha==sha(parent/'native_source.v')==proof['parent_sha256']
 if a.board:
  board=a.board.resolve();ma=json.loads((board/'retained-map-audit.json').read_text());assert ma['status']=='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS'
  key,_,new,_,np=pickle.loads((board/'retained-audit-parse.pkl').read_bytes());assert key[0]==sha(n/'native_source.v') and key[1]==sha(board/'outflow/ti60f225_oob.map.v')
 else:
  key,new,np=pickle.loads((n/'zero-source-parse.pkl').read_bytes());assert key==sha(n/'native_source.v');board=None
 boundary=list(map(tuple,proof['boundary']));assert len(boundary)==20
 target=('net','video_processing_pipeline/pipeline/cnn/core/dut/engine/wsr',5)
 lines=['module tb;reg[19:0] stim=0;wire before_q,after_q;integer i;']
 for side,(g,pars,selected) in enumerate(((old,op,proof['old_cells']),(new,np,proof['new_cells']))):
  tokens=sorted({t for name in selected for vs in g[name][1].values() for t in vs if t[0]=='net'},key=str);ix={t:i for i,t in enumerate(tokens)}
  lines.append(f'wire[{len(tokens)-1}:0] w{side};')
  def value(t):return f'w{side}[{ix[t]}]' if t[0]=='net' else t[1]
  driven={g[name][1]['O'][0] for name in selected};external=set(tokens)-driven
  assert external<=set(boundary),(side,external-set(boundary))
  for t in external:lines.append(f'assign {value(t)}=stim[{boundary.index(t)}];')
  for i,name in enumerate(selected):
   k,pins=g[name];assert k=='EFX_LUT4';ports=dict(pins)
   for pin in ('I0','I1','I2','I3'):ports.setdefault(pin,[('literal',"1'b0",None)])
   lines.append(f'EFX_LUT4 #(.LUTMASK(16\'d{pars[name]["LUTMASK"][1]})) cell_{side}_{i}('+','.join('.'+pin+'('+value(ts[0])+')' for pin,ts in ports.items())+');')
  lines.append(f'assign {"before_q" if side==0 else "after_q"}={value(target)};')
 lines+=['initial begin for(i=0;i<1048576;i=i+1)begin stim=i;#2;if(before_q!==after_q || (after_q!==0 && after_q!==1))$fatal(1,"ready differs i=%0d old=%b new=%b",i,before_q,after_q);end',
 '$display("PASS FAULT_READY_EXHAUSTIVE comparisons=1048576 known_controls=20 added_cycles=0");$finish;end endmodule']
 o.mkdir();start=time.monotonic();(o/'tb.sv').write_text('\n'.join(lines))
 shutil.copy2(EF/'sim_models/verilog/efx_lut4.v',o/'efx_lut4.v')
 paths=[Path(__file__).resolve(),n/'native_source.v',n/'fault-ready-independent-audit.json',parent/'native_source.v',o/'tb.sv',o/'efx_lut4.v']
 if board:paths +=[board/'retained-map-audit.json',board/'outflow/ti60f225_oob.map.v']
 inputs={str(f):sha(f) for f in paths};(o/'inputs.json').write_text(json.dumps(inputs,indent=2));result=dict(status='RUNNING',hardware_access=False,board_release=False)
 try:
  ms=Path('E:/quatra/questa_fse/win64');os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
  run([ms/'vlib.exe','work'],o,'vlib.log',30)
  log,_=run([ms/'vlog.exe','-sv','efx_lut4.v','tb.sv'],o,'compile.log',60);assert 'Errors: 0, Warnings: 0' in log
  log,_=run([ms/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],o,'simulation.log',180)
  assert 'PASS FAULT_READY_EXHAUSTIVE' in log and 'Errors: 0, Warnings: 0' in log
  for f,h in inputs.items():assert sha(f)==h
  result.update(status='ACTUAL_FAULT_READY_EXHAUSTIVE_PASS',mapped=bool(board),source_sha256=sha(n/'native_source.v'),map_sha256=key[1] if board else None,comparisons=1048576,known_control_inputs=20,added_cycles=0,scope='Actual old versus new EFX primitive handshake cut; all Boolean control combinations; no whole-board or arbitrary-X claim')
 except Exception as e:result.update(status='FAIL',error=repr(e));raise
 finally:
  result.update(seconds=time.monotonic()-start,inputs_sha256=sha(o/'inputs.json'));(o/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
