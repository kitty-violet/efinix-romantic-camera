"""Actual A words, all SRAM phases, and fail-closed bad bit tests on vendor SRLs."""
import argparse,json,os,pickle,re,shutil,time
from pathlib import Path
from run_units import ROOT,EF,sha
from map_video_service_safe import run
from camera_joint_native import OUT

def main():
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--board',type=Path);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 n=a.native.resolve();o=a.root.resolve();assert not o.exists() and o.is_relative_to(ROOT/'tmp')
 r=json.loads((n/'weight-zero-edit.json').read_text());inv=json.loads((n/'a-zero-inventory.json').read_text())
 proof=json.loads((n/'weight-zero-independent-audit.json').read_text());assert proof['status']=='GUARDED_A_ZERO_EXACT_SOURCE_GRAPH_PASS_NOT_PHYSICAL'
 for f,h in proof['inputs'].items():assert sha(f)==h
 parent=ROOT/'tmp/frame-execution/connected-board-video-service-probe33-data-reset-only'
 key,old,_,op,_=pickle.loads((parent/'retained-audit-parse.pkl').read_bytes());assert key[0]==r['source_sha256']==sha(parent/'native_source.v')
 if a.board:
  board=a.board.resolve();mpf=json.loads((board/'retained-map-audit.json').read_text());assert mpf['status']=='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS'
  nk,_,new,_,np=pickle.loads((board/'retained-audit-parse.pkl').read_bytes());assert nk[0]==sha(n/'native_source.v') and nk[1]==sha(board/'outflow/ti60f225_oob.map.v')
 else:
  nk,new,np=pickle.loads((n/'zero-source-parse.pkl').read_bytes());assert nk==sha(n/'native_source.v');board=None
 pre='video_processing_pipeline/pipeline/cnn/core/dut/engine/mac/'
 storage=sorted(name for name,(k,pins) in old.items() if k=='EFX_SRL8' and name.startswith(pre+'cache/TERM['));assert len(storage)==528
 err=r['error_cell'];removed=set(r['removed']);selected=[set(storage),set(storage)-removed|set(r['new_graph'])|{err}]
 # Include the actual original term-CE decoder cone in both sides.
 drv={tok:name for name,(kind,pins) in old.items() for port in OUT[kind] for tok in pins.get(port,[]) if tok[0]=='net'}
 grant=tuple(r['grant_token']);selectors=list(map(tuple,r['selector_tokens']));fault=old[err][1]['Q'][0]
 boundary={grant,fault,*selectors};ce_cells=set()
 def walk(tok):
  if tok[0]!='net' or tok in boundary:return
  name=drv[tok]
  if name in ce_cells:return
  kind,pins=old[name];assert kind=='EFX_LUT4';ce_cells.add(name)
  for pin,ts in pins.items():
   if pin!='O':
    for t in ts:walk(t)
 for tok in r['term_ce'].values():walk(tuple(tok))
 selected=[x|ce_cells for x in selected]
 lines=['module tb;reg clk=0;always #5 clk=~clk;reg[63:0] data=0;reg[8:0] ce=0;reg[26:0] address=0;reg rst=1;reg original_error=0;',
  'reg[3:0] selected_term;integer ct;always @* begin selected_term=15;for(ct=0;ct<9;ct=ct+1)if(ce[ct])selected_term=ct;end wire[527:0] old_out,new_out;reg[527:0] old_capture=0,new_capture=0;wire cache_error;',
  'integer i,j,t,b,round,cycle=0,checks=0,guard_checks=0,valid_words=0;reg[63:0] rng=64\'hc13a278ecba97761;reg[63:0] masks[0:8];reg[63:0] words[0:1403];reg[3:0] terms[0:1403];']
 words=[w for gr in inv['groups'] for w in gr['words']];assert len(words)==1404
 cuts={('net','process_clk_150',None):'clk',old[err][1]['D'][0]:'original_error',old[err][1]['SR'][0]:'rst'}
 cuts[grant]='|ce';cuts[fault]='cache_error'
 for i,tok in enumerate(selectors):cuts[tok]=f'selected_term[{i}]'
 for i in range(64):cuts[('net',pre+f'cache/inverted_load[{i}]',None)]=f'~data[{i}]'
 for name in storage:
  t=int(re.search(r'TERM\[(\d+)\]',name)[1]);pins=old[name][1]
  for token,value in ([ (pins['CE'][0],f'ce[{t}]') ] if t==8 else [])+[(tok,f'address[{t*3+b}]') for b,tok in enumerate(pins['A'])]:
   assert token not in cuts or cuts[token]==value;cuts[token]=value
 for side,(g,pars,sel) in enumerate(zip([old,new],[op,np],selected)):
  tokens=sorted({tok for name in sel for ts in g[name][1].values() for tok in ts if tok[0]=='net'});ix={tok:i for i,tok in enumerate(tokens)}
  lines.append(f'wire[{len(tokens)-1}:0] w{side};')
  def scalar(tok):return f'w{side}[{ix[tok]}]' if tok[0]=='net' else tok[1]
  def value(ts):return scalar(ts[0]) if len(ts)==1 else '{'+','.join(map(scalar,ts))+'}'
  driven={tok for name in sel for port,ts in g[name][1].items() if port in ('Q','O','Q7') for tok in ts}
  external=set(tokens)-driven
  # All primitive-cut inputs have explicit identical stimulus bindings.
  assert external<=set(cuts),(side,external-set(cuts))
  for tok in sorted(external):lines.append(f'assign {scalar(tok)}={cuts[tok]};')
  for idx,name in enumerate(sorted(sel)):
   kind,pins=g[name];pins=dict(pins)
   if kind=='EFX_LUT4':
    for pin in ('I0','I1','I2','I3'):pins.setdefault(pin,[('literal',"1'b0",None)])
   args=','.join('.'+k+'('+str(w)+"'d"+str(v)+')' for k,(w,v) in pars[name].items())
   ports=','.join('.'+k+'('+value(v)+')' for k,v in pins.items())
   if kind=='EFX_SRL8' and 'Q7' not in pins:ports+=',.Q7()'
   lines.append(f'{kind} #({args}) cell_{side}_{idx}({ports});')
  for i,name in enumerate(storage):
   val="1'b0" if side==1 and name in removed else value(g[name][1]['Q'])
   lines.append(f'assign {"old" if side==0 else "new"}_out[{i}]={val};')
  if side:lines.append(f'assign cache_error={value(g[err][1]["Q"])};')
 lines+=['always @(posedge clk)begin old_capture<=old_out;new_capture<=new_out;end',
 'task tick;begin @(posedge clk);#2;cycle=cycle+1;@(negedge clk);address=address+27\'h1249249;end endtask',
 'task compare;begin if(old_out!==new_out || old_capture!==new_capture)$fatal(1,"zero cache mismatch %0d",cycle);if(cache_error!==0)$fatal(1,"valid load rejected %0d",cycle);checks=checks+1056;end endtask',
 'initial begin']
 for t in range(9):lines.append(f"masks[{t}]=64'h{int(inv['guard_masks'].get(str(t),0)):016x};")
 for i,w in enumerate(words):lines.append(f"words[{i}]=64'h{w['data']:016x};terms[{i}]={w['term']};")
 lines+=['tick;rst=0;tick;',
 'for(round=0;round<8;round=round+1)begin for(i=0;i<1404;i=i+1)begin data=words[i];ce=9\'b1<<terms[i];tick;compare;valid_words=valid_words+1;if(i%7==0)begin ce=0;data=~data;tick;compare;end end end',
 'for(i=0;i<8192;i=i+1)begin rng={rng[62:0],rng[63]^rng[62]^rng[60]^rng[59]};t=i%9;data=rng&~masks[t];ce=9\'b1<<t;tick;compare;end',
 # After injecting invalid coefficients, old general cache is intentionally
 # different. Only guarded error/recovery is checked; never compare invalid data.
 'for(t=0;t<8;t=t+1)begin for(b=0;b<64;b=b+1)begin rst=1;ce=0;tick;rst=0;data=64\'b1<<b;tick;if(cache_error!==0)$fatal(1,"unaccepted data rejected");guard_checks=guard_checks+1;ce=9\'b1<<t;tick;if(cache_error!==masks[t][b])$fatal(1,"bad-bit detect t=%0d b=%0d",t,b);guard_checks=guard_checks+1;ce=0;data=0;tick;if(cache_error!==masks[t][b])$fatal(1,"error not sticky");guard_checks=guard_checks+1;end end',
 'rst=1;tick;rst=0;ce=0;original_error=1;tick;if(cache_error!==1)$fatal(1,"original error lost");original_error=0;tick;if(cache_error!==1)$fatal(1,"original error not sticky");rst=1;tick;if(cache_error!==0)$fatal(1,"reset did not clear");guard_checks=guard_checks+3;',
 '$display("PASS ACTUAL_A_ZERO_CACHE cycles=%0d storage=528 valid_words=%0d comparisons=%0d guard_checks=%0d",cycle,valid_words,checks,guard_checks);$finish;end endmodule']
 o.mkdir();start=time.monotonic();(o/'tb.sv').write_text('\n'.join(lines))
 (o/'cuts.json').write_text(json.dumps(dict(cuts=[(k,v) for k,v in cuts.items()],old_cells=sorted(selected[0]),new_cells=sorted(selected[1])),indent=2))
 for f in ('efx_ff.v','efx_lut4.v','efx_srl8.v'):shutil.copy2(EF/'sim_models/verilog'/f,o/f)
 paths=[Path(__file__).resolve(),n/'native_source.v',n/'weight-zero-edit.json',n/'a-zero-inventory.json',n/'weight-zero-independent-audit.json',parent/'native_source.v',*o.iterdir()]
 if board:paths += [board/'retained-map-audit.json',board/'outflow/ti60f225_oob.map.v']
 inputs={str(f):sha(f) for f in paths};(o/'inputs.json').write_text(json.dumps(inputs,indent=2));result=dict(status='RUNNING',hardware_access=False,board_release=False)
 try:
  ms=Path('E:/quatra/questa_fse/win64');os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
  run([ms/'vlib.exe','work'],o,'vlib.log',30)
  log,_=run([ms/'vlog.exe','-sv','efx_ff.v','efx_lut4.v','efx_srl8.v','tb.sv'],o,'compile.log',60);assert 'Errors: 0, Warnings: 0' in log
  log,_=run([ms/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],o,'simulation.log',180)
  assert 'PASS ACTUAL_A_ZERO_CACHE' in log and 'Errors: 0, Warnings: 0' in log
  for f,h in inputs.items():assert sha(f)==h
  result.update(status='ACTUAL_A_ZERO_CACHE_GUARD_PASS',mapped=bool(board),source_sha256=sha(n/'native_source.v'),map_sha256=nk[1] if board else None,storage=528,valid_words=11232,ce_decoder_cells=len(ce_cells),scope='Actual primitive cache plus original term-CE decoder cut: all A words, address phases, backpressure, allowed arbitrary data, rejected bit/sticky/reset checks. Controls known Boolean. No whole CNN or hardware claim.',lines=[l for l in log.splitlines() if 'PASS ACTUAL' in l])
 except Exception as e:result.update(status='FAIL',error=repr(e));raise
 finally:
  result.update(seconds=time.monotonic()-start,inputs_sha256=sha(o/'inputs.json'));(o/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
