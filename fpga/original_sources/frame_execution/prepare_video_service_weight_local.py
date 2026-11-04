"""Split one measured long weight-byte broadcast into two identical stores.

Eight EFX_SRL8 copies keep the original clock, enable, serial load and read
address. The lower DSP row pair consumes the copies; no cycle is inserted.
"""
import argparse,json,pickle,re,shutil
from pathlib import Path
from run_units import ROOT,sha
from native_local_drivers import CELL,render_token,graph

def main():
 p=argparse.ArgumentParser()
 for name in ('base','board','root'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();base=a.base.resolve();board=a.board.resolve();out=a.root.resolve()
 assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 pp=base/'native-graph-audit.json';proof=json.loads(pp.read_text())
 for f,h in proof['inputs'].items():assert sha(f)==h,f
 assert proof['data_reset_bits']==512
 key,g,mg,pa,pb=pickle.loads((board/'retained-audit-parse.pkl').read_bytes())
 text=(base/'native_source.v').read_text();assert key[0]==sha(base/'native_source.v')==sha(board/'native_source.v')
 pre='video_processing_pipeline/pipeline/cnn/core/dut/engine/mac/'
 old=[pre+f'cache/TERM[2].CHANNEL[3].BIT[{i}].storage' for i in range(8)]
 sinks=[pre+f'arithmetic/ROW[{r}].CHANNEL[3].TAP[2].mac' for r in (0,2)]
 names=set(old+sinks);blocks={m[2].lstrip('\\'):m[0] for m in CELL.finditer(text) if m[2].lstrip('\\') in names}
 assert set(blocks)==names
 declaration='';insert='';copies=[];sub={}
 for i,n in enumerate(old):
  kind,pins=g[n];assert kind=='EFX_SRL8' and set(pins)=={'D','CLK','CE','Q','A'}
  assert pins['CLK']==[('net','process_clk_150',None)]
  new=f'SERVICE_LOCAL_WEIGHT_{i}';q=('net',f'service_local_weight_{i}',None);sub[pins['Q'][0]]=q
  np={**pins,'Q':[q]}
  def value(ts):return render_token(ts[0]) if len(ts)==1 else '{'+','.join(map(render_token,ts))+'}'
  cell='\n    EFX_SRL8 '+new+' ('+', '.join('.'+k+'('+value(v)+')' for k,v in np.items())+') /* identical byte branch */ ;\n'
  assert graph(cell)[new]==(kind,np)
  cell+=''.join(f"    defparam {new}.{k} = {w}'d{v};\n" for k,(w,v) in pa[n].items())
  insert+=cell;declaration+=f'    (* syn_keep="true" *) wire {q[1]};\n'
  copies.append(dict(original=n,copy=new,pins=np,parameters=pa[n]))
 edits=[];changed=text
 for n in sinks:
  kind,pins=g[n];assert kind.startswith('EFX_DSP')
  before=blocks[n];bp=[sub.get(t,t) for t in pins['B']];assert bp!=pins['B']
  assert {t for t in pins['B'] if t in sub}==set(sub)
  pattern=r'\.B\([^()]*\)';assert len(re.findall(pattern,before))==1
  after=re.sub(pattern,lambda m:'.B({'+','.join(map(render_token,bp))+'})',before)
  assert graph(after)[n]==(kind,{**pins,'B':bp})
  assert changed.count(before)==1;changed=changed.replace(before,after,1)
  edits.append(dict(cell=n,before=before,after=after))
 pos=CELL.search(changed).start();changed=changed[:pos]+declaration+changed[pos:]
 pos=changed.index('    protected_cell protected_inst (');changed=changed[:pos]+insert+changed[pos:]
 inverse=changed.replace(insert,'',1).replace(declaration,'',1)
 for e in edits:inverse=inverse.replace(e['after'],e['before'],1)
 assert inverse==text
 out.mkdir()
 for f in base.glob('*.sv'):shutil.copy2(f,out/f.name)
 (out/'native_source.v').write_text(changed,encoding='utf8')
 receipt=out/'weight-local-edit.json';receipt.write_text(json.dumps(dict(base=str(base),base_source_sha256=sha(base/'native_source.v'),copies=copies,edits=edits,insert=insert,declaration=declaration,
  proof='Each copied SRL8 has exactly the same initial state, serial input, enable, clock and read address as its original; same state transition and combinational read every cycle.',latency_added=0,ordinary_fixed_locations_added=0),indent=2))
 graft=json.loads((base/'graft.json').read_text());graft['weight_local']=dict(parent=str(base),receipt=str(receipt),sha256=sha(receipt))
 graft['inputs'][str(Path(__file__).resolve())]=sha(__file__);(out/'graft.json').write_text(json.dumps(graft,indent=2))
 inputs={**proof['inputs'],**{str(f):sha(f) for f in (pp,receipt,Path(__file__).resolve(),out/'native_source.v',out/'graft.json')}}
 result={**proof,'retained_primitives':proof['retained_primitives']+8,'weight_local_copies':8,'weight_local_parent':str(base),'weight_local_reversible_text':True,'inputs':inputs,'hardware_access':False,'board_release':False}
 (out/'native-graph-audit.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(dict(status=result['status'],retained=result['retained_primitives'],source_sha256=sha(out/'native_source.v'),copies=8,changed_dsp_ports=2,latency_added=0)))
if __name__=='__main__':main()
