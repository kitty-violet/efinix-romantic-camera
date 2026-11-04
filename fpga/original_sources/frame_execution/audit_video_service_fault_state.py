"""Independent full-text inverse and actual-map state-copy induction check."""
import argparse,json,pickle
from pathlib import Path
from run_units import ROOT,sha
from continuous_control_native import connections

def main():
 p=argparse.ArgumentParser()
 for n in ('board','native','out'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();b=a.board.resolve();n=a.native.resolve();out=a.out.resolve()
 assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 receipt=n/'fault-state-edit.json';r=json.loads(receipt.read_text());base=Path(r['base'])
 assert sha(base/'native_source.v')==r['base_sha256']
 text=(n/'native_source.v').read_text()
 assert text.count(r['insert'])==text.count(r['declaration'])==1
 inverse=text.replace(r['insert'],'',1).replace(r['declaration'],'',1)
 assert len(r['records'])==2
 for row in r['records']:
  assert inverse.count(row['after'])==1
  inverse=inverse.replace(row['after'],row['before'],1)
 assert inverse==(base/'native_source.v').read_text()
 for f in base.glob('*.sv'):assert sha(f)==sha(n/f.name)
 audit=json.loads((b/'retained-map-audit.json').read_text())
 assert audit['source_sha256']==sha(n/'native_source.v')==sha(b/'native_source.v')
 key,source,g,sp,mp=pickle.loads((b/'retained-audit-parse.pkl').read_bytes())
 assert key[1]==sha(b/'outflow/ti60f225_oob.map.v')==audit['map_sha256']
 _,loads=connections(g);checks=[]
 for row in r['records']:
  old,new=row['original'],row['copy']
  assert source[old][0]==source[new][0]==g[old][0]==g[new][0]=='EFX_FF'
  for graph in (source,g):
   assert {k:v for k,v in graph[old][1].items() if k!='Q'}=={k:v for k,v in graph[new][1].items() if k!='Q'}
  assert sp[old]==sp[new]==mp[old]==mp[new]
  sinks=sorted(loads[g[new][1]['Q'][0]])
  assert sinks==[(row['load'],row['pin'])]
  checks.append(dict(original=old,copy=new,pins=g[new][1],parameters=mp[new],loads=sinks))
 files=[Path(__file__).resolve(),receipt,base/'native_source.v',n/'native_source.v',b/'native_source.v',b/'outflow/ti60f225_oob.map.v',b/'retained-map-audit.json']
 files += list(base.glob('*.sv'))+list(n.glob('*.sv'))
 result=dict(status='ACTUAL_MAPPED_FAULT_STATE_EQUIVALENT_COPIES_PASS',map_sha256=audit['map_sha256'],
  copies=checks,full_text_inverse=True,all_sv_equal=True,latency_added=0,
  proof='Same vendor FF type, default/explicit initialization and all parameters; exactly equal complete D/CE/CLK/SR signals. Equal initial states plus identical next-state function imply equality every edge, including four-state inputs. No timing equivalence claim.',
  evidence={str(f):sha(f) for f in files},hardware_access=False,board_release=False)
 out.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('status','map_sha256','latency_added')}))
if __name__=='__main__':main()
