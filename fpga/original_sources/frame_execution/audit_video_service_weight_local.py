"""Independent source inverse and mapped SRL induction/byte-order check."""
import argparse,json,pickle
from pathlib import Path
from run_units import ROOT,sha
from continuous_control_native import connections

def main():
 p=argparse.ArgumentParser()
 for n in ('board','native','out'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();b=a.board.resolve();n=a.native.resolve();out=a.out.resolve()
 assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 receipt=n/'weight-local-edit.json';r=json.loads(receipt.read_text());base=Path(r['base'])
 assert sha(base/'native_source.v')==r['base_source_sha256']
 s=(n/'native_source.v').read_text();assert s.count(r['insert'])==s.count(r['declaration'])==1
 inverse=s.replace(r['insert'],'',1).replace(r['declaration'],'',1)
 assert len(r['copies'])==8 and len(r['edits'])==2
 for e in r['edits']:
  assert inverse.count(e['after'])==1;inverse=inverse.replace(e['after'],e['before'],1)
 assert inverse==(base/'native_source.v').read_text()
 for f in base.glob('*.sv'):assert sha(f)==sha(n/f.name)
 audit=json.loads((b/'retained-map-audit.json').read_text())
 assert audit['source_sha256']==sha(n/'native_source.v')==sha(b/'native_source.v')
 key,source,g,sp,mp=pickle.loads((b/'retained-audit-parse.pkl').read_bytes())
 assert key[1]==sha(b/'outflow/ti60f225_oob.map.v')==audit['map_sha256']
 _,loads=connections(g);sub={};rows=[]
 expected=sorted((e['cell'],'B') for e in r['edits'])
 for c in r['copies']:
  old,new=c['original'],c['copy']
  assert source[old][0]==source[new][0]==g[old][0]==g[new][0]=='EFX_SRL8'
  for graph in (source,g):
   assert {k:v for k,v in graph[old][1].items() if k!='Q'}=={k:v for k,v in graph[new][1].items() if k!='Q'}
  assert sp[old]==sp[new]==mp[old]==mp[new]
  assert sorted(loads[g[new][1]['Q'][0]])==expected
  sub[source[old][1]['Q'][0]]=source[new][1]['Q'][0]
  rows.append(dict(original=old,copy=new,ports=g[new][1],parameters=mp[new],loads=expected))
 # Confirm the complete mapped sign extension and byte order are retained.
 for e in r['edits']:
  cell=e['cell'];assert source[cell][0]==g[cell][0]
  assert source[cell][1]['B']==g[cell][1]['B']
  assert set(sub.values()).issubset(set(g[cell][1]['B']))
 paths=[Path(__file__).resolve(),receipt,base/'native_source.v',n/'native_source.v',b/'native_source.v',b/'outflow/ti60f225_oob.map.v',b/'retained-map-audit.json']
 paths += list(base.glob('*.sv'))+list(n.glob('*.sv'))
 d=dict(status='ACTUAL_MAPPED_WEIGHT_BYTE_SPLIT_EQUIVALENCE_PASS',map_sha256=audit['map_sha256'],copies=rows,
  copied_state_bits=64,full_text_inverse=True,all_sv_equal=True,latency_added=0,
  proof='Identical SRL8 initialization, parameters and complete D/CE/CLK/A. Equal state induction and identical combinational read imply every-cycle Q equality, including four-state inputs; full DSP B vector checked without bit/sign reordering.',
  evidence={str(f):sha(f) for f in paths},hardware_access=False,board_release=False)
 out.write_text(json.dumps(d,indent=2));print(json.dumps({k:d[k] for k in ('status','map_sha256','copied_state_bits','latency_added')}))
if __name__=='__main__':main()
