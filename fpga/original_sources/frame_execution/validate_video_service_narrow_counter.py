"""Actual-map phase/word/epoch next-state proof at accepted-snapshot boundary."""
import argparse,functools,json,pickle,re
from pathlib import Path
from run_units import ROOT,sha
from continuous_control_native import connections,net
from probe_full_native_split_map import params
from video_service_serializer_native import P

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();out=a.output.resolve();assert not out.exists() and out.is_relative_to(ROOT/'tmp')
 src=board/'outflow/ti60f225_oob.map.v';key,g=pickle.loads((board/'service-validation-graph.pkl').read_bytes());assert key[0]==sha(src)
 ps=params(src.read_text());d,_=connections(g)
 fields={'service_page_phase':3,'service_page_part':4,'service_page_sequence':8,'service_legacy_epoch':2}
 selected={n for n,(k,pins) in g.items() if k=='EFX_FF' and any(n.startswith(f+'[') for f in fields)}
 assert len(selected)==17
 for n in selected:
  p=g[n][1];assert p['CLK']==[net('process_clk_150')] and p['SR']==[('literal',"1'b0",None)]
  assert ps[n]['CLK_POLARITY']==(1,1) and ps[n]['SR_POLARITY']==(1,1)
 checks=0
 for phase in range(6):
  for part in range(16):
   for epoch in range(4):
    for sequence in (0,1,127,128,254,255):
     for request in (0,1):
      words=dict(zip(fields,(phase,part,sequence,epoch)))
      values={net(f,i):(v>>i)&1 for f,v in words.items() for i in range(fields[f])}
      values[net(P+'n5')]=request
      @functools.lru_cache(None)
      def ev(t):
       if t in values:return values[t]
       if t[0]=='literal':
        m=re.fullmatch(r"\d+'[sS]?([hbd])([0-9a-fA-F]+)",t[1]);assert m,t
        return int(m[2],{'h':16,'b':2,'d':10}[m[1]])
       n=d[t];k,p=g[n]
       if k=='EFX_LUT4':return ps[n]['LUTMASK'][1]>>sum(ev(p.get('I'+str(i),[('literal',"1'b0",None)])[0])<<i for i in range(4))&1
       assert k=='EFX_ADD',(n,k)
       total=sum(ev(p[z][0]) for z in ('I0','I1','CI'));return total&1 if p.get('O')==[t] else total>>1
      next_values=dict(values)
      for n in selected:
       p=g[n][1];par=ps[n]
       enable=ev(p['CE'][0])^int(par['CE_POLARITY']==(1,0))
       if enable:next_values[p['Q'][0]]=ev(p['D'][0])^int(par['D_POLARITY']==(1,0))
      observed={f:sum(next_values[net(f,i)]<<i for i in range(w)) for f,w in fields.items()}
      expected=words.copy()
      if request:
       expected['service_page_phase']=(phase+1)%6
       expected['service_page_sequence']=(sequence+1)%256
       if phase==5:
        expected['service_page_part']=(part+1)%16
        if part==15:expected['service_legacy_epoch']=(epoch+1)%4
      assert observed==expected,(words,request,observed,expected)
      checks+=1
 result=dict(status='ACTUAL_MAP_NARROW_SWEEP_COUNTER_PASS',cases=checks,registers=17,accepted_requests_per_epoch=96,
  all_phase_word_epoch_combinations=True,sequence_boundary_values=[0,1,127,128,254,255],
  snapshot_event_cut=P+'n5',handshake_has_separate_actual_map_test=True,map_sha256=sha(src),checker_sha256=sha(__file__),hardware_access=False,board_release=False)
 out.write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
