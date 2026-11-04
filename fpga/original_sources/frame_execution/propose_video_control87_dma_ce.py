"""Read-only exact fallback: remove one late-bank stage from RGB head CE.

This records an alternative, never modifies the active84 implementation.
"""
import itertools,json,pickle
from run_units import ROOT,sha

def main():
 b=ROOT/'tmp/frame-execution/connected-board-video-control84-map1'
 h,g,ps,dr,ld=pickle.loads((b/'integration-parse.pkl').read_bytes());assert h==sha(b/'outflow/ti60f225_oob.map.v')
 names=['LUT__146657','LUT__146659'];original=[]
 for n in names:
  k,p=g[n];assert k=='EFX_LUT4'
  original.append(dict(name=n,inputs=[p.get(f'I{i}',[None])[0] for i in range(4)],output=p['O'][0],mask=ps[n]['LUTMASK'][1]))
 assert [r['mask'] for r in original]==[244,16448]
 a,bv,c,d=original[0]['inputs'];e,f=original[1]['inputs'][1:3]
 assert original[1]['inputs'][0]==original[0]['output'] and original[1]['inputs'][3] is None
 boundaries=[a,bv,c,d,e,f];assert len(set(boundaries))==6
 pnet=('net','dma_ce87_precondition',None);qnet=('net','dma_ce87_qualified',None)
 def lut(inputs,fn,name,output):
  mask=sum(int(fn(*[(v>>i)&1 for i in range(4)]))<<v for v in range(16))
  return dict(name=name,inputs=inputs,output=output,mask=mask)
 candidate=[lut([d,c,bv,e],lambda D,C,B,E:E and (D or (not C and not B)),'DMA_CE87_PRECONDITION',pnet),
  lut([c,e,None,None],lambda C,E,u,v:E and not C,'DMA_CE87_QUALIFIED',qnet),
  lut([a,pnet,qnet,f],lambda A,P,Q,F:F and (P or (A and Q)),'LUT__146659',original[1]['output'])]
 def evaluate(rows,env):
  env=dict(env)
  for row in rows:
   ix=sum((env[t] if t else 0)<<i for i,t in enumerate(row['inputs']))
   env[row['output']]=(row['mask']>>ix)&1
  return env[original[1]['output']]
 for bits in itertools.product((0,1),repeat=6):
  env=dict(zip(boundaries,bits));assert evaluate(original,env)==evaluate(candidate,env),(bits,env)
 # Failed branch must never be hidden: tampered result is rejected by at least
 # one of the complete64 assignments.
 bad=[dict(r) for r in candidate];bad[-1]['mask']^=1<<15
 assert any(evaluate(original,dict(zip(boundaries,bits)))!=evaluate(bad,dict(zip(boundaries,bits))) for bits in itertools.product((0,1),repeat=6))
 result=dict(status='DMA_CE87_BINARY_PROPOSAL_64_PASS_NOT_IMPLEMENTED',map_sha256=h,boundary=boundaries,original=original,candidate=candidate,
  retained_original=['LUT__146657'],changed=['LUT__146659'],added=['DMA_CE87_PRECONDITION','DMA_CE87_QUALIFIED'],
  retained_n70854_consumers=ld[original[0]['output']],output_consumers=ld[original[1]['output']],
  equivalent_combinations=64,negative_mutation_rejected=True,source_primitive_delta=2,new_state=0,latency_added=0,
  late_control2486_depth_before=2,late_control2486_depth_after=1,
  new_tradeoff='Early !aborting&rgb decode n70292_2 gets an extra combinational level. Final state/reset qualification n67360_2 remains final-LUT input. Other8 n70854_2 consumers retain original function.',
  evidence={str(x):sha(x) for x in [__file__,b/'outflow/ti60f225_oob.map.v',ROOT/'tmp/frame-execution/video-control84-canonical1/outflow/canonical_setup.rpt']},
  hardware_access=False,board_release=False,physical_pass=False)
 out=ROOT/'tmp/frame-execution/video-control84-canonical1/dma-ce87-proposal.json';assert not out.exists()
 out.write_text(json.dumps(result,indent=2));print(result['status'])
if __name__=='__main__':main()
