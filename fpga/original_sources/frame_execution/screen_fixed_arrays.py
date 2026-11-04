"""Necessary arithmetic/RQ bounds, not an RTL or memory-throughput claim."""
import argparse,collections,hashlib,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--program',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists()
commands=json.loads(a.program.read_text());nodes=collections.defaultdict(collections.Counter)
configs=[('dot9_7x4_rq32',7,4,9,32,126),('dot9_8x4_rq16',8,4,9,16,144),
 ('dot9_4x7_rq32',4,7,9,32,126),('dot8tail_4x8_rq32',4,8,8,32,128)]
totals={name:collections.Counter() for name,*_ in configs}
for x in commands:
 if x['kind']!='compute':continue
 w,h=x['pixels'],x['rows'];ci=x['input_groups']*8;co=x['output_groups']*int(x['mask']).bit_count()
 terms=27 if x['operator']=='conv3x3' else 9 if x['operator']=='dw3x3' else ci
 scalar=w*h*co;rq=scalar*(3 if x['skip'] else 1)
 nodes[x['node']].update(scalars=scalar,macs=scalar*terms,rq=rq,commands=1)
 for name,sp,ch,dot,rqlanes,dsps in configs:
  # 7x4 schedules along x; 4x8 uses four independent rows. Do not
  # credit combining tail rows across commands with independent lifetimes.
  spatial_batches=(math.ceil(w/sp)*h if sp!=4 else w*math.ceil(h/sp))
  step_terms=9 if name.startswith('dot8tail') and x['operator']!='pw1x1' else dot
  mac_cycles=spatial_batches*math.ceil(co/ch)*math.ceil(terms/step_terms)
  rq_cycles=math.ceil(rq/rqlanes)
  totals[name].update(mac_cycles=mac_cycles,rq_cycles=rq_cycles,
   optimistic_cycles=max(mac_cycles,rq_cycles))
result=dict(status='NECESSARY_BOUNDS_ONLY_NOT_PERFORMANCE_PASS',program=str(a.program.resolve()),
 program_sha256=hashlib.sha256(a.program.read_bytes()).hexdigest(),nodes=nodes,candidates=[])
for name,sp,ch,dot,rq,dsp in configs:
 t=dict(totals[name]);t.update(name=name,mac_physical_dsp=dsp,rq_physical_dsp=rq,
   physical_dsp_sum=dsp+rq,optimistic_ms_at_150=t['optimistic_cycles']/150000,
   arithmetic_bound_below_30fps_budget=t['optimistic_cycles']<=5000000)
 t['omitted_from_bound']=['RAM ports/actual layout','coefficient supply','DMA and dependencies','command setup/drain','RQ return routing','other board masters','physical timing']
 result['candidates'].append(t)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['candidates'],indent=2))
