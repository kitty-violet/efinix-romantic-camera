"""Exact legacy cut audit plus independent truth proof for every CE-fold LUT."""
from pathlib import Path
source=Path(__file__).with_name('audit_video_service_native.py');code=source.read_text()
old="new=expand_graph(s);np=params(s);removed=set(proof['removed']);assert set(new)==set(old)-removed"
new="""new=expand_graph(s);np=params(s);removed=set(proof['removed'])
 rows=proof['enable_extension']['rows'];extra={r['lut'] for r in rows};assert len(rows)==len(extra)==1134
 assert set(new)==set(old)-removed|extra
 for r in rows:
  kind,pins=new[r['lut']];ports={k:[tuple(t)] for k,t in r['ports'].items()}
  assert kind=='EFX_LUT4' and pins==ports and np[r['lut']]=={'LUTMASK':(16,r['mask'])}
  ff,fp=new[r['ff']];p=D['ps'][r['ff']]
  assert ff=='EFX_FF' and p==np[r['ff']]
  assert fp['D']==ports['O'] and fp['CE']==[('literal',\"1'b\"+str(p['CE_POLARITY'][1]),None)]
  assert ports['I1']==fp['Q'] and ports['I2']==old[r['ff']][1]['CE']
  # Check next Q at both binary CE states, both old data and Q values.
  for q in range(2):
   for d in range(2):
    for ce in range(2):
     expected=d^(1-p['D_POLARITY'][1]) if ce==p['CE_POLARITY'][1] else q
     actual=((r['mask']>>(d+2*q+4*ce))&1)^(1-p['D_POLARITY'][1])
     assert actual==expected
"""
assert code.count(old)==1;code=code.replace(old,new)
code=code.replace(' for n,(k,pins) in new.items():',' for n,(k,pins) in new.items():\n  if n in extra:continue')
code=code.replace('retained_primitives=len(new),','retained_primitives=len(new),same_edge_feedback_luts=len(extra),binary_truth_cases=1134*8,')
exec(compile(code,str(source)+':enable-fold','exec'),globals())
