"""Unchanged health/path oracle with an explicitly proved APB signal cut."""
from pathlib import Path
source=Path(__file__).with_name('validate_video_service_narrow_status_selection_v2.py')
code=source.read_text()
anchor="exec(compile(code,str(source)+':proved-duplicate-FF-alias','exec'),globals())"
patch='''
code=code.replace('from run_units import ROOT,sha','from run_units import ROOT,sha\\nfrom shared_frontier_status_binding import prove_status_apb')
old=' front_merge=prove_front_bank_alias(old,new)';assert code.count(old)==1
code=code.replace(old,old+'\\n apb_binding=prove_status_apb(old,new)')
old="  values={}";assert code.count(old)==1
code=code.replace(old,"  values={('external',net('apb_enable')):trial&1}")
old=" g,d,ps=c['g'],c['d'],c['p'];leaves=set()";assert code.count(old)==1
code=code.replace(old,old+"\\n apb_token=net('apb_enable')")
old='  if t in forced:return forced[t]';assert code.count(old)==1
code=code.replace(old,"  if t==apb_token:\\n   leaves.add(('external',t));return state(('external',t))\\n"+old)
old='merged_state_proof=merge,';assert code.count(old)==1
code=code.replace(old,'apb_observation_binding=apb_binding,merged_state_proof=merge,')
old="inputs={**{str(c['path']):sha(c['path']) for c in (old,new)},";assert code.count(old)==1
code=code.replace(old,"inputs={**apb_binding['evidence'],**{str(c['path']):sha(c['path']) for c in (old,new)},")
'''
assert code.count(anchor)==1;code=code.replace(anchor,patch+'\n'+anchor)
exec(compile(code,str(source)+':proved-same-edge-status','exec'),globals())
