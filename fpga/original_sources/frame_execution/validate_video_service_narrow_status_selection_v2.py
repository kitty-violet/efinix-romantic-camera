"""Bind a mapper-renamed duplicate state only after exact FF input proof."""
from pathlib import Path
source=Path(__file__).with_name('validate_video_service_narrow_status_selection.py')
code=source.read_text()
old="def state_name(n):return OLD_ALIAS if n==NEW_ALIAS else n"
new="""STATE_ALIASES={}
def state_name(n):return STATE_ALIASES.get(n,OLD_ALIAS if n==NEW_ALIAS else n)

def prove_front_bank_alias(old,new):
 a='service_front_bank[1]~FF';b='service_owner/disp_slot[1]~FF'
 assert a not in new['g'] and old['g'][a][0]==new['g'][b][0]=='EFX_FF'
 assert old['p'][a]==new['p'][b], 'duplicate state FF attributes differ'
 ap=old['g'][a][1];bp=new['g'][b][1]
 assert {k:v for k,v in ap.items() if k!='Q'}=={k:v for k,v in bp.items() if k!='Q'}, 'duplicate state input connections differ'
 assert set(ap)==set(bp)=={'D','CE','CLK','SR','Q'}
 assert old['p'][a]['SR_VALUE']==(1,0) and old['p'][a]['SR_SYNC']==(1,1)
 # Same clock, data, enable and reset with identical polarity/reset value:
 # both states agree after reset and inductively at every subsequent edge.
 STATE_ALIASES[b]=a
 return dict(old=a,new=b,all_non_output_pins_identical=True,all_parameters_identical=True,reset_inductive_state_alias=True)
"""
assert code.count(old)==1;code=code.replace(old,new)
old=' merge=prove_merged_fault_bit(old,new)'
assert code.count(old)==1;code=code.replace(old,old+'\n front_merge=prove_front_bank_alias(old,new)')
old='   new_name=NEW_ALIAS if n==OLD_ALIAS else n'
assert code.count(old)==1;code=code.replace(old,"   new_name=next((alias for alias,original in STATE_ALIASES.items() if original==n),NEW_ALIAS if n==OLD_ALIAS else n)")
code=code.replace('merged_state_proof=merge,','merged_state_proof=merge,front_bank_alias_proof=front_merge,')
old="  inputs={str(c['path']):sha(c['path']) for c in (old,new)},checker_sha256=sha(__file__),"
assert code.count(old)==1
code=code.replace(old,"  inputs={**{str(c['path']):sha(c['path']) for c in (old,new)},str(source):sha(source),str(Path(__file__)):sha(__file__)},checker_sha256=sha(__file__),")
exec(compile(code,str(source)+':proved-duplicate-FF-alias','exec'),globals())
