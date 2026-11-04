"""Exact retained-map audit, recognizing unused LOGIC carry pins only."""
from pathlib import Path
source=Path(__file__).with_name('audit_video_service_epoch_map.py')
outer=source.read_text().replace("=='service_page_sequence[4]'","=='service_legacy_epoch[1]'")
old="  else:assert q==z,('ports',n,q,z)"
new='''  else:
   if kind=='EFX_COMB4':
    assert pa[n]['MODE']==pb[n]['MODE']=='"LOGIC"'
    assert q.get('CI',"1'b0")==z.get('CI',"1'b0")=="1'b0"
    assert not any(q.get(p) or z.get(p) for p in ('CO','FCO'))
    for p in ('CI','CO','FCO'):q.pop(p,None);z.pop(p,None)
   assert q==z,('ports',n,q,z)'''
needle='code=source.read_text()'
assert outer.count(needle)==1
outer=outer.replace(needle,needle+'\nassert code.count('+repr(old)+')==1\ncode=code.replace('+repr(old)+','+repr(new)+')')
exec(compile(outer,str(source)+':explicit-unused-LOGIC-carry-ports','exec'),globals())
