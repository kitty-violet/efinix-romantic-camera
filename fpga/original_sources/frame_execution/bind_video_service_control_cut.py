"""Bind every actually needed control cut through current retained-map proof."""
import json,re
from native_local_drivers import lexical_tokens
from run_units import sha

def bind_control_cut(source,mapped,cuts,external,board):
 proof=json.loads((board/'retained-map-audit.json').read_text());assert proof['status']=='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS' and proof['conflicts']==0
 assert proof['source_sha256']==sha(board/'native_source.v') and proof['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
 outputs={'EFX_LUT4':('O',),'EFX_FF':('Q',),'EFX_ADD':('O','CO'),'EFX_SRL8':('Q','Q7'),'EFX_DSP24':('O','CASCOUT'),'EFX_DSP48':('O','CASCOUT'),'EFX_RAM10':('RDATA',),'EFX_DPRAM10':('RDATA',)}
 drivers={t:(n,p,j) for n,(k,ps) in source.items() for p in outputs.get(k,()) for j,t in enumerate(ps.get(p,[])) if t[0]=='net'}
 def protected(text):
  i=text.index('protected_cell protected_inst (');s=text[i:text.index(';',i)]
  return {n.lstrip('\\'):lexical_tokens(v) for n,v in re.findall(r'\.(\\\S+|\w+)\s*\(([^()]*)\)',s)}
 pp=[protected((board/f).read_text()) for f in ('native_source.v','outflow/ti60f225_oob.map.v')];assert pp[0].keys()==pp[1].keys()
 protected_map={}
 for p,ts in pp[0].items():
  assert len(ts)==len(pp[1][p])
  for t,u in zip(ts,pp[1][p]):
   if t[0]=='net':
    assert t not in protected_map or protected_map[t]==u
    protected_map[t]=u
 out={};rows=[]
 for t,value in cuts.items():
  method='';mt=None
  if t in drivers:
   n,p,j=drivers[t]
   if n in mapped:
    assert mapped[n][0]==source[n][0] and len(mapped[n][1][p])==len(source[n][1][p]);mt=mapped[n][1][p][j];method='retained_driver'
  elif t in protected_map:mt=protected_map[t];method='exact_protected_port_bit'
  else:
   candidate={mapped[n][1][p][j] for n,(k,ps) in source.items() if k!='EFX_LUT4' and n in mapped for p,ts in ps.items() if p not in outputs.get(k,()) and p in mapped[n][1] and len(ts)==len(mapped[n][1][p]) for j,x in enumerate(ts) if x==t}
   if candidate:assert len(candidate)==1;mt=next(iter(candidate));method='retained_non_lut_input'
   elif t in external:mt=t;method='identical_named_boundary_in_full_LUT_function_proof'
  if mt not in external:continue
  assert mt not in out or out[mt]==value,('unproved cut alias',t,mt)
  out[mt]=value;rows.append(dict(source=t,mapped=mt,value=value,method=method))
 assert external<=set(out),('missing actual cut',external-set(out))
 return out,rows
