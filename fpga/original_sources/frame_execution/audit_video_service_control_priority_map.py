CURRENT_PROOF=' name=\'LUT__140273\';dup=\'LUT__133287\';up=\'LUT__133283_LUT__140223_splitO\'\n assert set(a)-set(b)=={name,dup},(\'unexpected missing primitive\',set(a)-set(b))\n assert a[up]==b[up] and a[up][0]==\'EFX_LUT4\'\n k,p=a[name];src=p[\'I2\'][0];dst=p[\'O\'][0]\n assert k==\'EFX_LUT4\' and pa[name][\'LUTMASK\']==(16,0xf0f0) and a[up][1][\'O\']==[src]\n assert a[dup][0]==\'EFX_LUT4\' and pa[dup][\'LUTMASK\']==pa[up][\'LUTMASK\']\n assert {k:v for k,v in a[dup][1].items() if k!=\'O\'}=={k:v for k,v in a[up][1].items() if k!=\'O\'}\n ins=[p.get(\'I\'+str(i),[(\'literal\',"1\'b0",None)])[0] for i in range(4)]\n assert {t for t in ins if t[0]==\'net\'}=={src}\n ui=[a[up][1].get(\'I\'+str(i),[(\'literal\',"1\'b0",None)])[0] for i in range(4)]\n leaves=list(dict.fromkeys(t for t in ui if t[0]==\'net\'))\n count=0\n for values in itertools.product((0,1,\'x\',\'z\'),repeat=len(leaves)):\n  v=lut(pa[up][\'LUTMASK\'][1],ui,dict(zip(leaves,values)));assert v in (0,1,\'x\')\n  assert lut(pa[name][\'LUTMASK\'][1],ins,{src:v})==v;count+=1\n alias={dst:src,a[dup][1][\'O\'][0]:src}\n del a[name];del a[dup]\n for kind,ports in a.values():\n  for port,ts in ports.items():\n   if port not in OUT[kind]:ports[port]=[alias.get(t,t) for t in ts]\n return dict(cells=[name,dup],retained=up,alias_pairs=list(alias.items()),exhaustive_four_state_cases=count,upstream_never_z=True,all_loads_rebound=True)\n\n'
"""Canonicalize explicit binary vector constants before the exact map check."""
from pathlib import Path
source=Path(__file__).with_name('audit_video_service_weight_zero_map.py');code=source.read_text()
old='def normalize_a_zero_aliases(a,source,board):'
new='''def normalize_a_zero_aliases(a,b,source,board):
 # Efinity concatenates adjacent constant bits (e.g. 0,0 -> 2'b00).
 # Expand only sized binary literals into their exact Verilog bit order.
 # This proves representation equality, not a relaxed connection check.
 expanded=0
 for gr in (a,b):
  for kind,pins in gr.values():
   for pin,ts in pins.items():
    out=[]
    for t in ts:
     m=re.fullmatch(r"([0-9]+)'b([01xXzZ]+)",t[1]) if t[0]=='literal' else None
     if m and int(m[1])>1:
      width=int(m[1]);bits=m[2].lower();assert len(bits)<=width
      bits=bits.rjust(width,bits[0] if bits[0] in 'xz' else '0')
      out.extend(('literal',"1'b"+v,None) for v in bits);expanded+=1
     else:out.append(t)
    pins[pin]=out'''
assert code.count(old)==1;code=code.replace(old,new)
assert code.count('a_zero=normalize_a_zero_aliases(a,source,board)')==1
code=code.replace('a_zero=normalize_a_zero_aliases(a,source,board)','a_zero=normalize_a_zero_aliases(a,b,source,board)')
code=code.replace('return dict(aliases=len(actual),','return dict(expanded_binary_literals=expanded,aliases=len(actual),')
anchor="old=\"updates = [\";new="
# Insert a strict current-candidate identity-merge proof before the nested audit executes.
position=code.index('exec(compile(code,')
replacement = "def prove_rq_identity(a,b,pa):\n" + CURRENT_PROOF
patch="start=code.index('def prove_rq_identity(');end=code.index('\\ndef bind_lut_boundaries',start)\ncode=code[:start]+"+repr(replacement)+"+code[end:]\n"
code=code[:position]+patch+code[position:]
exec(compile(code,str(source)+':control-priority-merges','exec'),globals())
