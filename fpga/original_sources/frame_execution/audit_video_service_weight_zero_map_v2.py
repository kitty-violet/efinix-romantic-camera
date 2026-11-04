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
exec(compile(code,str(source)+':exact-binary-vectors','exec'),globals())
