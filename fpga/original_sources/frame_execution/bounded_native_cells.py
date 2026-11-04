"""Bound each native cell regex to its own declaration, preserving parsing."""
import re
from audit_local_reset_implementation import norm
START=re.compile(r'^[ \t]*EFX_\w+[ \t]+\S+[ \t]*\(',re.M)
CELL=re.compile(r'^\s*(EFX_\w+)\s+(\S+)\s*\((.*?)\) /\*([^\n]*)',re.M|re.S)
def cells(text):
    starts=[m.start() for m in START.finditer(text)];out={}
    for i,start in enumerate(starts):
        block=text[start:starts[i+1] if i+1<len(starts) else len(text)]
        matches=list(CELL.finditer(block))
        assert len(matches)==1,('unsupported native cell declaration',block[:100])
        k,n,p,a=matches[0].groups();n=norm(n)
        assert n not in out,n
        out[n]=(k,{x:norm(v) for x,v in re.findall(r'\.(\w+)\(([^()]*)\)',p)},a)
    return out
