"""Keep each original physical CE load group distinct after factoring."""
from pathlib import Path
FACTOR_BASE=Path(__file__).with_name('camera_factor_native.py')
code=FACTOR_BASE.read_text()
changes={
 'def emit(expr):':'def emit(expr,unique=False):',
 'if expr in oldexpr:return oldexpr[expr]':'if not unique and expr in oldexpr:return oldexpr[expr]',
 'if expr in built:return built[expr]':'if not unique and expr in built:return built[expr]',
 "pars[name]={'LUTMASK':(16,mask)};built[expr]=tok":"pars[name]={'LUTMASK':(16,mask)}\n  if not unique:built[expr]=tok",
 "out=emit(normalize([leaf(low),leaf(high),('leaf',root),0],0xcaca))":"out=emit(normalize([leaf(low),leaf(high),('leaf',root),0],0xcaca),unique=True)",
 '(src,place,BASE,Path(__file__))':'(src,place,BASE,FACTOR_BASE,Path(__file__))',
}
for before,after in changes.items():
 assert code.count(before)==1,before
 code=code.replace(before,after)
exec(compile(code,str(FACTOR_BASE)+':preserve-ce-groups','exec'),globals())
