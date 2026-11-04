"""Retain exact mapped pin binding when Efinity renames a boundary net."""
from pathlib import Path
source=Path(__file__).with_name('validate_video_service_fault_ready.py');code=source.read_text()
code=code.replace('key,_,new,_,np=','key,source_graph,new,_,np=')
needle="  driven={g[name][1]['O'][0] for name in selected};external=set(tokens)-driven\n  assert external<=set(boundary),(side,external-set(boundary))\n  for t in external:lines.append(f'assign {value(t)}=stim[{boundary.index(t)}];')"
replacement="""  driven={g[name][1]['O'][0] for name in selected};external=set(tokens)-driven
  bindings={t:i for i,t in enumerate(boundary)}
  if side==1 and board:
   bindings={}
   # The retained-map audit has proved instance/port identity. Resolve the
   # independent cut inputs by those exact ports, not by similar net names.
   for name in selected:
    for pin,old_ts in source_graph[name][1].items():
     if pin=='O':continue
     new_ts=g[name][1][pin];assert len(old_ts)==len(new_ts)
     for old_t,new_t in zip(old_ts,new_ts):
      if old_t in boundary:
       idx=boundary.index(old_t);assert new_t not in bindings or bindings[new_t]==idx
       bindings[new_t]=idx
  assert external<=set(bindings),(side,external-set(bindings))
  for t in external:lines.append(f'assign {value(t)}=stim[{bindings[t]}];')"""
assert needle in code;code=code.replace(needle,replacement)
code=code.replace("paths=[Path(__file__).resolve(),", "paths=[Path(__file__).resolve(),source,")
exec(compile(code,str(source)+':mapped-boundary-binding','exec'),globals())
