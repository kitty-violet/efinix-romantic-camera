"""Classify current warnings, proving every newly dangling output has no load."""
import ast,json,re
from pathlib import Path
from run_units import sha
from control_remap_cut import boundaries
from continuous_control_native import connections

def extra_literal(name):
 p=Path(__file__).with_name(name);tree=ast.parse(p.read_text())
 rows=[x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='extra' for t in x.targets)]
 assert len(rows)==1;return ast.literal_eval(rows[0].value)

def prove_new_dangling(text,body,old,g,board):
 before=set(re.findall(r'Net (.*?) is dangling',old));now=set(re.findall(r'Net (.*?) is dangling',body));assert before<=now
 meta=json.loads((board/'inputs.json').read_text());native=Path(meta['native']);edit=json.loads((native/'control-remap-edit.json').read_text());proof=json.loads((native/'control-remap-independent-audit.json').read_text())
 assert proof['status']=='CONTROL_REMAP_FULL_SOURCE_CEC_PASS_NOT_PHYSICAL' and proof['source_sha256']==sha(board/'native_source.v')
 for f,h in proof['inputs'].items():assert sha(f)==h
 outside,_,bm=boundaries(g,text,set(edit['new_cells']));dr,loads=connections(g);rows=[]
 for netname in sorted(now-before):
  t=('net',netname,None);assert t in dr;driver=dr[t]
  assert driver in edit['outputs'] and driver in edit['new_cells'] and driver not in outside and not loads.get(t),('not proved unloaded',netname,driver)
  assert g[driver][0]=='EFX_LUT4'
  rows.append(dict(net=netname,driver=driver,public_consumers=0,external_or_protected_consumers=0,reason='extra conservative CEC observation output retained with syn_keep; no functional consumer'))
 assert len(rows)==37
 (board/'control-unused-output-proof.json').write_text(json.dumps(dict(status='ALL_NEW_DANGLING_OUTPUTS_PROVED_UNCONSUMED',rows=rows,boundary_metadata=bm,source_sha256=sha(board/'native_source.v'),map_sha256=sha(board/'outflow/ti60f225_oob.map.v'),proof_sha256=sha(native/'control-remap-independent-audit.json'),hardware_access=False),indent=2))
 return {r['net'] for r in rows}

source=Path(__file__).with_name('audit_video_service_synthesis.py');code=source.read_text()
old=" assert set(re.findall(r'Net (.*?) is dangling',body))==set(re.findall(r'Net (.*?) is dangling',old))"
assert code.count(old)==1;code=code.replace(old," new_dangling=prove_new_dangling(text,body,old,g,board)")
anchor='  if norm(line) in inherited:'
extra=extra_literal('audit_video_service_rq_warnings.py')+extra_literal('audit_video_service_protected_warnings.py')
extra += "  dangling=re.search(r'Net (.*?) is dangling',line)\n  if dangling and dangling[1] in new_dangling:counts['proved_unloaded_conservative_control_output']+=1;continue\n"
assert code.count(anchor)==1;code=code.replace(anchor,extra+anchor).replace("name.removeprefix('protected_inst/')","name.replace('protected_inst/','')")
code=code.replace('evidence=[src,warn,oldwarn,',"evidence=[board/'control-unused-output-proof.json',Path(__file__).with_name('control_remap_cut.py'),source,Path(__file__).with_name('audit_video_service_rq_warnings.py'),Path(__file__).with_name('audit_video_service_protected_warnings.py'),src,warn,oldwarn,")
exec(compile(code,str(source)+':all-current-control-warnings','exec'),globals())
