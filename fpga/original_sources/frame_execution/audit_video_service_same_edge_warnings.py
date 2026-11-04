"""Classify current synthesis warnings, proving every new unconsumed output."""
import ast,json,re
from pathlib import Path
from run_units import sha
from control_remap_cut import boundaries
from continuous_control_native import connections

def extra_literal(name):
 tree=ast.parse(Path(__file__).with_name(name).read_text());rows=[x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='extra' for t in x.targets)];assert len(rows)==1;return ast.literal_eval(rows[0].value)
def prove_new_dangling(text,body,old,g,board):
 before=set(re.findall(r'Net (.*?) is dangling',old));now=set(re.findall(r'Net (.*?) is dangling',body));assert before<=now
 native=Path(json.loads((board/'inputs.json').read_text())['native']);proof=json.loads((native/'same-edge-independent-audit.json').read_text());assert proof['status']=='SAME_EDGE_ENABLE_COMPOSITIONAL_EQUIVALENCE_PASS_NOT_PHYSICAL' and proof['source_sha256']==sha(board/'native_source.v')
 for f,h in proof['inputs'].items():assert sha(f)==h
 dr,loads=connections(g);rows=[];owned=set()
 for name in now-before:
  candidates=[t for t in dr if t[1]==name and t[2] is None];assert len(candidates)==1,(name,candidates)
  t=candidates[0];n=dr[t];assert g[n][0]=='EFX_LUT4' and not loads.get(t),(name,n,loads.get(t));owned.add(n);rows.append(dict(net=name,driver=n,public_consumers=0,external_or_protected_consumers=0))
 outside,_,bm=boundaries(g,text,owned);assert not outside,outside
 (board/'same-edge-unused-output-proof.json').write_text(json.dumps(dict(status='ALL_NEW_DANGLING_OUTPUTS_PROVED_UNCONSUMED',rows=rows,boundary_metadata=bm,source_sha256=sha(board/'native_source.v'),map_sha256=sha(board/'outflow/ti60f225_oob.map.v'),proof_sha256=sha(native/'same-edge-independent-audit.json'),hardware_access=False),indent=2));return {r['net'] for r in rows}
source=Path(__file__).with_name('audit_video_service_synthesis.py');code=source.read_text();old=" assert set(re.findall(r'Net (.*?) is dangling',body))==set(re.findall(r'Net (.*?) is dangling',old))";assert code.count(old)==1;code=code.replace(old,' new_dangling=prove_new_dangling(text,body,old,g,board)')
anchor='  if norm(line) in inherited:';extra=extra_literal('audit_video_service_rq_warnings.py')+extra_literal('audit_video_service_protected_warnings.py')
extra+="  dangling=re.search(r'Net (.*?) is dangling',line)\n  if dangling and dangling[1] in new_dangling:counts['proved_unloaded_old_decode_output']+=1;continue\n"
extra+='''  if line.startswith('[EFX-0200 WARNING] Removing redundant signal : init_cur_state[6:2]. '):
   decl=r'wire\\s*\\[6:0\\]\\s*init_cur_state\\s*;'
   assert len(re.findall(decl,text))==1
   # The discarded declaration bits must have no public, concatenation or
   # protected-port reference. The two real DDR observation bits remain live.
   refs=re.findall(r'(?<![\\w.])init_cur_state\\s*(\\[[^\\]]+\\])?',re.sub(decl,'',text))
   assert refs and set(refs)=={'[0]','[1]','[1:0]'},refs
   assert proof['protected_input_unchanged'] and proof['protected_boundary']['other_connections_equal']
   reason='proved_declaration_only_init_state_bits_6_to_2'
   counts[reason]+=1;new.append(dict(warning=line,reason=reason));continue
'''
assert code.count(anchor)==1;code=code.replace(anchor,extra+anchor).replace("name.removeprefix('protected_inst/')","name.replace('protected_inst/','')")
code=code.replace('evidence=[src,warn,oldwarn,',"evidence=[board/'same-edge-unused-output-proof.json',Path(__file__).with_name('control_remap_cut.py'),source,Path(__file__).with_name('audit_video_service_rq_warnings.py'),Path(__file__).with_name('audit_video_service_protected_warnings.py'),src,warn,oldwarn,")
exec(compile(code,str(source)+':same-edge-current-warnings','exec'),globals())
