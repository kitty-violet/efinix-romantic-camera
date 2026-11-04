"""Retained map audit with only audited A-zero aliases normalized."""
import json,re
from pathlib import Path
from run_units import sha
from native_local_drivers import lexical_tokens
from camera_joint_native import OUT
def normalize_a_zero_aliases(a,source,board):
 meta=json.loads((board/'inputs.json').read_text());native=Path(meta['native'])
 proof=json.loads((native/'weight-zero-independent-audit.json').read_text())
 assert proof['status']=='GUARDED_A_ZERO_EXACT_SOURCE_GRAPH_PASS_NOT_PHYSICAL'
 assert proof['source_sha256']==sha(native/'native_source.v')==sha(board/'native_source.v')
 for f,h in proof['inputs'].items():assert sha(f)==h
 record=json.loads((native/'weight-zero-edit.json').read_text())
 expected={tuple(k):tuple(v) for k,v in record['constants']};actual={}
 for left,right in re.findall(r'^\s*assign\s+([^;=]+)=([^;]+);',source,re.M):
  try:x,y=lexical_tokens(left),lexical_tokens(right)
  except AssertionError:continue
  if len(x)==len(y)==1 and x[0] in expected:actual[x[0]]=y[0]
 assert len(actual)==246 and actual==expected
 count=0
 for n,(kind,pins) in a.items():
  for port,ts in pins.items():
   if port in OUT[kind]:assert not any(t in actual for t in ts),(n,port)
   else:
    count+=sum(t in actual for t in ts);pins[port]=[actual.get(t,t) for t in ts]
 return dict(aliases=len(actual),primitive_input_uses=count,proof_sha256=sha(native/'weight-zero-independent-audit.json'))
source=Path(__file__).with_name('audit_video_service_rq_map.py');code=source.read_text()
old="insert = ' expand_cached_ports(a,source);expand_cached_ports(b,mapped)\\n'+needle"
new="insert = ' expand_cached_ports(a,source);expand_cached_ports(b,mapped)\\n a_zero=normalize_a_zero_aliases(a,source,board)\\n'+needle"
assert code.count(old)==1;code=code.replace(old,new)
old="updates = [";new="updates = [(\"retained=len(a),\", \"a_zero_aliases=a_zero,retained=len(a),\"),"
assert code.count(old)==1;code=code.replace(old,new)
exec(compile(code,str(source)+':a-zero-aliases','exec'),globals())
