"""Preserve complete release gates and bind the additional exact ready proof."""
from pathlib import Path
source=Path(__file__).with_name('finish_camera_payload_release.py')
code=source.read_text(encoding='utf8')
code=code.replace('from camera_payload_board import BOARD,PARENT as CAMERA_PARENT,audit_delta','from camera_ready_board import BOARD,audit_delta\nfrom camera_checked_v2 import BOARD as CAMERA_PARENT')
code=code.replace('camera_payload_board','camera_ready_board').replace('camera_payload_implementation','camera_ready_implementation').replace('camera_payload_flow','camera_ready_flow').replace('finish_camera_payload_release','finish_camera_ready_release')
code=code.replace('camera-payload-evidence1','camera-ready-evidence1').replace('camera-payload-reuse1','camera-ready-reuse3')
code=code.replace('UNCHANGED_CAMERA_MAP_TESTS_BOUND_TO_PAYLOAD_MAP','UNCHANGED_CAMERA_MAP_TESTS_BOUND_TO_READY_MAP')
needle='    return dict(clones=59,payload_bits=768,max_driver_fanout=18,added_cycles=0,official_ram_seeds=3)'
assert code.count(needle)==1
code=code.replace(needle,'    ready=ready_evidence(bind,verify)\n'+needle[:-1]+',ready_factoring=ready)')
needle="    allowed={r['old_select'][1] for r in groups}"
assert code.count(needle)==1
code=code.replace(needle,needle+'\n    allowed.update(ready_retired_signals())')
code=code.replace('Full video/A retained, checked camera configuration plus proved local RAM payload specialization; current-map tests and physical gates required',
 'Full video/A retained, checked camera configuration, proved local RAM payload selection and exact combinational ready factoring; current-map tests and physical gates required')
# Keep definitions available before main invokes the generated release runner.
code=code.replace("if __name__=='__main__':main()",'')
exec(compile(code,str(source)+':ready-bound','exec'),globals())

def ready_evidence(bind,verify):
 src=ROOT/'tmp/frame-execution/camera-ready-native4';r=bind(src/'result.json')
 assert r['status']=='READY_FACTOR_GRAPH_AND_EXHAUSTIVE_BINARY_PASS' and r['checks']==1696
 assert r['added_luts']==10 and r['changed_existing_luts']==7 and r['added_cycles']==r['added_ff']==r['added_ram']==r['added_dsp']==0
 verify(r['inputs']);verify({str(src/f):h for f,h in r['files'].items()})
 test=ROOT/'tmp/board-test/camera-ready-fourstate1';t=bind(test/'result.json');i=bind(test/'inputs.json')
 assert t['status']=='READY_FACTOR_EXHAUSTIVE_OFFICIAL_FOURSTATE_PASS' and t['checks']==1315840 and t['cuts']==7
 assert t['inputs_sha256']==sha(test/'inputs.json') and i['source_result_sha256']==sha(src/'result.json')
 verify({str(test/f):h for f,h in i['files'].items()});verify({str(test/'simulation.log'):t['log_sha256'],str(HERE/'test_camera_ready_factor.py'):i['runner_sha256']})
 for n in ('camera_ready_board.py','camera_ready_flow.py','camera_ready_implementation.py','camera_ready_reuse.py','camera_payload_flow.py','camera_payload_reuse.py'):
  verify({str(HERE/n):sha(HERE/n)})
 return dict(binary_cases=1696,official_fourstate_cases=1315840,changed_cuts=7,added_luts=10,added_cycles=0)

def ready_retired_signals():
 from collections import Counter
 from camera_ready_board import PARENT as READY_PARENT
 from native_local_drivers import graph
 before=graph((READY_PARENT/'outflow/ti60f225_oob.map.v').read_text());after=graph((BOARD/'native_source.v').read_text())
 def loads(g):
  c=Counter()
  for n,(k,pins) in g.items():
   output={'EFX_LUT4':'O','EFX_FF':'Q','EFX_ADD':'O','EFX_SRL8':'Q'}.get(k)
   for p,ts in pins.items():
    if p!=output:c.update(ts)
  return c
 a,b=loads(before),loads(after)
 return {pins['O'][0][1] for n,(k,pins) in before.items() if k=='EFX_LUT4' and a[pins['O'][0]]>0 and b[pins['O'][0]]==0}

if __name__=='__main__':main()
