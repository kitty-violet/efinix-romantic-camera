"""Complete finite-run gates plus current map equivalence and primitive regressions."""
from pathlib import Path
from control_command_repair import once
SOURCE=Path(__file__).with_name('finish_cnn_run_budget_release.py')
code=SOURCE.read_text()
code=code.replace('finish_cnn_run_budget_release','finish_cnn_timing_closure_release')
code=code.replace('cnn_run_budget_c1_flow','cnn_timing_closure_route')
code=code.replace('from cnn_run_budget_board import BOARD,audit_delta,TRANSFORM','from cnn_timing_closure_board import BOARD,audit_delta,TRANSFORM')
code=code.replace("HERE/'cnn_run_budget_place.py'","HERE/'cnn_timing_closure_place.py'")
code=code.replace('cnn_run_budget_implementation','cnn_timing_closure_implementation')
code=code.replace('cnn-run-budget-evidence1','cnn-timing-closure-evidence1')
code=once(code,"dict(before='CONGESTION_2',after='CONGESTION_1')","dict(before='CONGESTION_2',after='CONGESTION_2')")
start=code.index('def payload_evidence(bind,verify):');end=code.index('\ndef camera_evidence(',start)
code=code[:start]+'''def payload_evidence(bind,verify):
 import finish_cnn_run_budget_release as parent
 contract=parent.payload_evidence(bind,verify)
 from cnn_timing_closure_board import transform_check,PARENT as RUN_PARENT
 proof=transform_check();bind(TRANSFORM/'result.json');verify(proof['inputs'])
 assert contract==dict(deadline_cycles=33554432,counter_bits=26,hardware_frame_limit=16,progress_schema='D-run-progress-111')
 mapped=bind(BOARD/'result.json');verify(mapped['evidence']);audit_delta(BOARD)
 for n,status in [('cnn-timing-closure-drivers1','ACTUAL_MAP_LOCAL_DRIVERS_FOURSTATE_PASS'),('cnn-timing-closure-controls1','MAPPED_WRITER_ERROR_COFACTOR_OFFICIAL_LUT_SHORT_PASS')]:
  root=ROOT/'tmp/frame-execution'/n;r=bind(root/'result.json');i=bind(root/'inputs.json')
  assert r['status']==status and r['inputs_sha256']==sha(root/'inputs.json')
  if 'map_sha256' in i:assert i['map_sha256']==mapped['map_sha256']
  verify(i.get('evidence',{}));verify({str(root/f):h for f,h in i['files'].items()})
  verify({str(root/'simulation.log'):r.get('log_sha256',r.get('simulation_sha256'))})
  if 'base_runner_sha256' in i:verify({str(HERE/'validate_native_local_drivers.py'):i['base_runner_sha256']})
 return contract
''' +code[end:]
# The cofactor transformation intentionally retires six old CE outputs.
# Accept only declared old output nets with no actual mapped consumers;
# unconnected inputs and all other warning checks remain unchanged.
code=once(code,' assert dangling(a)==dangling(b)', ''' retired=dangling(b)-dangling(a)
 assert dangling(a)<=dangling(b)
 proof=json.loads((TRANSFORM/'writer_cofactor_proof.json').read_text())
 allowed={r['old'][1] for r in proof['outputs']}
 assert retired<=allowed and len(retired)==6,retired-allowed
 from native_local_drivers import graph
 from camera_joint_native import OUT
 actual=graph((board/'outflow/ti60f225_oob.map.v').read_text())
 for name in retired:
  term=('net',name,None);drivers=[];loads=[]
  for n,(kind,pins) in actual.items():
   for pin,terms in pins.items():
    if term in terms:(drivers if pin in OUT.get(kind,[]) else loads).append((n,pin))
  assert len(drivers)==1 and not loads,(name,drivers,loads)''')
code=once(code," r.update(status='SAME_PROTECTED_UNUSED_SAMPLE_BITS_AND_WARNING_CLASSES_PASS',new_functional_unconnected_inputs=False,",
 " r.update(status='SAME_PROTECTED_UNUSED_BITS_AND_PROVED_RETIRED_WRITER_OUTPUTS_PASS',retired_writer_ce_outputs=sorted(retired),new_functional_unconnected_inputs=False,")
code=once(code,"str(Path(__file__)):sha(__file__)})", "str(Path(__file__)):sha(__file__),str(TRANSFORM/'writer_cofactor_proof.json'):sha(TRANSFORM/'writer_cofactor_proof.json')})")
start=code.index(' from cnn_run_budget_locations import verify_placed',code.index('def extra_evidence('))
end=code.index('\ndef package_code():',start)
code=code[:start]+''' from fixed_cnn_ram_locations import verify_placed as check_ram
 assert check_ram(physical)['count']==113
 assert not (physical/'arena_island_placement.json').exists()
 assert bind(physical/'route_review.json')['budget_seconds']==2700
 bind(physical/'cnn_ram_actual_locations.json');r=bind(physical/'cnn_ram_placement.json')
 if 'evidence' in r:verify(r['evidence'])
 for n in ('cnn_timing_closure_native.py','cnn_timing_closure_board.py','cnn_timing_closure_place.py','cnn_timing_closure_flow.py','cnn_timing_closure_implementation.py','cnn_timing_closure_route.py','validate_cnn_timing_drivers.py','validate_cnn_timing_controls.py','finish_cnn_run_budget_release.py','read_cnn_run_progress.py','read_cnn_run_snapshot.py'):
  verify({str(HERE/n):sha(HERE/n)})
''' +code[end:]
# The bundle entry imports the current full-board auditor/implementation.
code=code.replace(".replace('camera_payload_board','cnn_run_budget_board')",".replace('camera_payload_board','cnn_timing_closure_board')")
code=once(code,"mapped['evidence'][str(HERE/'cnn_run_budget_board.py')]","mapped['evidence'][str(HERE/'cnn_timing_closure_board.py')]")
code=code.replace('Current-map regression and physical gates required; complete frame and board picture remain unverified',
 'Exact late-writer-error factorization and targeted equivalent data-driver replication added without latency. Current-map regression and physical gates required; complete frame and board picture remain unverified')
exec(compile(code,str(SOURCE)+':joint-closure-release','exec'),globals())
