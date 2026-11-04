"""Bind actual production-frame timings to the complete-board frozen sources."""
import argparse,json,re
from pathlib import Path
from run_units import ROOT,sha
from audit_connected_system import manifest

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--board',default='connected-board-row-pair1');p.add_argument('--frames',nargs=2,default=['row-pair-production-frame1','row-pair-production-frame-random1'])
p.add_argument('--physical',default='row-pair-physical1');p.add_argument('--descriptors',default='row-pair-descriptors2');a=p.parse_args()
assert not a.output.exists();base=ROOT/'tmp/frame-execution'
board=base/a.board;br=json.loads((board/'result.json').read_text());_,bs=manifest(board)
for file,key in [('inputs.json','inputs_sha256'),('outflow/ti60f225_oob.map.rpt','report_sha256'),('outflow/ti60f225_oob.map.v','map_sha256')]:assert sha(board/file)==br[key]
frames=[]
bench={'efx_dsp24.v','efx_dsp48.v','efx_srl8.v','efx_ram10.v','efx_dpram10.v','frame_execution_meter.sv','tb_fourway_full.sv'}
for name in a.frames:
 root=base/name;r=json.loads((root/'result.json').read_text());m,s=manifest(root)
 assert r['status']=='CONTINUOUS_720P_RTL_NUMERIC_MEASUREMENT_PASS'
 assert sha(root/'inputs.json')==r['inputs_sha256'] and sha(root/'simulation.log')==r['log_sha256']
 assert sha(m['binary'])==m['binary_sha256']
 shared=[]
 for n,h in s.items():
  if n in bench:continue
  assert n in bs and bs[n]==h,n
  shared.append(n)
 assert r['line'][1:2]==['19229'] and r['line'][3:]==['65952000','2764800','476156','403200','6390','1fffffff']
 assert r['descriptor_manifest_sha256']==sha(base/a.descriptors/'result.json')
 compute=sum(n['cycles_with_header_control'] for n in r['nodes'])
 frames.append(dict(name=name,result_sha256=sha(root/'result.json'),inputs_sha256=r['inputs_sha256'],
  log_sha256=r['log_sha256'],cycles=r['cycles'],frame_ms=r['measured_frame_ms'],clock_hz_assumed=r['clock_hz_assumed'],
  equivalent_serial_cnn_fps=150000000/r['cycles'],node_cycles=compute,between_commands_cycles=r['cycles']-compute,
  dma_overlap=r['dma_overlap'],nodes=r['nodes'],stalls=r['stalls'],shared_sources=shared,host_seconds=r['simulation_seconds']))
physical=base/a.physical;pi=json.loads((physical/'physical_inputs.json').read_text())
assert pi['base_map_sha256']==br['map_sha256'] and pi['base_inputs_sha256']==br['inputs_sha256']
log=physical/'outflow/ti60f225_oob.place.out';s=log.read_text();m=re.search(r'capacity=(\d+) usage=(\d+)',s);assert m
result=dict(status='ROW_PAIR_PRODUCTION_AND_BOARD_EVIDENCE_BOUND',frames=frames,board=str(board),resources=br['resources'],
 packed_logic_usage=int(m[2]),usable_logic_capacity=int(m[1]),physical_log=str(log),physical_log_sha256=sha(log),
 board_inputs_sha256=br['inputs_sha256'],board_map_sha256=br['map_sha256'],
 full_graph_numeric_pass=True,resource_fit=False,strict_sta_pass=False,whole_board_frame_interval_measured=False,
 boundaries=['diagnostic weights','150MHz assumption','behavioral DDR; no PHY/refresh/other masters',
 'host coefficient upload testbench driven','gesture/demon/true-FPS leaves not integrated'],board_release=False,script_sha256=sha(__file__))
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('frames','boundaries')}))
