"""Bind integrated RTL, full-frame measurement, mapped resources and fault gates."""
import collections,json,re
from pathlib import Path
from run_units import ROOT,sha
from audit_compute_resource_attribution import parse
base=ROOT/'tmp/frame-execution'
def result(name):
 root=base/name;r=json.loads((root/'result.json').read_text());m=json.loads((root/'inputs.json').read_text())
 assert sha(root/'inputs.json')==r['inputs_sha256']
 for x in m['inputs']:assert sha(x['snapshot'])==x['sha256']
 return root,r,m
fr,frame,fm=result('datapath-frame1');mr,mapped,mm=result('datapath-map1')
frozen={Path(x['snapshot']).name:x['sha256'] for x in fm['inputs']}
for x in mm['inputs']:assert frozen[Path(x['snapshot']).name]==x['sha256'],x['path']
assert frame['cycles']==4898508 and frame['line'][1:]==['19229','4898508','65952000','2764800','430560','403200','6390','1fffffff']
assert sum(x['cycles'] for x in frame['reasons'])==frame['cycles']
assert sha(fr/'simulation.log')==frame['log_sha256']
assert sha(Path(fm['binary']))==fm['binary_sha256']
net=mr/'outflow/cnn_frame_datapath.map.v';report=mr/'outflow/cnn_frame_datapath.map.rpt'
assert sha(net)==mapped['map_sha256'] and sha(report)==mapped['report_sha256']
counts=dict(collections.Counter(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M)))
assert counts==mapped['resources']
rows=parse(mr/'outflow/cnn_frame_datapath.res.csv');top=rows[0];parts=[x for x in rows if x['depth']==1]
for key in top['total']:assert top['own'][key]+sum(x['total'][key] for x in parts)==top['total'][key]
assert top['total']['LUT4']==71526 and top['total']['RAM']==120
gates=[]
for name in ('datapath-dma-abort2','compact-dma-abort1','datapath-guard1','datapath-guard-fourstate2','production-barrel-unit1','production-barrel-diagonal-unit1','byte-rotated-unit1'):
 root,r,m=result(name)
 log=root/'simulation.log';key='simulation_sha256' if 'simulation_sha256' in r else 'log_sha256'
 assert sha(log)==r[key]
 gates.append(dict(name=name,status=r['status'],coverage=r['line']))
for name in ('datapath-dma-abort2','datapath-guard1','datapath-guard-fourstate2'):
 _,_,m=result(name)
 selected={Path(x['snapshot']).name:x['sha256'] for x in m['inputs']}
 for key in ('cnn_frame_datapath.sv','frame_arena_row_dma.sv','arena_client_router2.sv','tensor_arena12.sv'):
  if key in selected:assert selected[key]==frozen[key],(name,key)
out=dict(status='INTEGRATED_CNN_DATAPATH_EVIDENCE_VERIFIED_NOT_SYSTEM_FIT',frame=dict(cycles=frame['cycles'],target_clock_hz=150000000,ms=frame['measured_frame_ms'],line=frame['line']),
 resources=counts,physical_dsp=counts['DSP24']//2+counts['DSP48'],physical_ram=counts['RAM10']+counts['DPRAM10'],
 lut_only_over_device=counts['LUT4']-60800,
 resource_partition=[dict(name=x['name'],resources=x['total']) for x in parts]+[dict(name='top-owned',resources=top['own'])],
 gates=gates,verified_frame_inputs_sha256=sha(fr/'inputs.json'),verified_map_inputs_sha256=sha(mr/'inputs.json'),
 functional_datapath=True,hardware_fault_guard=True,production_scheduler=False,final_ddr_fabric_cdc=False,full_system_resource_fit=False,
 final_sta=False,board_test=False,full_baseline=False,notes=['LUT4 is not a placed XLR count; LUT alone already exceeds Ti60 capacity.',
 'Full frame uses finite real on-chip memories, synthetic diagnostic weights, precompiled TB commands and behavioral DDR.',
 'Guard tests do not exercise active CNN cancellation or continuous frame ownership. These remain required.',
 'compact_dma and alternate dispatch maps are separate candidates and are not silently adopted into the measured frame.'])
dest=ROOT/'output/toolchain/integrated_datapath_verified_2026-10-01.json'
assert not dest.exists();dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status=out['status'],cycles=frame['cycles'],lut=counts['LUT4'],gates=len(gates)),ensure_ascii=False))
