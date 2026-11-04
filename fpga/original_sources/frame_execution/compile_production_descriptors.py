"""Lossless hardware-field packing of the measured two-stream frame program.
This is an offline ABI candidate, not a synthesized scheduler or a throughput claim.
"""
import argparse,hashlib,json,struct
from pathlib import Path
from run_units import ROOT,sha
from resource_route_contract import check_program
COMPUTE_WIDTHS={1:5,2:9,3:3,4:3,5:3,6:6,7:11,8:11,9:11,10:12,11:32,12:28,13:8,14:32,15:8,16:15,17:15,18:15,19:15,20:8,21:32,22:8,23:11,24:18,25:18,28:8,29:29,30:10,31:15}
DMA_WIDTHS={0:1,1:5,2:9,3:3,4:2,5:10,6:1,7:28,8:1,9:10,10:11,30:10,31:15}
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--vectors',type=Path,required=True);p.add_argument('--specialized',action='store_true');a=p.parse_args()
route_contract_file=Path(__file__).with_name('resource_route_contract.py')
if a.specialized:
 from specialized_route_contract import check_program
 route_contract_file=Path(__file__).with_name('specialized_route_contract.py')
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False);vec=a.vectors.resolve()
m=json.loads((vec/'frame_manifest.json').read_text())
for name in ('program.hex','compute_commands.hex','compute_dependencies.hex','dma_commands.hex','dma_dependencies.hex'):
 assert sha(vec/name)==m['files'][name],name
program=[int(line,16) for line in (vec/'program.hex').read_text().splitlines()]
route_contract=check_program(vec/'program.hex')
results={}
for stream,widths,record_bytes in (('compute',COMPUTE_WIDTHS,64),('dma',DMA_WIDTHS,16)):
 indices=[int(x,16) for x in (vec/(stream+'_commands.hex')).read_text().splitlines()]
 deps=[int(x,16) for x in (vec/(stream+'_dependencies.hex')).read_text().splitlines()]
 assert len(indices)==len(deps)
 output=bytearray()
 schema=[('peer_completed',14),('program_index',15)] if stream=='compute' else [('peer_completed',14)]
 schema += [('f'+str(i),n) for i,n in widths.items()]
 bits=sum(n for _,n in schema);assert bits<=record_bytes*8
 for index,dep in zip(indices,deps):
  f=[(program[index]>>(32*i))&0xffffffff for i in range(32)]
  assert (f[0]==2)==(stream=='compute')
  excluded={0,26,27} if stream=='compute' else set()
  assert all(v==0 for i,v in enumerate(f) if i not in widths and i not in excluded),(stream,index,f)
  values=([dep,index] if stream=='compute' else [dep])+[f[i] for i in widths]
  packed=0;offset=0
  for value,(_,n) in zip(values,schema):
   assert 0<=value<(1<<n),(stream,index,value,n)
   packed |= value<<offset;offset+=n
  decoded=[];offset=0
  for _,n in schema:decoded.append((packed>>offset)&((1<<n)-1));offset+=n
  assert decoded==values
  output.extend(packed.to_bytes(record_bytes,'little'))
 (root/(stream+'.bin')).write_bytes(output)
 results[stream]=dict(records=len(indices),record_bytes=record_bytes,used_bits=bits,total_bytes=len(output),schema=schema,sha256=sha(root/(stream+'.bin')))
assert sum(r['records'] for r in results.values())==19229
total=sum(r['total_bytes'] for r in results.values());old=len(program)*128
result=dict(status='LOSSLESS_HARDWARE_DESCRIPTOR_ABI_CANDIDATE_PASS',route_contract=route_contract,route_contract_sha256=sha(route_contract_file),streams=results,total_bytes=total,original_tb_bytes=old,reduction_bytes=old-total,
 average_program_bytes_per_second_at_30fps=total*30,program_axi_128bit_beats=total//16,program_base=None,
 vector_manifest_sha256=sha(vec/'frame_manifest.json'),network_contract_sha256=sha(ROOT/'network_ab/model_b_contract.json'),
 excludes=['Golden offsets/strides f26/f27 are assertions only.','Compute opcode is implicit in its dedicated stream.','Reserved zero TB fields are not transmitted.'],
 preserves=['All live compute/DMA configuration fields.','Compute node-completion bitmap and tile index.','Both cross-engine completion dependencies.','Final RGB store coordinates/commit marking.'],
 hardware_scheduler_implemented=False,fifo_resource_measured=False,additional_latency_measured=False,
 safety='Program placement, fetch bounds, CRC, frame ownership and replay/abort handling remain production RTL obligations.')
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('status','total_bytes','original_tb_bytes','average_program_bytes_per_second_at_30fps','program_axi_128bit_beats')}))
