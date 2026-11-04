"""Bind short-test snapshots to connected resource measurements, no board pass."""
import argparse,json
from pathlib import Path
from run_units import ROOT,HERE,sha
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--extended',action='store_true');a=p.parse_args();assert not a.output.exists()
base=ROOT/'tmp/frame-execution';checks=[]
for name in ('row-pair-arena-gate3','row-pair-reader-gate2','row-pair-reader-ram3','row-pair-subsystem-gate1','row-pair-aux-gate1','row-pair-writer-gate2')+ (('row-pair-dma-recycle2','row-pair-dma-recycle-routed1','row-pair-skip-gate2') if a.extended else ()):
 folder=base/name;r=json.loads((folder/'result.json').read_text());manifest=json.loads((folder/'inputs.json').read_text())
 assert 'PASS' in r['status']
 for record in manifest['inputs']:
  snap=Path(record['snapshot']);expected=record.get('snapshot_sha256',record.get('sha256'))
  assert sha(snap)==expected,(name,snap)
 runs=r.get('runs',[])
 for run in runs:assert sha(folder/f"simulation_{run['seed']}.log")==run['log_sha256']
 checks.append(dict(name=name,status=r['status'],result_sha256=sha(folder/'result.json'),inputs_sha256=sha(folder/'inputs.json')))
mapped=json.loads((base/'row-pair-subsystem-map1/inputs.json').read_text())
tested=json.loads((base/'row-pair-subsystem-gate1/inputs.json').read_text())
for rec in mapped['inputs']:
 matched=next(t for t in tested['inputs'] if Path(t['source']).name==Path(rec['source']).name)
 assert rec['sha256']==matched['source_sha256']==sha(rec['source'])
writer_map=json.loads((base/'row-pair-writer-map1/result.json').read_text())
writer_test=json.loads((base/'row-pair-writer-gate2/inputs.json').read_text())
assert writer_map['source_sha256']==next(r['sha256'] for r in writer_test['inputs'] if Path(r['source']).name=='row_pair_column_writer.sv')
program=base/'row-pair-program7';pr=json.loads((program/'result.json').read_text())
assert pr['status']=='ROW_PAIR_FULL_GRAPH_ADDRESS_PASS' and pr['padding_initialization_or_consumer_mask_proved']
assert pr['lifetimes_from_actual_production_order'] and pr['cross_tile_lifetimes']
assert pr['semantic_last_writer_reads']==13058752 and pr['semantic_writes']==9684000
assert sha(program/'program.json')==pr['program_sha256'] and sha(program/'allocations.json')==pr['allocations_sha256']
assert pr['tail_proof_sha256']==sha(base/'row-pair-tail-poison2/result.json')
extended={}
if a.extended:
 dma_map=json.loads((base/'row-pair-dma-map3/result.json').read_text())
 for name in ('row-pair-dma-recycle2','row-pair-dma-recycle-routed1'):
  evidence=json.loads((base/name/'inputs.json').read_text())
  match=next(i for i in evidence['inputs'] if Path(i['source']).name=='row_pair_dma.sv')
  assert sha(match['source'])==match['sha256']==dma_map['source_sha256']
 skip_map=json.loads((base/'row-pair-skip-map1/inputs.json').read_text())
 skip_test=json.loads((base/'row-pair-skip-gate2/inputs.json').read_text())
 for rec in skip_map['inputs']:
  match=next(i for i in skip_test['inputs'] if Path(i['source']).name==Path(rec['source']).name)
  assert sha(rec['source'])==rec['sha256']==match['sha256']
 cmds=json.loads((program/'program.json').read_text());dma=[c for c in cmds if c['kind']!='compute'];skip=[c for c in cmds if c['kind']=='compute' and c['skip']]
 assert len(dma)==10440 and all(c['row']['phase']==0 and c['row']['groups'] in (1,4) for c in dma)
 assert len(skip)==560 and all(len(set(d['bank'] for d in c['skip']))==len(c['skip']) and all(d['phase']==0 for d in c['skip']) for c in skip)
 extended=dict(dma_resources=dma_map['resources'],skip_including_feedback_fifo_resources=json.loads((base/'row-pair-skip-map1/result.json').read_text())['resources'],dma_commands=10440,skip_commands=560,dma_and_skip_sources_bound=True,whole_board_savings_claimed=False)
result=dict(status='ROW_PAIR_SHORT_EVIDENCE_BOUND_NOT_BOARD_PASS',checks=checks,
 connected_local_resources=json.loads((base/'row-pair-subsystem-map1/result.json').read_text())['resources'],
 connected_sources_match_test=4,writer_source_matches_test=True,full_graph_address_checks=pr['read_scalar_addresses_checked'],
 executed_official_ram_cases=9,official_reader_ram3_legacy_pass_label_corrected='frozen final line says144; actual executed9 CASE rows and OFFICIAL_RAM_INTEGRATED cases=9; current TB uses completed_cases',
 extended=extended,
 program=str(program),semantic_last_writer_reads=pr['semantic_last_writer_reads'],
 scope='Bound local gates only; production and whole-board evidence are audited separately.',
 missing=['whole-board capacity fit/STA','gesture/demon/real FPS leaf groups'],
 whole_board_capacity_pass=False,full_frame_30fps=False,script_sha256=sha(__file__))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(result['status'])
