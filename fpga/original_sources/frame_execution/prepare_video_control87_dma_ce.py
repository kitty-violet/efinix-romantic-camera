"""Isolated fallback source: one equivalent DMA-CE cone, no active-run edits."""
import json,re,shutil
from pathlib import Path
from run_units import ROOT,sha
from native_local_drivers import CELL,graph,render_token
from probe_full_native_split_map import params
from complete_native_logic_split import protected_blocks
from probe_native_open_ports import protected_pins

B=ROOT/'tmp/frame-execution'
BASE=B/'connected-board-video-control84-map1'
SOURCE=B/'video-control84-input1'
DEST=B/'video-control87-input1'
BOARD=B/'connected-board-video-control87-map1'
PROPOSAL=B/'video-control84-canonical1/dma-ce87-proposal.json'

def main():
 assert not DEST.exists() and not BOARD.exists()
 proposal=json.loads(PROPOSAL.read_text());assert proposal['status']=='DMA_CE87_BINARY_PROPOSAL_64_PASS_NOT_IMPLEMENTED'
 assert sha(BASE/'outflow/ti60f225_oob.map.v')==proposal['map_sha256']
 for f,h in proposal['evidence'].items():assert sha(f)==h
 text=(SOURCE/'native_source.v').read_text();assert sha(SOURCE/'native_source.v')==sha(BASE/'native_source.v')
 g=graph(text);ps=params(text)
 def pins(row):
  p={f'I{i}':[tuple(t)] for i,t in enumerate(row['inputs']) if t is not None};p['O']=[tuple(row['output'])];return p
 for row in proposal['original']:assert g[row['name']]==('EFX_LUT4',pins(row)) and ps[row['name']]['LUTMASK']==(16,row['mask'])
 changed=set(proposal['changed']);assert changed=={'LUT__146659'}
 def stripped(t):
  t=CELL.sub(lambda m:'' if m[2].lstrip('\\') in changed else m[0],t)
  return re.sub(r'^\s*defparam\s+(\\\S+|\w+)\s*\.\w+\s*=[^;]+;[^\n]*',lambda m:'' if m[1].lstrip('\\') in changed else m[0],t,flags=re.M)
 body=stripped(text);addition=[]
 for row in proposal['candidate']:
  if row['name'] not in changed:addition.append('wire '+render_token(tuple(row['output']))+';')
  port=[f'.I{i}({render_token(tuple(t))})' for i,t in enumerate(row['inputs']) if t is not None]
  port.append('.O('+render_token(tuple(row['output']))+')')
  addition.append('    EFX_LUT4 '+row['name']+' ('+', '.join(port)+') /* exact DMA CE late-bank factor */;')
  addition.append('    defparam '+row['name']+f".LUTMASK = 16'd{row['mask']};")
 addition='\n'.join(addition)+'\n';at=body.index('    protected_cell protected_inst (');body=body[:at]+addition+body[at:]
 ng=graph(body);np=params(body);assert not set(g)-set(ng) and set(ng)-set(g)==set(proposal['added'])
 for n in set(g)-changed:assert g[n]==ng[n] and ps.get(n,{})==np.get(n,{}),n
 for row in proposal['candidate']:assert ng[row['name']]==('EFX_LUT4',pins(row)) and np[row['name']]['LUTMASK']==(16,row['mask'])
 assert protected_blocks(text)==protected_blocks(body) and protected_pins(text)==protected_pins(body)
 assert re.sub(r'\s+','',stripped(text))==re.sub(r'\s+','',body.replace(addition,''))
 DEST.mkdir();BOARD.mkdir()
 for f in SOURCE.iterdir():
  if f.is_file() and f.name!='integration.json':shutil.copy2(f,DEST/f.name)
 (DEST/'native_source.v').write_text(body,encoding='utf8')
 old=json.loads((BASE/'inputs.json').read_text());inputs=[]
 for row in old['inputs']:
  src=Path(row['snapshot']);assert sha(src)==row['sha256']
  rel=src.relative_to(BASE);origin=DEST/rel if (DEST/rel).is_file() else src;dst=BOARD/rel
  dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(origin,dst)
  inputs.append(dict(source=str(origin),snapshot=str(dst),sha256=sha(dst)))
 shutil.copy2(BASE/'map.cmd',BOARD/'map.cmd')
 result=dict(status='CONTROL87_DMA_CE_SOURCE_EQUIVALENCE_PASS_NOT_SYNTHESIZED',parent=str(BASE),parent_input=str(SOURCE),
  native_sha256=sha(DEST/'native_source.v'),inputs=inputs,truth_combinations=64,
  added_cells=proposal['added'],removed_cells=[],changed_cells=proposal['changed'],
  all_other_cells_and_parameters_equal=True,protected_blocks_and_pins_equal=True,latency_cycles_added=0,
  source_primitive_delta=2,active84_run_untouched=True,
  evidence={str(f):sha(f) for f in (Path(__file__),PROPOSAL,SOURCE/'integration.json',SOURCE/'native_source.v',BASE/'inputs.json',BASE/'result.json')},
  hardware_access=False,physical_pass=False,board_release=False,
  limits=['Only a six-boundary Boolean proof and exact unchanged native source. Actual mapped cones/clock/reset/physical gates not yet checked.',
   'This is an isolated fallback precheck, not adopted into the active84 physical run.'])
 for f in (DEST/'integration.json',BOARD/'inputs.json'):f.write_text(json.dumps(result,indent=2),encoding='utf8')
 print(json.dumps({k:result[k] for k in ('status','native_sha256','truth_combinations','added_cells','changed_cells')}),flush=True)
if __name__=='__main__':main()
