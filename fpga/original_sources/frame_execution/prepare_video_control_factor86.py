"""Create an isolated, exact combinational cone refactor; no tool or board run."""
import argparse,itertools,json,re,shutil
from pathlib import Path
from run_units import ROOT,sha
from native_local_drivers import CELL,graph,render_token
from probe_full_native_split_map import params
from complete_native_logic_split import protected_blocks
from probe_native_open_ports import protected_pins

BASE=ROOT/'tmp/frame-execution/connected-board-video-control84-map1'
SOURCE=ROOT/'tmp/frame-execution/video-control84-input1'
PROPOSAL=ROOT/'tmp/frame-execution/video-control84-canonical1/control-factor86-proposal.json'

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--board',type=Path,required=True);a=p.parse_args()
 out=a.input.resolve();board=a.board.resolve()
 assert all(x.is_relative_to(ROOT/'tmp/frame-execution') and not x.exists() for x in (out,board))
 proposal=json.loads(PROPOSAL.read_text());assert proposal['status']=='CANDIDATE86_COMBINATIONAL_FACTOR_EXHAUSTIVE_PASS_NOT_IMPLEMENTED'
 assert sha(BASE/'outflow/ti60f225_oob.map.v')==proposal['map_sha256']
 text=(SOURCE/'native_source.v').read_text();assert sha(SOURCE/'native_source.v')==sha(BASE/'native_source.v')
 g=graph(text);ps=params(text)
 def check_row(row,cells,pp):
  n=row['name'];assert cells[n][0]=='EFX_LUT4' and pp[n]['LUTMASK']==(16,row['mask'])
  expected={'O':[('net',row['output'],None)]}
  expected.update({f'I{i}':[('net',v,None)] for i,v in enumerate(row['inputs']) if v is not None})
  assert cells[n][1]==expected,(n,cells[n][1],expected)
 for row in proposal['original']+[proposal['shared']]:check_row(row,g,ps)
 changed=set(proposal['removed']+proposal['changed'])
 # Removed STOP25/29 each have exactly one consumer, the replaced mux.
 for n,(_,pins) in g.items():
  for pin,values in pins.items():
   for removed,port in [('SERVICE64_STOP_25','I0'),('SERVICE64_STOP_29','I1')]:
    if ('net',removed.lower(),None) in values:assert (n,pin) in {(removed,'O'),('CONTROL_REMAP_1152',port)}
 def strip_body(body):
  body=CELL.sub(lambda m:'' if m[2].lstrip('\\') in changed else m[0],body)
  return re.sub(r'^\s*defparam\s+(\\\S+|\w+)\s*\.\w+\s*=[^;]+;[^\n]*',lambda m:'' if m[1].lstrip('\\') in changed else m[0],body,flags=re.M)
 body=strip_body(text);rows=proposal['candidate'];insert=[]
 for row in rows:
  if row['output']!='control_remap_1152':insert.append('wire '+row['output']+';')
  pins=[f'.I{i}({render_token(("net",v,None))})' for i,v in enumerate(row['inputs']) if v is not None]
  pins.append('.O('+render_token(('net',row['output'],None))+')')
  insert.append('    EFX_LUT4 '+row['name']+' ('+', '.join(pins)+') /* scoped binary-equivalent control factoring */;')
  insert.append('    defparam '+row['name']+f".LUTMASK=16'd{row['mask']};")
 addition='\n'.join(insert)+'\n'
 anchor=body.index('    protected_cell protected_inst (');body=body[:anchor]+addition+body[anchor:]
 ng=graph(body);np=params(body)
 assert set(g)-set(ng)==set(proposal['removed'])
 added={r['name'] for r in rows}-{'CONTROL_REMAP_1152'};assert set(ng)-set(g)==added
 for row in rows:check_row(row,ng,np)
 for n in set(g)&set(ng)-{'CONTROL_REMAP_1152'}:
  assert g[n]==ng[n] and ps.get(n,{})==np.get(n,{}),n
 assert protected_blocks(text)==protected_blocks(body) and protected_pins(text)==protected_pins(body)
 # Byte-normalized rest includes all old declarations and assignments.
 assert re.sub(r'\s+','',strip_body(text))==re.sub(r'\s+','',body.replace(addition,''))
 def evaluate(records,env):
  env=dict(env)
  for r in [proposal['shared']]+[r for r in proposal['original'] if r['name'] in proposal['retained_cascade']]+records:
   env[r['output']]=(r['mask']>>sum((env[v] if v else 0)<<i for i,v in enumerate(r['inputs'])))&1
  return env['control_remap_1152']
 for bits in itertools.product((0,1),repeat=len(proposal['boundary'])):
  env=dict(zip(proposal['boundary'],bits));assert evaluate(proposal['original'],env)==evaluate(rows,env)
 out.mkdir();board.mkdir()
 for f in SOURCE.iterdir():
  if f.is_file() and f.name!='integration.json':shutil.copy2(f,out/f.name)
 (out/'native_source.v').write_text(body,encoding='utf8')
 old=json.loads((BASE/'inputs.json').read_text());inputs=[]
 for row in old['inputs']:
  src=Path(row['snapshot']);assert sha(src)==row['sha256']
  rel=src.relative_to(BASE);origin=out/rel if (out/rel).is_file() else src
  dst=board/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(origin,dst)
  inputs.append(dict(source=str(origin),snapshot=str(dst),sha256=sha(dst)))
 shutil.copy2(BASE/'map.cmd',board/'map.cmd')
 proof=dict(status='CONTROL_FACTOR86_SOURCE_EQUIVALENCE_PASS_NOT_SYNTHESIZED',
  parent=str(BASE),parent_map_sha256=proposal['map_sha256'],parent_input=str(SOURCE),
  native_sha256=sha(out/'native_source.v'),inputs=inputs,truth_combinations=2**len(proposal['boundary']),
  added_cells=sorted(added),removed_cells=proposal['removed'],changed_cells=['CONTROL_REMAP_1152'],
  all_other_cells_and_parameters_equal=True,protected_blocks_and_pins_equal=True,latency_cycles_added=0,
  evidence={str(f):sha(f) for f in (Path(__file__),PROPOSAL,SOURCE/'integration.json',SOURCE/'native_source.v',BASE/'inputs.json',BASE/'result.json')},
  hardware_access=False,physical_pass=False,board_release=False,
  limits=['Binary combinational proof, not a routing or reset-timing proof.','Existing83/84 evidence needs unchanged-source binding; current86 control, clock/reset and physical gates pending. Source primitive delta is zero.'])
 (out/'integration.json').write_text(json.dumps(proof,indent=2),encoding='utf8')
 (board/'inputs.json').write_text(json.dumps(proof,indent=2),encoding='utf8')
 print(json.dumps({k:proof[k] for k in ('status','native_sha256','truth_combinations','added_cells','removed_cells','changed_cells')}))

if __name__=='__main__':main()
