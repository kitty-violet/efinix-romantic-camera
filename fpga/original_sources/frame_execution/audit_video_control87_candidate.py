"""Audit the actual fallback map before considering any physical run."""
import collections,importlib.util,itertools,json,pickle,re
from pathlib import Path
from run_units import ROOT,sha
from video_control84_physical import audit as audit84,BOARD as OLD
from video_service_physical_profile import periphery_audit
from review_video_sensor73_warnings import warnings
from control86_mapper_crc_equivalence import prove as prove_crc

BOARD=ROOT/'tmp/frame-execution/connected-board-video-control87-map1'
PROPOSAL=ROOT/'tmp/frame-execution/video-control84-canonical1/dma-ce87-proposal.json'

def main():
 inherited=audit84();meta=json.loads((BOARD/'inputs.json').read_text());result=json.loads((BOARD/'result.json').read_text())
 assert meta['status']=='CONTROL87_DMA_CE_SOURCE_EQUIVALENCE_PASS_NOT_SYNTHESIZED'
 assert result['status']=='VIDEO_SERVICE_COMPLETE_NATIVE_SYNTHESIS_PROBE_PASS' and result['inputs_sha256']==sha(BOARD/'inputs.json')
 for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
 for f,h in meta['evidence'].items():assert sha(f)==h
 source=ROOT/'tmp/board-test/video-gain82-offline1/survey-integration.py'
 spec=importlib.util.spec_from_file_location('control87_map_reader',source);reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
 reader.SRC=BOARD/'outflow/ti60f225_oob.map.v';reader.OUT=BOARD
 text,g,ps,drivers,loads=reader.load();h=sha(reader.SRC)
 old=pickle.loads((OLD/'integration-parse.pkl').read_bytes());oh,og,op,od,ol=old;assert oh==inherited['map_sha256']
 proposal=json.loads(PROPOSAL.read_text());boundary=[tuple(x) for x in proposal['boundary']]
 for token in boundary:
  name=od[token];assert drivers[token]==name and g[name]==og[name] and ps.get(name)==op.get(name),('boundary changed',token,name)
 targets=[f'video_processing_pipeline/pipeline/cnn/core/dma/rgb_head[{i}]~FF' for i in range(4)]
 cone=[set(),set()]
 def evaluate(graph,params,dr,token,env,side,active):
  if token in env:return env[token]
  if token[0]=='literal':
   m=re.fullmatch(r"\d+'([bhd])([0-9a-f]+)",token[1],re.I);assert m,token
   v=int(m[2],{'b':2,'h':16,'d':10}[m[1].lower()]);assert v in (0,1);return v
  n=dr[token];assert n not in active and graph[n][0]=='EFX_LUT4',(n,graph[n][0])
  cone[side].add(n);assert len(cone[side])<=128
  pins=graph[n][1];mask=params[n]['LUTMASK'][1];index=0
  for i in range(4):
   if f'I{i}' in pins:index|=evaluate(graph,params,dr,pins[f'I{i}'][0],env,side,active|{n})<<i
   else:assert all(((mask>>v)&1)==((mask>>(v^(1<<i)))&1) for v in range(16)),('unbound used input',n,i)
  return (mask>>index)&1
 for bits in itertools.product((0,1),repeat=6):
  env=dict(zip(boundary,bits))
  for n in targets:
   assert evaluate(og,op,od,og[n][1]['CE'][0],env,0,set())==evaluate(g,ps,drivers,g[n][1]['CE'][0],env,1,set()),(n,bits)
 added=set(g)-set(og);removed=set(og)-set(g)
 assert all(g[n][0]=='EFX_LUT4' for n in added) and all(og[n][0]=='EFX_LUT4' for n in removed),'New state or hard resource'
 changes=[];clockpins=resetpins=0;aliases=[]
 for n,(kind,pins) in g.items():
  if kind=='EFX_LUT4':continue
  assert n in og and kind==og[n][0] and ps.get(n,{})==op.get(n,{}),n
  assert set(pins)==set(og[n][1]),n
  for pin,now in pins.items():
   before=og[n][1][pin]
   if pin in ('CLK','CLKA','CLKB','RCLK','WCLK'):assert now==before;clockpins+=1
   if pin in ('SR','RST','RESET'):assert now==before;resetpins+=1
   if now==before:continue
   row=dict(cell=n,pin=pin,before=before,after=now,classification='pending mapper regrouping review')
   if n in targets and pin=='CE':row['classification']='Actual64-combination CE equation proven'
   elif pin in ('CI','CO') and len(now)==len(before)==1 and od.get(before[0])==drivers.get(now[0]) and sorted(ol.get(before[0],[]))==sorted(loads.get(now[0],[])):
    row['classification']='Exact carry driver and complete load list identical; net alias only';aliases.append(row)
   changes.append(row)
 crc=None
 if any(n.startswith('CutToMuxOpt_') for n in added):crc=prove_crc(old,(h,g,ps,drivers,loads))
 wa,wb=warnings(OLD),warnings(BOARD)
 proof=dict(status='CONTROL87_MAPPED_CONE_AND_CLOCK_CHECK_NOT_PHYSICAL_RELEASE',map_sha256=h,vdb_sha256=sha(BOARD/'outflow/ti60f225_oob.vdb'),
  source_sha256=meta['native_sha256'],actual_ce_boundary_combinations=64,actual_ce_output_pins=4,
  old_cone=sorted(cone[0]),new_cone=sorted(cone[1]),all_boundary_driver_graphs_and_parameters_equal=True,
  new_state_elements=0,all_nonlut_parameters_equal=True,retained_clock_pins=clockpins,retained_reset_pins=resetpins,
  mapped_added=sorted(added),mapped_removed=sorted(removed),net_lut_delta=len(added)-len(removed),
  nonlut_connection_changes=changes,unclassified_nonlut_changes=[r for r in changes if r['classification'].startswith('pending')],
  crc_proof=crc,carry_aliases=aliases,periphery=periphery_audit(BOARD),
  warnings_added=[dict(message=k,count=v) for k,v in (wb-wa).items()],warnings_removed=sum((wa-wb).values()),
  evidence={str(f):sha(f) for f in (Path(__file__),Path(__file__).with_name('control86_mapper_crc_equivalence.py'),source,
   PROPOSAL,BOARD/'inputs.json',BOARD/'result.json',BOARD/'native_source.v',reader.SRC,BOARD/'outflow/ti60f225_oob.vdb',
   BOARD/'outflow/ti60f225_oob.warn.log',OLD/'control84-preplace-audit.json')},
  limits=['Scope is exact unchanged-source binding plus actual6-input CE truth and clock/reset endpoints. No whole-map formal proof.',
   'Mapped regrouping/warnings and mapped interface scope still require review. This script grants no P&R or board release.'],hardware_access=False,board_release=False)
 (BOARD/'control87-candidate-audit.json').write_text(json.dumps(proof,indent=2))
 print(json.dumps({k:proof[k] for k in ('status','map_sha256','net_lut_delta','actual_ce_boundary_combinations','nonlut_connection_changes','warnings_added')},indent=2))
if __name__=='__main__':main()
