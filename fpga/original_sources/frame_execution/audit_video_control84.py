"""Bind the exact84 mapped combinational change to83 functional evidence."""
import collections,importlib.util,itertools,json,pickle,re
from pathlib import Path
from run_units import ROOT,sha
from video_bandwidth83_physical import audit as audit83,BOARD as OLD
from review_video_sensor73_warnings import warnings
from video_service_physical_profile import periphery_audit
from prepare_video_control_factor84 import PROPOSAL

BOARD=ROOT/'tmp/frame-execution/connected-board-video-control84-map1'
def main():
 inherited=audit83();b=BOARD;meta=json.loads((b/'inputs.json').read_text());res=json.loads((b/'result.json').read_text())
 assert meta['status']=='CONTROL_FACTOR84_SOURCE_EQUIVALENCE_PASS_NOT_SYNTHESIZED'
 assert res['status']=='VIDEO_SERVICE_COMPLETE_NATIVE_SYNTHESIS_PROBE_PASS' and res['inputs_sha256']==sha(b/'inputs.json')
 for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
 for f,h in meta['evidence'].items():assert sha(f)==h
 source=ROOT/'tmp/board-test/video-gain82-offline1/survey-integration.py'
 spec=importlib.util.spec_from_file_location('control84_map_reader',source);reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
 reader.SRC=b/'outflow/ti60f225_oob.map.v';reader.OUT=b
 text,g,ps,drivers,loads=reader.load();h=sha(reader.SRC)
 oh,og,op,_,_=pickle.loads((OLD/'integration-parse.pkl').read_bytes());assert oh==inherited['map_sha256']
 proposal=json.loads(PROPOSAL.read_text());added=set(meta['added_cells']);removed=set(meta['removed_cells']);changed=set(meta['changed_cells'])
 mapped_added=set(g)-set(og);mapped_removed=set(og)-set(g)
 assert added<=mapped_added and removed<=mapped_removed
 assert all(n.startswith('LUT__') and g[n][0]=='EFX_LUT4' for n in mapped_added-added)
 assert all(n.startswith('LUT__') and og[n][0]=='EFX_LUT4' for n in mapped_removed-removed)
 assert {n for n,(k,_) in g.items() if k!='EFX_LUT4'}=={n for n,(k,_) in og.items() if k!='EFX_LUT4'}
 for row in proposal['candidate']:
  n=row['name'];expected={'O':[('net',row['output'],None)]};expected.update({f'I{i}':[('net',v,None)] for i,v in enumerate(row['inputs']) if v is not None})
  assert g[n]==('EFX_LUT4',expected) and ps[n]['LUTMASK']==(16,row['mask']),n
 def eval_graph(graph,pp,names,env):
  env=dict(env)
  for n in names:
   pins=graph[n][1];index=sum(env[pins[f'I{i}'][0][1]]<<i for i in range(4) if f'I{i}' in pins)
   env[pins['O'][0][1]]=(pp[n]['LUTMASK'][1]>>index)&1
  return env['service63_ddr_33']
 for bits in itertools.product((0,1),repeat=len(proposal['boundary'])):
  env=dict(zip(proposal['boundary'],bits))
  assert eval_graph(og,op,[r['name'] for r in proposal['original']],env)==eval_graph(g,ps,[r['name'] for r in proposal['candidate']],env)
 clocks=0;reset_pins=0;nonlut_connections=[]
 for n,(k,pins) in g.items():
  if k=='EFX_LUT4':continue
  assert n in og and k==og[n][0] and ps.get(n,{})==op.get(n,{}),n
  assert set(pins)==set(og[n][1]),n
  for pin,values in pins.items():
   prior=og[n][1][pin]
   if values==prior:continue
   assert pin in ('D','CE','WDATA','I0','CI','CO'),(n,pin)
   assert n.startswith(('service_isp','service_page_part[','service_clear_toggle','service_legacy_epoch[','service_health_page[','service_serializer/storage/','service_boot/sub_','service_health/sub_','bandwidth83_capture/capture/')),(n,pin)
   assert len(values)==len(prior),(n,pin)
   nonlut_connections.append(dict(cell=n,pin=pin,before=prior,after=values,classification='Same source/primitive parameters; mapped aliases or CE/data optimization. No mapped formal-equivalence claim.'))
  for pin in ('CLK','CLKA','CLKB','RCLK','WCLK','SR','RST','RESET'):
   if pin in pins:assert pins[pin]==og[n][1][pin],(n,pin)
  clocks+=sum(pin in pins for pin in ('CLK','CLKA','CLKB','RCLK','WCLK'))
  reset_pins+=sum(pin in pins for pin in ('SR','RST','RESET'))
 # No new CDC sources or storage. The factored combinational output retains
 # its function and consumers, including downstream synchronous reset cones.
 assert loads[('net','service63_ddr_33',None)]==pickle.loads((OLD/'integration-parse.pkl').read_bytes())[4][('net','service63_ddr_33',None)]
 delta=periphery_audit(b);wa,wb=warnings(OLD),warnings(b);extra=[]
 for msg,count in (wb-wa).items():
  if msg in ("[EFX-0011 VERI-WARNING] port 'I3' is not connected on this instance","[EFX-0011 VERI-WARNING] input port 'I3' is not connected on this instance"):
   assert count==1
   for row in proposal['candidate']:
    for bit,v in enumerate(row['inputs']):
     if v is None:assert all(((row['mask']>>x)&1)==((row['mask']>>(x^(1<<bit)))&1) for x in range(16))
   extra.append(dict(message=msg,count=count,classification='Two new LUTs omit proven-unused I3; one old such LUT removed. All16 truth entries independent of missing inputs.'));continue
  assert msg.startswith('[EFX-0200 WARNING] Removing redundant signal : '),msg
  assert 'service63_ddr_29' in msg or 'u_ddr3_top/' in msg or 'inst_efx_csi2_rx/' in msg or 'init_cur_state[6:2]' in msg,msg
  extra.append(dict(message=msg,count=count,classification='Deleted combinational intermediate or unchanged vendor unused alias; full mapped graph difference checked.'))
 anchors=ROOT/'tmp/frame-execution/video-service-place50-input-local/fixed_dsp.placeloc'
 ns=[line.split()[0] for line in anchors.read_text().splitlines() if not line.startswith('#') and len(line.split())==4]
 assert len(ns)==404 and all(n in g and g[n][0] in ('EFX_DSP24','EFX_DSP48','EFX_RAM10','EFX_DPRAM10') for n in ns)
 proofs=[OLD/'bandwidth83-preplace-audit.json',OLD/'bandwidth83-cdc-delta.json',OLD/'service-cdc-inventory-expanded.json',
  ROOT/'tmp/frame-execution/video-bandwidth83-pin2/result.json',ROOT/'tmp/board-test/video-gain82-offline1/joint-full1080-83-1/result.json']
 files=[Path(__file__),source,PROPOSAL,b/'inputs.json',b/'result.json',b/'native_source.v',reader.SRC,b/'outflow/ti60f225_oob.vdb',b/'outflow/ti60f225_oob.warn.log',OLD/'outflow/ti60f225_oob.warn.log',anchors,*proofs]
 proof=dict(status='CONTROL84_SOURCE_CLOCK_TESTS_PREPLACE_PASS',map_sha256=h,vdb_sha256=sha(b/'outflow/ti60f225_oob.vdb'),
  current_mapped_truth_combinations=512,current_mapped_added_lut=sorted(added),current_mapped_removed_lut=sorted(removed),current_mapped_changed_lut=sorted(changed),
  all_other_native_source_equal=True,all_nonlut_parameters_equal=True,all_other_mapped_cells_and_parameters_equal=False,retained_clock_pins=clocks,retained_reset_pins=reset_pins,new_state_elements=0,new_cross_clock_paths=0,
  mapped_lut_net_delta=len(mapped_added)-len(mapped_removed),mapped_added=sorted(mapped_added),mapped_removed=sorted(mapped_removed),mapped_nonlut_connection_changes=nonlut_connections,
  inherited_tests_scope='83 capture32/split-DDR-RAM source and all other source modules are byte-equivalent outside the four-cell control factor.83 RTL numeric/protocol/concurrency tests apply to that unchanged source; no claim all mapped cells identical. Current84 factored cone checked for all512 combinations. Fresh84 actual mapped camera pin test is required before release.',
  periphery=delta,hard_anchors_unchanged=404,ordinary_anchors=0,warnings_added=extra,warnings_removed=sum((wa-wb).values()),unclassified_warnings=[],
  evidence={str(f):sha(f) for f in files},hardware_access=False,board_release=False,
  limits=['Changed open control cone is binary combinational-equivalent; protected blocks are not whole-system formal equivalence.','Automatic mapper regrouping exists; attempted global public-cone BDD expansion exceeded its1M-node limit and gives no PASS. Source-scope tests plus exact512 mapped control combinations, clock/reset checks and fresh mapped interface tests form the segmented evidence.','Actual84 resource/P&R/STA, CDC/Gray and routed reset/PLL timing are pending.','15fps, RAW stability and SW1 recovery require board evidence.'])
 (b/'control84-preplace-audit.json').write_text(json.dumps(proof,indent=2),encoding='utf8')
 inv=json.loads((OLD/'service-cdc-inventory-expanded.json').read_text());assert inv['map_sha256']==oh and not inv['unclassified']
 inv['map_sha256']=h;inv['classification_evidence'].update({str(f):sha(f) for f in (Path(__file__),b/'control84-preplace-audit.json',OLD/'service-cdc-inventory-expanded.json')})
 inv['control84_reuse_scope']='All mapped storage/clock/reset pins and parameters unchanged. Only the9-input512-combination control cone is factored without extra state. Fresh routed CDC/Gray/PLL required.'
 (b/'service-cdc-inventory-expanded.json').write_text(json.dumps(inv,indent=2))
 cdc=dict(status='CONTROL84_CDC_DELTA_CLASSIFIED_NOT_ROUTED_PASS',map_sha256=h,retained_clock_pins_checked=clocks,retained_reset_pins_checked=reset_pins,new_storage_clock_pins=0,new_cross_domain_cones=0,
  current_inventory_sha256=sha(b/'service-cdc-inventory-expanded.json'),evidence={str(f):sha(f) for f in (Path(__file__),b/'control84-preplace-audit.json',OLD/'service-cdc-inventory-expanded.json')},hardware_access=False,board_release=False)
 (b/'control84-cdc-delta.json').write_text(json.dumps(cdc,indent=2))
 print(json.dumps({k:proof[k] for k in ('status','map_sha256','retained_clock_pins','retained_reset_pins','warnings_added')}))
if __name__=='__main__':main()
