"""Map the isolated producer tap scatter before budgeting integration."""
import argparse,collections,json,re,shutil,traceback
from pathlib import Path
from run_units import HERE,ROOT,EF,sha,run
def selection(kind):
 if kind=='rgb_pipeline':
  from pipeline_sources import rgb_pipeline_sources
  return 'cnn_rgb_frame_pipeline',selection('production_frame')[1]+rgb_pipeline_sources()
 if kind=='production_frame':
  return 'cnn_production_frame',selection('datapath')[1]+[HERE/n for n in ('frame_program_fetch.sv','frame_program_issue.sv','axi_read_merge2.sv','cnn_production_frame.sv')]
 if kind=='datapath':
  from validate_fourway import selected
  paths=selected(True,production=True)
  paths=[HERE/'variants/early_request'/f.name if (HERE/'variants/early_request'/f.name).is_file() else f for f in paths]
  paths += [HERE/n for n in ('cnn_frame_datapath.sv','tensor_arena12.sv','resident_weight_bank64.sv','next_header_prefetch.sv','resident_read_arbiter2.sv','arena_client_router2.sv','variants/integrated_datapath/frame_arena_row_dma.sv')]
  paths += [HERE.parent/'full_frame_closure/coefficient_bank_store_fast.sv']
  return 'cnn_frame_datapath',paths
 if kind=='quant':return 'requantize_i32_mul18_q17_pipeline',[HERE/'variants/compact_quant/requantize_i32_mul18_q17_pipeline.sv']
 if kind=='directaccum':return 'int8_dual_mac_lane_prefix_stream',[HERE/'variants/direct_accum/int8_dual_mac_lane_prefix_stream.sv']
 if kind in ('fourway','production','production_srl','production_factored','production_multiissue','production_rotating'):
  from validate_fourway import selected
  paths=selected(True,production=kind!='fourway')
  if kind=='production_srl':paths=[HERE/'variants/srl_history'/f.name if f.name=='unified_column_dispatch32.sv' else f for f in paths]
  if kind in ('production_factored','production_multiissue','production_rotating'):paths=[HERE/'variants/factored_dispatch'/f.name if f.name=='unified_column_dispatch32.sv' else f for f in paths]
  if kind in ('production_multiissue','production_rotating'):paths=[HERE/'variants/multi_issue_reader'/f.name if f.name=='unified_column_reader.sv' else f for f in paths]
  if kind=='production_rotating':paths=[HERE/'variants/rotating_dispatch'/f.name if f.name=='unified_column_dispatch32.sv' else f for f in paths]
  return 'compute_column_cluster',paths
 if kind=='rowdmawide':return 'frame_arena_row_dma',[HERE/'variants/throughput/frame_arena_row_dma.sv']
 if kind=='rowdmaproduction':return 'frame_arena_row_dma',[HERE/'variants/production_banked/frame_arena_row_dma.sv']
 if kind=='residentarb':return 'resident_read_arbiter2',[HERE/'resident_read_arbiter2.sv']
 if kind=='arenarouter':return 'arena_client_router2',[HERE/'arena_client_router2.sv']
 if kind=='scatter4':return 'stride_tap_scatter4',[HERE/'stride_tap_scatter4.sv']
 if kind=='scatter4bank':return 'stride_tap_scatter4',[HERE/'variants/scatter_bank_lock/stride_tap_scatter4.sv']
 if kind=='scatter4compact':return 'stride_tap_scatter4',[HERE/'variants/scatter_compact/stride_tap_scatter4.sv']
 if kind=='stridetile':
  from validate_stride_tile import selected
  return 'stride_tile_cluster',selected()
 if kind=='stridelock':return 'stride_tap_feed24',[HERE/'variants/stride_lockstep/stride_tap_feed24.sv']
 if kind in ('mixed','mixedcompact'):
  from validate_mixed_pool import selected
  return 'shared_pool_banked_fifo',selected(kind=='mixedcompact')
 top={'stride':'stride_tap_feed24','header':'next_header_prefetch','transport':'resident_read_mux2'}[kind]
 return top,[HERE/(top+'.sv')]

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=['speed','area','area2'],default='speed');p.add_argument('--kind',choices=['stride','stridelock','header','transport','mixed','mixedcompact','stridetile','scatter4','scatter4bank','scatter4compact','rowdmawide','rowdmaproduction','residentarb','arenarouter','fourway','production','production_srl','production_factored','production_multiissue','production_rotating','directaccum','quant'],required=True)
 p._option_string_actions['--kind'].choices += ['datapath','production_frame','rgb_pipeline']
 p.add_argument('--retiming',type=int,choices=[0,1,2]);p.add_argument('--operator-sharing',type=int,choices=[0,1]);p.add_argument('--max-threads',type=int,default=2);p.add_argument('--overrides-dir',type=Path);p.add_argument('--frozen-base',type=Path);p.add_argument('--pack-luts-to-comb4',type=int,choices=[0,1,2]);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 try:
  top,paths=selection(a.kind);records=[]
  from frozen_sources import frozen_paths
  paths=frozen_paths(paths,a.frozen_base)
  if a.overrides_dir:
   override=a.overrides_dir.resolve();assert override.is_relative_to(HERE/'variants')
   paths=[override/f.name if (override/f.name).is_file() else f for f in paths]
  for f in paths:
   if f.parent==EF/'sim_models/verilog':continue
   dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  xml=(ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml').read_text().replace('frame_execution_meter',top)
  xml=xml.replace('name="mode" value="speed"','name="mode" value="'+a.mode+'"')
  options={'retiming':a.retiming,'operator-sharing':a.operator_sharing,'max_threads':a.max_threads,'pack-luts-to-comb4':a.pack_luts_to_comb4}
  xml=xml.replace('</efx:synthesis>',''.join('<efx:param name="'+k+'" value="'+str(v)+'" value_type="e_integer" />' for k,v in options.items() if v is not None)+'</efx:synthesis>')
  xml=xml.replace('<efx:design_file name="'+top+'.sv" version="default" library="default" />',
   ''.join('<efx:design_file name="'+Path(r['snapshot']).name+'" version="default" library="default" />' for r in records))
  (root/(top+'.xml')).write_text(xml);(root/(top+'.sdc')).write_text('create_clock -name clk -period 6.6667 [get_ports {clk}]\n')
  (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
  (root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat '+top+'.xml --flow map --timeout 900\nexit /b %errorlevel%\n')
  body,secs=run(['cmd.exe','/d','/c','run.cmd'],root,'map.log',timeout=1800);assert re.search(r'map\s*:\s*PASS',body)
  net=root/('outflow/'+top+'.map.v');rpt=root/('outflow/'+top+'.map.rpt');warn=root/('outflow/'+top+'.warn.log')
  assert 'EFX-0201' not in warn.read_text()
  assert 'EFX-0654' not in warn.read_text(), 'Combinational loops are not admissible'
  counts=dict(collections.Counter(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M)))
  assert (counts.get('DSP24',0),counts.get('DSP48',0))==((0,1) if a.kind=='quant' else (2,0) if a.kind=='directaccum' else (192,24) if a.kind=='stridetile' else (256,32) if a.kind in ('rgb_pipeline','production_frame','datapath','mixed','mixedcompact','fourway','production','production_srl','production_factored','production_multiissue','production_rotating') else (0,0)), 'No extra DSPs allowed'
  for k,v in counts.items():assert re.findall(r'^EFX_'+k+r'\s*:\s*(\d+)\s*$',rpt.read_text(),re.M)==[str(v)]
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='ISOLATED_PARALLEL_CANDIDATE_MAP_REQUIRES_INTEGRATION',kind=a.kind,mode=a.mode,options=options,project_sha256=sha(root/(top+'.xml')),resources=counts,seconds=secs,inputs_sha256=sha(root/'inputs.json'),
   map_sha256=sha(net),warning_sha256=sha(warn),report_sha256=sha(rpt),producer_rq_connected=a.kind in ('production_frame','datapath','fourway','production','production_srl','production_factored','production_multiissue','production_rotating'),arena_external=a.kind not in ('datapath','production_frame'),full_frame=False,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
