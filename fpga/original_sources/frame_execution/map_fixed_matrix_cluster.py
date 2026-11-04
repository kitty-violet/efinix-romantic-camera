"""Map the complete candidate feed/MAC/partial/RQ path in one netlist."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,run,sha
from fixed_variant_overlay import resolve_overlay
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--production',action='store_true');p.add_argument('--video',action='store_true');p.add_argument('--cluster',action='store_true');p.add_argument('--dsp-profile',choices=('original','balanced'),default='original');a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
import os
arithmetic_override=os.environ.get('FPGA_FIXED_ARITHMETIC')
v=HERE/'variants/fixed_window_dot8x32'
names=('fixed_matrix_cluster','fixed_matrix_feed6','fixed_window_dot8x32','fixed_ring_weights9','fixed_partial32','fixed_rq32')
if a.production:names=('fixed_production_matrix','fixed_source_columns6','fixed_window_dot8x32','fixed_ring_weights9','fixed_partial32','fixed_rq32')
if a.video or a.cluster:names=('fixed_video_compute','fixed_production_mac','fixed_source_columns6','fixed_window_dot8x32','fixed_ring_weights9','fixed_partial32','fixed_rq32_mixed','fixed_skip_vectors4','fixed_column_store4')
if a.cluster:names=names+('fixed_weight_loader',)
files=[v/(n+'.sv') for n in names]+[ROOT/'tmp/frame-execution/row-pair-taps-fixed-production-short1/source/requantize_i32_mul18_q17_pipeline.sv']
if a.video or a.cluster:files+=[HERE/'variants/row_pair_taps_fixed/row_pair_aux_router.sv']
if a.cluster:files=[HERE/'variants/fixed_production_cluster'/n for n in ('fixed_production_cluster_map.sv','row_pair_compute_cluster.sv')]+files
if arithmetic_override:files=[Path(arithmetic_override).resolve() if f.name=='fixed_window_dot8x32.sv' else f for f in files]
if os.environ.get('FPGA_FIXED_OVERLAY'):
 overlay=resolve_overlay(os.environ['FPGA_FIXED_OVERLAY'])
 # Only these known outer-pipeline files are absent from a compute-only map.
 # They still require the production runner and whole-board proof before use.
 outer={'coefficient_bank_store_fast.sv','cnn_frame_datapath.sv','cnn_production_frame.sv','row_pair_dma.sv','frame_program_fetch.sv','frame_program_issue.sv','row_pair_arena6.sv'}
 assert overlay and set(overlay)<=set(f.name for f in files)|outer
 files=[overlay.get(f.name,f) for f in files]
records=[]
for f in files:
 d=r/f.name
 if a.cluster and f.name=='fixed_production_cluster_map.sv' and 'NEXT_WEIGHT_PREFETCH' in next(x for x in files if x.name=='row_pair_compute_cluster.sv').read_text():
  wrapper=f.read_text().replace('input wire clk,reset,cancel,','input wire clk,reset,cancel,\n input wire next_weight_valid,next_weight_pw,next_weight_stem,\n input wire [2:0] next_weight_groups,input wire [14:0] next_weight_offset,',1)
  wrapper=wrapper.replace('#(.EXTERNAL_HEADER(1))','#(.EXTERNAL_HEADER(1),.NEXT_WEIGHT_PREFETCH(1))',1)
  if 'aux_write_collision' in next(x for x in files if x.name=='row_pair_compute_cluster.sv').read_text():
   wrapper=wrapper.replace('input wire clk,reset,cancel,','input wire clk,reset,cancel,\n output wire [5:0] aux_write_collision,',1)
  d.write_text(wrapper)
 else:shutil.copy2(f,d)
 records.append(dict(source=str(f),snapshot=str(d),sha256=sha(d),source_sha256=sha(f)))
top=files[0].stem
project=(ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml').read_text().replace('frame_execution_meter',top)
old=f'<efx:design_file name="{top}.sv" version="default" library="default" />'
project=project.replace(old,'\n'.join(f'<efx:design_file name="{f.name}" version="default" library="default" />' for f in files))
mode=os.environ.get('FPGA_FIXED_MAP_MODE','speed');assert mode in ('speed','area','area2','cong')
project=project.replace('name="mode" value="speed"',f'name="mode" value="{mode}"')
(r/(top+'.xml')).write_text(project)
(r/(top+'.sdc')).write_text('create_clock -name clk -period 6.667 [get_ports {clk}]\n')
(r/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat '+top+'.xml --flow map --timeout 180\nexit /b %errorlevel%\n')
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__),project_sha256=sha(r/(top+'.xml'))),indent=2)+'\n')
body,secs=run(['cmd.exe','/d','/c','run.cmd'],r,'map.log',timeout=210)
assert re.search(r'map\s*:\s*PASS',body),body[-2000:]
rpt=r/'outflow'/(top+'.map.rpt');warn=r/'outflow'/(top+'.warn.log')
counts={k:int(v) for k,v in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',rpt.read_text(),re.M)}
assert 'EFX-0654' not in warn.read_text(errors='replace')
for x in records:assert sha(x['source'])==x['source_sha256'] and sha(x['snapshot'])==x['sha256']
expected_dsp=(288,16) if a.dsp_profile=='balanced' else (256,32)
assert (counts.get('DSP24'),counts.get('DSP48'))==expected_dsp,counts
assert counts['DSP24']//2+counts['DSP48']==160
result=dict(status='COMBINED_FEED_MATRIX_RQ_MAP_PASS_NOT_BOARD',resources=counts,seconds=secs,inputs_sha256=sha(r/'inputs.json'),report_sha256=sha(rpt),warning_sha256=sha(warn),full_cnn=False,external_ram=True,production_coordinates=a.production or a.video or a.cluster,residual_and_writeback=a.video or a.cluster,production_adapter=a.cluster,production_placement=False)
(r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
