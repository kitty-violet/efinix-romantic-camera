"""Official primitive short gates for natural-order production candidates."""
import argparse,json,os,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,run,sha
from fixed_variant_overlay import resolve_overlay
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--kind',choices=['source','production','residual','video','weights'],required=True);p.add_argument('--budget',type=int,default=240);p.add_argument('--source',type=Path);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
v=HERE/'variants/fixed_window_dot8x32'
names=['fixed_source_columns6'];models=['efx_dpram10.v']
top='tb_fixed_source_columns6';marker='FIXED_SOURCE_COLUMNS6'
if a.kind=='production':
 names+=['fixed_production_matrix','fixed_window_dot8x32','fixed_ring_weights9','fixed_partial32','fixed_rq32']
 models+=['efx_dsp24.v','efx_dsp48.v','efx_srl8.v'];top='tb_fixed_production_matrix';marker='FIXED_PRODUCTION_MATRIX'
if a.kind=='residual':
 names=['fixed_rq32_mixed','fixed_skip_vectors4','fixed_column_store4'];models+=['efx_dsp48.v','efx_srl8.v'];top='tb_fixed_residual_store';marker='FIXED_RESIDUAL_STORE'
if a.kind=='video':
 names=['fixed_video_compute','fixed_production_mac','fixed_source_columns6','fixed_window_dot8x32','fixed_ring_weights9','fixed_partial32','fixed_rq32_mixed','fixed_skip_vectors4','fixed_column_store4']
 models+=['efx_dsp24.v','efx_dsp48.v','efx_srl8.v'];top='tb_fixed_video_compute';marker='FIXED_VIDEO_COMPUTE'
if a.kind=='weights':
 names=['fixed_weight_loader','fixed_ring_weights9'];models=['efx_srl8.v'];top='tb_fixed_weight_loader';marker='FIXED_WEIGHT_LOADER'
files=[v/(n+'.sv') for n in names+[top]]+[HERE/'variants/row_pair_arena/row_pair_arena6.sv']+[EF/'sim_models/verilog'/n for n in models]
if a.source:
 candidate=a.source.resolve();assert a.kind=='source' and candidate.name=='fixed_source_columns6.sv' and candidate.is_relative_to(HERE/'variants')
 files=[candidate if f.name==candidate.name else f for f in files]
 if 'EFX_SRL8' in candidate.read_text() and not any(f.name=='efx_srl8.v' for f in files):
  files.append(EF/'sim_models/verilog/efx_srl8.v')
if os.environ.get('FPGA_FIXED_TESTBENCH'):
 tb=Path(os.environ['FPGA_FIXED_TESTBENCH']).resolve();assert tb.name==top+'.sv' and tb.is_relative_to(ROOT)
 files=[tb if f.name==tb.name else f for f in files]
if a.kind in ('production','residual','video'):files+=[ROOT/'tmp/frame-execution/row-pair-taps-fixed-production-short1/source/requantize_i32_mul18_q17_pipeline.sv']
if a.kind in ('residual','video'):files+=[HERE/'variants/row_pair_taps_fixed/row_pair_aux_router.sv']
if os.environ.get('FPGA_FIXED_OVERLAY'):
 overlay=resolve_overlay(os.environ['FPGA_FIXED_OVERLAY'])
 if a.kind=='video':
  # A full production suite also declares these unused outer modules. Compile
  # them for interface syntax, while the selected video test remains the top.
  present={f.name for f in files}
  files += [overlay[n] for n in ('row_pair_compute_cluster.sv','fixed_weight_loader.sv') if n in overlay and n not in present]
 assert overlay and set(overlay)<=set(f.name for f in files)
 files=[overlay.get(f.name,f) for f in files]
records=[]
for f in files:
 d=r/f.name;shutil.copy2(f,d);records.append(dict(source=str(f),snapshot=str(d),sha256=sha(f)))
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2)+'\n')
run([MS/'vlib.exe','work'],r,'vlib.log');body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[x['snapshot'] for x in records]],r,'compile.log');assert 'Errors: 0, Warnings: 0' in body
scratch=r/'temp';scratch.mkdir();env=os.environ.copy();env.update(TEMP=str(scratch),TMP=str(scratch),TMPDIR=str(scratch))
t=time.monotonic();log=r/'simulation.log'
with log.open('w',encoding='utf-8') as f:
 proc=subprocess.run(list(map(str,[MS/'vsim.exe','-c','work.'+top,'-do','onerror {quit -code 1}; run -all; quit -code 0'])),cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=a.budget)
body=log.read_text();assert proc.returncode==0 and not re.search(r'(?m)^#?\s*(FAIL|\*\* (Error|Fatal))',body),body[-2500:]
lines=re.findall(r'^# (PASS '+marker+r'.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
for x in records:assert sha(x['source'])==sha(x['snapshot'])==x['sha256']
result=dict(status=marker+'_SHORT_PASS',line=lines[0],seconds=time.monotonic()-t,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(log),full_cnn=False)
(r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
