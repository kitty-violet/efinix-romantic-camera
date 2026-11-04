"""Short four-state ingress/capture gates using frozen production dependencies."""
import argparse,json,re,shutil,traceback,os
from pathlib import Path
from run_units import ROOT,HERE,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--kind',choices=['writer','ingress','frontend'],required=True);p.add_argument('--packer',type=Path);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
(r/'host-temp').mkdir();os.environ['TEMP']=os.environ['TMP']=str(r/'host-temp')
base=ROOT/'tmp/frame-execution/connected-board-fixed-split-bias1'
v=HERE/'variants/raw_stream';tb=HERE/'variants/raw_stream_test'
if a.kind=='writer':
 top='tb_rgb720_stream_writer';prefix='RGB_STREAM_WRITER'
 files=[base/'assembly/rgb24_stream_pack.sv',base/'assembly/axi_frame_write_dma.sv',v/'rgb720_stream_writer.sv',tb/(top+'.sv')]
 cases=[['-gBURST=1'],['-gBURST=4']]
elif a.kind=='ingress':
 top='tb_raw_stream_ingress';prefix='RAW_STREAM_INGRESS'
 files=[base/'rtl/frame_buffer_V3/DC_FIFO.v',base/'assembly/guarded_async_fifo.sv',base/'assembly/raw_axi_beat_unpacker.sv',base/'assembly/raw_line_bank_ram.sv',base/'assembly/raw_line_feeder.sv',v/'raw32_validated_packer.sv',v/'raw_stream_ingress.sv',tb/(top+'.sv')]
 cases=[['-gSEED=71'],['-gSEED=391'],['-gSEED=1027']]
else:
 top='tb_raw_stream_frontend';prefix='RAW_STREAM_FRONTEND'
 names=['guarded_async_fifo.sv','raw_axi_beat_unpacker.sv','raw_line_bank_ram.sv','raw_line_feeder.sv','legacy_debayer_stream_wrapper.sv','rgb_pair_cdc_serializer.sv','rgb24_stream_pack.sv','axi_frame_write_dma.sv','frame_pair_owner.sv','rgb_area_downscaler_3to2.sv','rgb_gain_mul_u8x16_logic.sv']
 all_files={f.name:f for f in base.rglob('*') if f.suffix in ('.sv','.v') and 'outflow' not in f.parts and 'work_syn' not in f.parts}
 files=[base/'rtl/frame_buffer_V3/DC_FIFO.v',*[all_files[n] for n in names],*[base/'rtl/debayer'/n for n in ['debayer_top_2to1.v','rgb_gain_v1.v','line_buffer.v','true_dual_port_ram.v','raw_to_rgb.v']],v/'raw32_validated_packer.sv',v/'raw_stream_ingress.sv',v/'raw_stream_frontend.sv',v/'rgb720_stream_writer.sv',tb/(top+'.sv')]
 cases=[['-gSEED=71'],['-gSEED=391'],['-gSEED=1027']]
if a.packer:
 assert a.kind!='writer' and a.packer.resolve().is_relative_to(HERE/'variants') and a.packer.name=='raw32_validated_packer.sv'
 assert sum(f.name==a.packer.name for f in files)==1
 files=[a.packer.resolve() if f.name==a.packer.name else f for f in files]
rows=[]
for f in files:
 d=r/f.name;shutil.copy2(f,d);rows.append(dict(source=str(f),snapshot=str(d),sha256=sha(d)))
(r/'inputs.json').write_text(json.dumps(dict(inputs=rows,runner_sha256=sha(__file__),cases=cases),indent=2)+'\n')
try:
 run([MS/'vlib.exe','work'],r,'vlib.log')
 b,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[x['snapshot'] for x in rows]],r,'compile.log');assert 'Errors: 0, Warnings: 0' in b
 results=[]
 for i,opts in enumerate(cases):
  name=f'simulation{i}.log'
  b,t=run([MS/'vsim.exe','-c','work.'+top,*opts,'-do','onerror {quit -code 1}; run -all; quit -code 0'],r,name,timeout=45)
  lines=re.findall(r'^# (PASS '+prefix+r'.*)$',b,re.M);assert len(lines)==1
  warnings=re.findall(r'^# \*\* Warning: (.*)$',b,re.M)
  if a.kind=='frontend':
   # Existing frozen modules leave only two diagnostic outputs unconnected.
   # Each instance produces one count warning and one named-port warning.
   allowed=[r'\(vsim-3017\) .*/debayer_top_2to1.v\(76\): \[TFMPC\] - Too few port connections\. Expected 15, found 14\.',r'\(vsim-3017\) .*/rgb_pair_cdc_serializer.sv\(165\): \[TFMPC\] - Too few port connections\. Expected 15, found 14\.',r"\(vsim-3722\) .*/debayer_top_2to1.v\(76\): \[TFMPC\] - Missing connection for port 'o_x'\.",r"\(vsim-3722\) .*/rgb_pair_cdc_serializer.sv\(165\): \[TFMPC\] - Missing connection for port 'rd_available'\."]
   assert len(warnings)==8 and all(any(re.fullmatch(pat,w) for pat in allowed) for w in warnings),warnings
  else:assert not warnings,warnings
  assert f'Errors: 0, Warnings: {len(warnings)}' in b
  results.append(dict(line=lines[0],seconds=t,log_sha256=sha(r/name),known_unused_output_warnings=len(warnings)))
 for x in rows:assert sha(x['source'])==sha(x['snapshot'])==x['sha256']
 result=dict(status=prefix+'_PASS',results=results,inputs_sha256=sha(r/'inputs.json'),whole_camera_chain_proved=False)
 (r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
except Exception as e:
 (r/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
