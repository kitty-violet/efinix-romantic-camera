"""Map 43-bit reference and 25-bit framed FIFO, both depth4096."""
import argparse,collections,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,EF,sha,run
from validate_compact_video import selected,VIDEO
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',action='store_true');p.add_argument('--pipeline',action='store_true');p.add_argument('--area',action='store_true');a=p.parse_args();assert not a.area or a.pipeline
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 try:
  records=[];top='guarded_async_fifo' if a.reference else 'framed_rgb_async_fifo'
  paths=selected()[:2] if a.reference else selected()
  if a.pipeline:
   top='rgb_scaler_display_pipeline'
   paths += [VIDEO/n for n in ('rgb_bilinear_scaler_prototype.sv','resize_axis_down_3_to_2.sv','resize_axis_up_2_to_3.sv','hdmi_frame_scheduler.sv')]+[VIDEO/('' if a.reference else 'variants/compact_fifo/')/'rgb_scaler_display_pipeline.sv']
   if a.area:
    paths[-1]=ROOT/'tmp/efinity-smoke-2026.1/sc431-v23-fifo2026-seed1-effort5/rtl/rgb_scaler_display_pipeline.sv' if a.reference else VIDEO/'variants/compact_area_fifo/rgb_scaler_display_pipeline.sv'
    paths.append(VIDEO/'rgb_area_downscaler_3to2.sv')
  for f in paths:
   dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  # Reference wrapper avoids changing the frozen guarded FIFO default parameters.
  if a.reference and not a.pipeline:
   top='video_fifo_reference'
   wrapper='module video_fifo_reference(input wire reset,wr_clk,rd_clk,s_valid,m_ready,input wire [42:0] s_data,output wire s_ready,m_valid,output wire [42:0] m_data,output wire [12:0] wr_count,rd_count,output wire wr_guard_ready,rd_guard_ready,protocol_error);\n guarded_async_fifo #(.DATA_WIDTH(43),.FIFO_DEPTH(4096)) q(.*);\nendmodule\n'
   (root/(top+'.sv')).write_text(wrapper)
  names=[Path(r['snapshot']).name for r in records]+([top+'.sv'] if a.reference and not a.pipeline else [])
  xml=(ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml').read_text().replace('frame_execution_meter',top)
  xml=xml.replace('<efx:design_file name="'+top+'.sv" version="default" library="default" />',''.join('<efx:design_file name="'+n+'" version="default" library="default" />' for n in names))
  wrclk='process_clk' if a.pipeline else 'wr_clk';rdclk='display_clk' if a.pipeline else 'rd_clk'
  (root/(top+'.xml')).write_text(xml);(root/(top+'.sdc')).write_text(f'create_clock -name {wrclk} -period 6.6667 [get_ports {{{wrclk}}}]\ncreate_clock -name {rdclk} -period 13.468 [get_ports {{{rdclk}}}]\n')
  (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__),reference=a.reference,generated={n:sha(root/n) for n in names if n not in [Path(r['snapshot']).name for r in records]}),indent=2))
  (root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat '+top+'.xml --flow map --timeout 900\nexit /b %errorlevel%\n')
  body,secs=run(['cmd.exe','/d','/c','run.cmd'],root,'map.log');assert re.search(r'map\s*:\s*PASS',body)
  net=root/('outflow/'+top+'.map.v');rpt=root/('outflow/'+top+'.map.rpt');warn=root/('outflow/'+top+'.warn.log')
  assert 'EFX-0201' not in warn.read_text()
  counts=dict(collections.Counter(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M)))
  assert not counts.get('DSP24',0) and not counts.get('DSP48',0)
  for k,v in counts.items():assert re.findall(r'^EFX_'+k+r'\s*:\s*(\d+)\s*$',rpt.read_text(),re.M)==[str(v)]
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='VIDEO_PIPELINE_COMPACT_FIFO_MAP_PASS' if a.pipeline else 'VIDEO_FIFO_ISOLATED_MAP_PASS',reference=a.reference,pipeline=a.pipeline,area=a.area,resources=counts,seconds=secs,inputs_sha256=sha(root/'inputs.json'),map_sha256=sha(net),warning_sha256=sha(warn),report_sha256=sha(rpt),full_frame=False,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
