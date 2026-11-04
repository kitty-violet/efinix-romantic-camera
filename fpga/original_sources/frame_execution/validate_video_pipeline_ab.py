"""Exact displayed RGB/coordinates/frame-id comparison, old/new FIFO pipeline."""
import argparse,concurrent.futures,json,re,shutil,traceback,os,subprocess,time
from pathlib import Path
from run_units import HERE,ROOT,MS,sha,run
from validate_compact_video import VIDEO,selected
def run_isolated(args,root,name):
 env=os.environ.copy();temp=root/'temp';temp.mkdir(exist_ok=True);env.update(TEMP=str(temp),TMP=str(temp),TMPDIR=str(temp))
 start=time.monotonic()
 with (root/name).open('w',encoding='utf-8') as out:
  result=subprocess.run(list(map(str,args)),cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=600)
 body=(root/name).read_text()
 if result.returncode or re.search(r'(?m)^#?\s*(FAIL|\*\* (Error|Fatal))',body):raise RuntimeError(name+body[-2000:])
 return body,time.monotonic()-start
def one(root,compact,wide,area=False):
 root.mkdir();top='tb_rgb_scaler_display_pipeline'+('_wide_cadence' if wide else '')
 paths=selected()+[VIDEO/n for n in ('rgb_bilinear_scaler_prototype.sv','resize_axis_down_3_to_2.sv','resize_axis_up_2_to_3.sv','hdmi_frame_scheduler.sv')]+[VIDEO/('variants/compact_fifo/' if compact else '')/'rgb_scaler_display_pipeline.sv']
 if area:
  paths[-1]=VIDEO/'variants/compact_area_fifo/rgb_scaler_display_pipeline.sv' if compact else ROOT/'tmp/efinity-smoke-2026.1/sc431-v23-fifo2026-seed1-effort5/rtl/rgb_scaler_display_pipeline.sv'
  paths.append(VIDEO/'rgb_area_downscaler_3to2.sv')
 records=[]
 for f in paths:
  dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
 original=VIDEO/(top+'.sv');tb=original.read_text()
 tb=tb.replace('    integer accepted_input_pixels = 0;', '    integer pixel_file;\n    initial pixel_file=$fopen("pixels.hex","w");\n    integer accepted_input_pixels = 0;')
 tb=tb.replace('            active_output_pixels <= active_output_pixels + 1;', '            $fdisplay(pixel_file,"%04h %04h %04h %06h",source_frame_id,x,y,rgb);\n            active_output_pixels <= active_output_pixels + 1;')
 tb=tb.replace('        $finish;', '        $fclose(pixel_file);\n        $finish;');(root/(top+'.sv')).write_text(tb)
 (root/'inputs.json').write_text(json.dumps(dict(inputs=records,tb_original=str(original),tb_original_sha256=sha(original),generated_tb_sha256=sha(root/(top+'.sv')),runner_sha256=sha(__file__)),indent=2))
 run_isolated([MS/'vlib.exe','work'],root,'vlib.log')
 body,_=run_isolated([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records],top+'.sv'],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
 body,secs=run_isolated([MS/'vsim.exe','-c','work.'+top,'-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log')
 lines=re.findall(r'^# (PASS .*?(?:pipeline|cadence).*)$',body,re.M);assert len(lines)==1
 expected=1920*6 if wide else 12*6;assert len((root/'pixels.hex').read_text().splitlines())==expected
 for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
 return dict(compact=compact,wide=wide,seconds=secs,line=lines[0],pixels=expected,pixel_sha256=sha(root/'pixels.hex'),inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--area',action='store_true');a=p.parse_args();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
   futures=[ex.submit(one,root/f'{"compact" if c else "reference"}-{"wide" if w else "small"}',c,w,a.area) for w in (False,True) for c in (False,True)]
   results=[f.result() for f in futures]
  for i in (0,2):assert results[i]['pixel_sha256']==results[i+1]['pixel_sha256']
  result=dict(status='VIDEO_PIPELINE_FIFO_AB_EXACT_PIXELS_PASS',area=a.area,results=results,cnn_bypass=True,full_height=False,full_frame_cnn=False,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
