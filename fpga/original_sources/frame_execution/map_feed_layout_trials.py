"""Bounded isolated maps to screen data-layout/router structures, not board fit."""
import argparse,concurrent.futures,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,run,sha
from frozen_sources import frozen_paths

def one(root,src):
 root.mkdir(parents=True,exist_ok=False);top=src.stem;dst=root/src.name;shutil.copy2(src,dst)
 template=ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml'
 (root/(top+'.xml')).write_text(template.read_text().replace('frame_execution_meter',top))
 (root/(top+'.sdc')).write_text('create_clock -name clk -period 6.667 [get_ports {clk}]\n')
 (root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat '+top+'.xml --flow map --timeout 240\nexit /b %errorlevel%\n')
 (root/'inputs.json').write_text(json.dumps(dict(source=str(src),source_sha256=sha(src),runner_sha256=sha(__file__),files={f.name:sha(f) for f in root.iterdir() if f.is_file()}),indent=2))
 body,secs=run(['cmd.exe','/d','/c','run.cmd'],root,'map.log',timeout=270);assert re.search(r'map\s*:\s*PASS',body)
 rpt=root/'outflow'/(top+'.map.rpt');warn=root/'outflow'/(top+'.warn.log')
 counts={k:int(v) for k,v in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',rpt.read_text(),re.M)}
 assert sha(src)==sha(dst) and 'EFX-0654' not in warn.read_text(errors='replace')
 result=dict(status='ISOLATED_FEED_LAYOUT_MAP_NOT_BOARD',resources=counts,seconds=secs,source=str(src),source_sha256=sha(src),report_sha256=sha(rpt),warning_sha256=sha(warn))
 (root/'result.json').write_text(json.dumps(result,indent=2));return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 ref=frozen_paths([HERE/'unified_column_reader.sv',HERE/'unified_column_dispatch32.sv'],a.base)
 choices={'reader_reference':ref[0],'reader_capture':HERE/'variants/logical_row_capture/unified_column_reader.sv',
          'dispatch_reference':ref[1],'dispatch_radix3':HERE/'variants/radix3_dispatch/unified_column_dispatch32.sv'}
 results={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  pending={name:pool.submit(one,root/name,path) for name,path in choices.items()}
  for name,f in pending.items():results[name]=f.result();print(name,json.dumps(results[name]),flush=True)
 (root/'result.json').write_text(json.dumps(dict(status='ISOLATED_FEED_SCREEN_COMPLETE_NOT_BOARD',trials=results),indent=2))
if __name__=='__main__':main()
