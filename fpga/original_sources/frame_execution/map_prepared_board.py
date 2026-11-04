"""Map one already-frozen board assembly and record resources without releasing it."""
import argparse,collections,json,re
from pathlib import Path
from run_units import ROOT,EF,sha,run
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp')
 meta=json.loads((root/'inputs.json').read_text())
 for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
 assert not (root/'outflow').exists(),'Do not overwrite an existing build'
 (root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat ti60f225_oob.xml --flow map --timeout 1800\nexit /b %errorlevel%\n')
 body,secs=run(['cmd.exe','/d','/c','run.cmd'],root,'map.log',timeout=2100)
 assert re.search(r'map\s*:\s*PASS',body)
 net=root/'outflow/ti60f225_oob.map.v';rpt=root/'outflow/ti60f225_oob.map.rpt';warn=root/'outflow/ti60f225_oob.warn.log'
 counts={k:int(v) for k,v in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',rpt.read_text(),re.M)}
 dumped=dict(collections.Counter(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M)))
 for key in ('RAM10','DPRAM10','DSP24','DSP48'):assert counts.get(key,0)==dumped.get(key,0)
 warnings=warn.read_text(errors='replace');base=Path(meta.get('warning_reference_base',meta['base']))
 previous=(base/'outflow/ti60f225_oob.warn.log').read_text(errors='replace')
 pattern=r'^\[EFX-0201 WARNING\] (.*?)\. \('
 added=sorted(set(re.findall(pattern,warnings,re.M))-set(re.findall(pattern,previous,re.M)))
 for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
 result=dict(status='CONNECTED_BOARD_MAP_MEASURED_NOT_RELEASE',resources=counts,dumped_net_primitives=dumped,seconds=secs,
  new_undriven_signatures=added,comb_loop_warning='EFX-0654' in warnings,inputs_sha256=sha(root/'inputs.json'),
  map_sha256=sha(net),report_sha256=sha(rpt),warning_sha256=sha(warn),resource_fit_proven=False,physical_timing_pass=False,
  full_baseline=False,board_download_allowed=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
