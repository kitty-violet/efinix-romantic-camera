"""Resource check of the residual port adapter; arithmetic pool is external."""
import collections,json,re,shutil,traceback,argparse
from pathlib import Path
from run_units import HERE,ROOT,EF,sha,run
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);a=p.parse_args()
 build=a.root.resolve();assert build.is_relative_to(ROOT/'tmp');build.mkdir(parents=True,exist_ok=False)
 ref=json.loads((a.reference/'result.json').read_text());assert ref['status']=='LOCAL_RESIDUAL_LOOKUP_SINGLE_SHARED_POOL_PASS'
 manifest=json.loads((a.reference/'inputs.json').read_text());assert sha(a.reference/'inputs.json')==ref['inputs_sha256']
 top='residual_pool_dispatch32';source=HERE/(top+'.sv');r=next(r for r in manifest['inputs'] if Path(r['path'])==source)
 assert sha(source)==sha(r['snapshot'])==r['sha256']
 try:
  shutil.copy2(source,build/source.name);shutil.copy2(Path(__file__),build/Path(__file__).name)
  xml=(ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml').read_text().replace('frame_execution_meter',top)
  (build/(top+'.xml')).write_text(xml);(build/(top+'.sdc')).write_text('create_clock -name clk -period 6.6667 [get_ports {clk}]\n')
  (build/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat '+top+'.xml --flow map --timeout 600\nexit /b %errorlevel%\n')
  inputs=dict(source=str(source),source_sha256=sha(source),reference=str(a.reference/'result.json'),reference_sha256=sha(a.reference/'result.json'),
   files={f.name:sha(f) for f in build.iterdir() if f.is_file()})
  (build/'inputs.json').write_text(json.dumps(inputs,indent=2))
  body,seconds=run(['cmd.exe','/d','/c','run.cmd'],build,'map.log');assert re.search(r'map\s*:\s*PASS',body)
  net=build/('outflow/'+top+'.map.v');rpt=build/('outflow/'+top+'.map.rpt')
  counts=dict(collections.Counter(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M)))
  for k,v in counts.items():assert re.findall(r'^EFX_'+k+r'\s*:\s*(\d+)\s*$',rpt.read_text(),re.M)==[str(v)]
  assert not counts.get('DSP48',0) and not counts.get('DSP24',0) and not counts.get('RAM10',0) and not counts.get('DPRAM10',0)
  for f,h in inputs['files'].items():assert sha(build/f)==h
  assert sha(source)==inputs['source_sha256']
  result=dict(status='LOCAL_DISPATCH_MAP_PASS_NO_EXTRA_MAC_OR_RQ',resources=counts,seconds=seconds,map_sha256=sha(net),report_sha256=sha(rpt),
   inputs_sha256=sha(build/'inputs.json'),pnr=False,full_frame=False)
  (build/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (build/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
