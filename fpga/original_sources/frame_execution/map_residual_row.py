"""Map row adapter ONLY, referencing a completed exact-source simulation."""
import argparse,collections,json,re,shutil,traceback
from pathlib import Path
from run_units import HERE,ROOT,EF,sha,run
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--row-source',type=Path,default=HERE/'residual_row_shared.sv');a=p.parse_args()
 build=a.root.resolve();assert build.is_relative_to(ROOT/'tmp');build.mkdir(parents=True,exist_ok=False)
 ref=json.loads((a.reference/'result.json').read_text());assert ref['status']=='LOCAL_RESIDUAL_REAL_ARENA_SHARED_POOL_ROW_PASS'
 manifest=json.loads((a.reference/'inputs.json').read_text());assert sha(a.reference/'inputs.json')==ref['inputs_sha256']
 top='residual_row_shared';paths=[HERE/'residual_packet_dispatch32.sv',a.row_source.resolve()];records=[]
 try:
  for f in paths:
   row=next(r for r in manifest['inputs'] if Path(r['path'])==f);assert sha(f)==sha(row['snapshot'])==row['sha256']
   shutil.copy2(f,build/f.name);records.append(dict(path=str(f),snapshot=str(build/f.name),sha256=sha(f)))
  shutil.copy2(Path(__file__),build/Path(__file__).name)
  xml=(ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml').read_text().replace('frame_execution_meter',top)
  xml=xml.replace('<efx:design_file name="'+top+'.sv" version="default" library="default" />',
    ''.join('<efx:design_file name="'+f.name+'" version="default" library="default" />' for f in paths))
  (build/(top+'.xml')).write_text(xml);(build/(top+'.sdc')).write_text('create_clock -name clk -period 6.6667 [get_ports {clk}]\n')
  (build/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat '+top+'.xml --flow map --timeout 600\nexit /b %errorlevel%\n')
  inputs=dict(inputs=records,reference=str(a.reference/'result.json'),reference_sha256=sha(a.reference/'result.json'),files={f.name:sha(f) for f in build.iterdir() if f.is_file()})
  (build/'inputs.json').write_text(json.dumps(inputs,indent=2))
  body,seconds=run(['cmd.exe','/d','/c','run.cmd'],build,'map.log');assert re.search(r'map\s*:\s*PASS',body)
  net=build/('outflow/'+top+'.map.v');rpt=build/('outflow/'+top+'.map.rpt')
  warning_log=(build/('outflow/'+top+'.warn.log')).read_text()
  assert 'EFX-0201' not in warning_log, 'synthesis rewired non-driven payload: reject despite map exit PASS'
  counts=dict(collections.Counter(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M)))
  for k,v in counts.items():assert re.findall(r'^EFX_'+k+r'\s*:\s*(\d+)\s*$',rpt.read_text(),re.M)==[str(v)]
  assert counts.get('DSP24',0)==counts.get('DSP48',0)==counts.get('RAM10',0)==0
  for f,h in inputs['files'].items():assert sha(build/f)==h
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='LOCAL_RESIDUAL_ROW_ADAPTER_MAP_PASS',resources=counts,seconds=seconds,map_sha256=sha(net),report_sha256=sha(rpt),
   inputs_sha256=sha(build/'inputs.json'),includes_shared_pool_and_arena=False,pnr=False,full_frame=False)
  (build/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (build/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()

