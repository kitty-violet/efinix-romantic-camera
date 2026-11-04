"""Measure the fixed boot controller only; requires its behavioral bus gate."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--proof',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();proof=a.proof.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists()
g=json.loads((proof/'result.json').read_text());assert g['status']=='FIXED_BOOT_SEQUENCER_UNIT_PASS' and sha(proof/'inputs.json')==g['inputs_sha256']
pm=json.loads((proof/'inputs.json').read_text());source=Path(next(x['source'] for x in pm['inputs'] if Path(x['source']).name=='cnn_boot_sequencer.sv'));assert source.is_relative_to(HERE/'variants') and sha(source)==sha(proof/source.name)
r.mkdir();shutil.copy2(source,r/source.name);top=source.stem
project=(ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml').read_text().replace('frame_execution_meter',top)
(r/(top+'.xml')).write_text(project);(r/(top+'.sdc')).write_text('create_clock -name clk -period 6.667 [get_ports {clk}]\n')
(r/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat '+top+'.xml --flow map --timeout 180\nexit /b %errorlevel%\n')
(r/'inputs.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),proof=str(proof),proof_sha256=sha(proof/'result.json'),project_sha256=sha(r/(top+'.xml')),runner_sha256=sha(__file__)),indent=2))
try:
 body,seconds=run(['cmd.exe','/d','/c','run.cmd'],r,'map.log',timeout=210)
 assert re.search(r'map\s*:\s*PASS',body)
 report=r/'outflow'/(top+'.map.rpt');warning=r/'outflow'/(top+'.warn.log')
 counts={k:int(v) for k,v in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',report.read_text(),re.M)}
 assert 'EFX-0654' not in warning.read_text() and sha(source)==sha(r/source.name)
 result=dict(status='FIXED_BOOT_SEQUENCER_LOCAL_MAP_PASS',resources=counts,seconds=seconds,inputs_sha256=sha(r/'inputs.json'),report_sha256=sha(report),map_sha256=sha(r/'outflow'/(top+'.map.v')),board_integrated=False,board_fit=False)
 (r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
except Exception as e:
 (r/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
