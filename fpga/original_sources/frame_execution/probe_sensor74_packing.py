"""Single bounded official L2 packing experiment; no placement or hardware."""
import json,shutil,subprocess,time,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,EF,sha
from windows_process_tree_safe import stop_launched_tree

BASE=ROOT/'tmp/frame-execution/video-sensor74-place1'
DEST=ROOT/'tmp/frame-execution/video-sensor74-pack-l2-1'

def main():
 assert not DEST.exists()
 shutil.copytree(BASE,DEST,ignore=shutil.ignore_patterns('*.rpt','*.log','*.out','*.place','*.placeloc','*.netlist','*.clstdb','*.cnst','*.htree','*.troutingtraces','result.json','stages.json','failure.json','capacity_gate.json','capacity-failure-review.json'))
 shutil.copy2(BASE/'fixed_dsp.placeloc',DEST/'fixed_dsp.placeloc')
 xml=DEST/'ti60f225_oob.xml';tree=ET.parse(xml);ns='{http://www.efinixinc.com/enf_proj}'
 pnr=tree.getroot().find(ns+'place_and_route');assert pnr is not None
 assert not [x for x in pnr if x.attrib.get('name')=='packing_level']
 ET.SubElement(pnr,ns+'param',{'name':'packing_level','value':'L2','value_type':'e_option'});tree.write(xml,encoding='utf-8',xml_declaration=True)
 evidence={str(f):sha(f) for f in (Path(__file__),BASE/'outflow/ti60f225_oob.map.v',BASE/'outflow/ti60f225_oob.vdb',BASE/'ti60f225_oob.sdc',BASE/'fixed_dsp.placeloc')}
 for f in ('outflow/ti60f225_oob.map.v','outflow/ti60f225_oob.vdb','ti60f225_oob.sdc','fixed_dsp.placeloc'):assert sha(BASE/f)==sha(DEST/f)
 proof=dict(status='OFFICIAL_PACKING_L2_PROBE_NOT_RELEASE',base=str(BASE),packing_level='L2',evidence=evidence,source_changed=False,hardware_access=False,budget_s=120)
 (DEST/'probe-inputs.json').write_text(json.dumps(proof,indent=2))
 cmd=DEST/'pack-l2.cmd';cmd.write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat ti60f225_oob.xml --flow pnr --pnr_opts pack packing_level=L2 --timeout 120\nexit /b %errorlevel%\n')
 start=time.monotonic()
 with (DEST/'pack-l2.log').open('x') as log:
  proc=subprocess.Popen(['cmd.exe','/d','/c',str(cmd)],cwd=DEST,stdout=log,stderr=subprocess.STDOUT)
  try:rc=proc.wait(timeout=145)
  except subprocess.TimeoutExpired:
   (DEST/'timeout-stop.json').write_text(json.dumps(stop_launched_tree(proc)));raise
 result=dict(status='PACKING_PROBE_FINISHED_NOT_PLACED',exit_code=rc,seconds=time.monotonic()-start,log_sha256=sha(DEST/'pack-l2.log'),hardware_access=False)
 (DEST/'probe-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
