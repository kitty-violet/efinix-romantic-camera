"""Small structural gate before an expensive whole-board map/place."""
import argparse,json,re,shutil,subprocess,time,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from windows_process_tree import stop_launched_tree

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();candidate=a.source.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
base=ROOT/'tmp/efinity-dw8-stream-physical-attempt3b-seed1';v=HERE/'variants/fixed_window_dot8x32'
shutil.copy2(candidate,root/'fixed_window_dot8x32.sv');shutil.copy2(v/'window_physical_probe.sv',root/'dw8_stream_physical_probe.sv')
for name in ('dw8_stream_physical.peri.xml','dw8_stream_physical.sdc'):shutil.copy2(base/name,root/name)
ns='{http://www.efinixinc.com/enf_proj}';ET.register_namespace('efx',ns[1:-1]);tree=ET.parse(base/'dw8_stream_physical.xml');di=tree.find(ns+'design_info')
for e in list(di):
    if e.tag==ns+'design_file':di.remove(e)
for i,n in enumerate(('fixed_window_dot8x32.sv','dw8_stream_physical_probe.sv')):
    di.insert(i+1,ET.Element(ns+'design_file',name=n,version='default',library='default'))
tree.write(root/'dw8_stream_physical.xml',encoding='utf-8')
shutil.copy2(__file__,root/'runner.py')
(root/'inputs.json').write_text(json.dumps(dict(files={f.name:sha(f) for f in root.iterdir() if f.is_file()},source_sha256=sha(candidate),hardware_access=False),indent=2))
(root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\ncall efx_run.bat dw8_stream_physical.xml --flow map --timeout 150\nexit /b %errorlevel%\n')
start=time.monotonic()
with (root/'map.log').open('x') as log:
    child=subprocess.Popen(['cmd.exe','/d','/c','run.cmd'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
    try:code=child.wait(timeout=180)
    except subprocess.TimeoutExpired:stop_launched_tree(child);raise
assert code==0 and re.search(r'map\s*:\s*PASS',(root/'map.log').read_text())
net=root/'outflow/dw8_stream_physical.map.v';text=re.sub(r'/\*.*?\*/|//[^\n]*','',net.read_text(),flags=re.S)
drivers=[]
for kind,name,body in re.findall(r'^\s*(EFX_\w+)\s+(\S+)\s*\((.*?)\)\s*;',text,re.M|re.S):
    if ('_driver' in name) and ('arithmetic/' in name):
        drivers.append(dict(kind=kind,name=name,ports=dict(re.findall(r'\.(\w+)\(([^()]*)\)',body))))
assert len(drivers)==63,(len(drivers),drivers)
assert all(d['kind']=='EFX_LUT4' for d in drivers)
assert len({d['ports']['O'] for d in drivers})==63
dsp=re.findall(r'^\s*EFX_DSP24\s+\S+\s*\((.*?)\)\s*;',text,re.M|re.S)
assert len(dsp)==256 and all('pair_advance' in re.search(r'\.CE\(([^()]*)\)',b)[1] for b in dsp)
result=dict(status='LOCAL_WINDOW_63_BRANCHES_256_DSP_CE_STRUCTURAL_PASS',seconds=time.monotonic()-start,
            drivers=drivers,map_sha256=sha(net),source_sha256=sha(candidate),physical=False,hardware_access=False)
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='drivers'}))
