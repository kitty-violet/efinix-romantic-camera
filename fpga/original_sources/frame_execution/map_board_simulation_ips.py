"""Probe official netlist export of unchanged board IP for simulation.

These are simulation artifacts, not new IP implementations or resource-fit
measurements. Export may retain encrypted regions and is NOT evidence that a
simulator can compile or run the result. The full-board resource report remains
the authority for area.
No interface, P&R, bitstream, programming or source edits are performed.
"""
import argparse, concurrent.futures, json, re, shutil, subprocess, time
import xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,EF,sha

NS={'e':'http://www.efinixinc.com/enf_proj'}
ET.register_namespace('efx',NS['e'])

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
 p.add_argument('--boot-image',type=Path,help='Explicit replacement OCR image; CPU simulation export only')
 p.add_argument('--board',type=Path,default=ROOT/'tmp/frame-execution/connected-board-sharing1')
 p.add_argument('--modules',nargs='+',default=['csi_rx_controller','dvi_encoder','EfxSapphireSoc','ddr3_top'])
 a=p.parse_args();base=a.board.resolve();out=a.root.resolve()
 assert out.is_relative_to(ROOT/'tmp') and not out.exists();out.mkdir(parents=True)
 manifest=json.loads((base/'inputs.json').read_text())
 for r in manifest['inputs']: assert sha(Path(r['snapshot']))==r['sha256']
 assert set(a.modules)<=set(('csi_rx_controller','dvi_encoder','EfxSapphireSoc','ddr3_top','EfxCPUSp1','EfxCPUSp2'))
 if a.boot_image:
  assert a.modules==['EfxSapphireSoc'],'A replacement boot image must not silently alter unrelated IP exports'
  boot=a.boot_image.resolve();boot_meta=json.loads((boot/'boot.json').read_text())
  assert sha(boot/'boot.bin')==boot_meta['sha256']
 def one(module):
  run=out/module;run.mkdir();name='board_ip'
  for folder in ('rtl','assembly'):shutil.copytree(base/folder,run/folder)
  for f in base.glob('*.bin'):shutil.copy2(f,run/f.name)
  if a.boot_image:
   from board_integration.boot_image import make_program
   expected,_=make_program();assert expected==(boot/'boot.bin').read_bytes()
   for byte in range(4):
    f=boot/f'EfxSapphireSoc.v_toplevel_system_ramA_logic_ram_symbol{byte}.bin'
    assert bytes(int(v,2) for v in f.read_text().split())==expected[byte::4]
    shutil.copy2(f,run/f.name)
  for f in base.glob('*.vh'):shutil.copy2(f,run/f.name)
  tree=ET.parse(base/'ti60f225_oob.xml');project=tree.getroot()
  project.set('name',name);project.find('e:design_info/e:top_module',NS).set('name',module)
  # No board timing claims are made for these isolated simulation netlists.
  for parent in project.iter():
   for child in list(parent):
    if child.tag.endswith('constraint_file'):parent.remove(child)
  for elem in project.findall('.//e:include_path',NS):
   value=elem.get('name')
   if value and str(base) in value:elem.set('name',value.replace(str(base),str(run)))
  ET.indent(tree,space='    ');tree.write(run/(name+'.xml'),encoding='utf-8',xml_declaration=False)
  (run/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat '+name+'.xml --flow map --timeout 900\nexit /b %errorlevel%\n')
  t=time.monotonic()
  try:
   with (run/'map.log').open('wb') as log:
    proc=subprocess.run(['cmd.exe','/d','/c','run.cmd'],cwd=run,stdout=log,stderr=subprocess.STDOUT,timeout=960)
   body=(run/'map.log').read_text(errors='replace')
   net=run/'outflow'/f'{name}.map.v'
   ok=proc.returncode==0 and bool(re.search(r'map\s*:\s*PASS',body)) and net.is_file()
   result=dict(module=module,seconds=time.monotonic()-t,returncode=proc.returncode,
    map_complete=ok,net_sha256=sha(net) if net.is_file() else None,
    board_inputs_sha256=sha(base/'inputs.json'),simulation_pass=False,
    boot_sha256=boot_meta['sha256'] if a.boot_image else None,
    purpose='Unchanged vendor IP official netlist for simulation, not a replacement design')
  except subprocess.TimeoutExpired:
   result=dict(module=module,map_complete=False,error='map timeout',simulation_pass=False)
  (run/'result.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result),flush=True);return result
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  results=list(pool.map(one,a.modules))
 (out/'result.json').write_text(json.dumps(dict(results=results,production_rtl_modified=False),indent=2)+'\n')

if __name__=='__main__':main()
