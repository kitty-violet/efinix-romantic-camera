"""Verify package files, project references and firmware ROM lanes."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
 for name,h in manifest['files'].items():assert digest(ROOT/name)==h,name
 project=ROOT/'project';tree=ET.parse(project/'ti60f225_oob.xml')
 for e in tree.iter():
  if e.tag.split('}')[-1] in ('design_file','sdc_file','peri_file','default_peri_file'):assert (project/e.attrib['name']).is_file(),e.attrib['name']
 data=(ROOT/'firmware/prebuilt/control.bin').read_bytes();data+=bytes(8192-len(data))
 for lane in range(4):
  p=project/f'EfxSapphireSoc.v_toplevel_system_ramA_logic_ram_symbol{lane}.bin'
  assert bytes(int(x,2) for x in p.read_text().splitlines())==data[lane::4]
 print('PACKAGE_OK',len(manifest['files']),'files')
if __name__=='__main__':main()
