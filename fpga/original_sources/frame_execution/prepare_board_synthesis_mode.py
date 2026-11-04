"""Change only the documented synthesis objective in a new frozen board."""
import argparse,json,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,sha
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=['area2','cong'],required=True);a=p.parse_args()
base=a.base.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
meta=json.loads((base/'inputs.json').read_text());prior=json.loads((base/'result.json').read_text());assert sha(base/'inputs.json')==prior['inputs_sha256']
root.mkdir();records=[]
for folder in ('rtl','assembly'):shutil.copytree(base/folder,root/folder)
for r in meta['inputs']:
 old=Path(r['snapshot']);assert sha(old)==r['sha256'];dst=root/old.relative_to(base);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,dst)
 records.append(dict(source=str(old),snapshot=str(dst),sha256=sha(dst),base_sha256=r['sha256']))
ns={'e':'http://www.efinixinc.com/enf_proj'};ET.register_namespace('efx',ns['e']);xml=root/'ti60f225_oob.xml';tree=ET.parse(xml)
param=tree.getroot().find("e:synthesis/e:param[@name='mode']",ns);assert param.get('value')=='speed';param.set('value',a.mode)
ET.indent(tree,space='    ');tree.write(xml,encoding='utf-8',xml_declaration=False)
for r in records:
 if Path(r['snapshot'])==xml:r['sha256']=sha(xml)
meta.update(inputs=records,base=str(base),base_inputs_sha256=sha(base/'inputs.json'),source_files=[str(root/Path(f).relative_to(base)) for f in meta['source_files']],synthesis_mode=a.mode,runner_sha256=sha(__file__),board_release=False)
(root/'inputs.json').write_text(json.dumps(meta,indent=2)+'\n')
assert sum(r['sha256']!=r['base_sha256'] for r in records)==1
print('SYNTHESIS_MODE_FROZEN',a.mode,len(records),flush=True)
