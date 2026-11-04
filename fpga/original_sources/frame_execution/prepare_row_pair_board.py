"""Replace the entire CNN storage service in the frozen real board assembly."""
import argparse,json,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
base=ROOT/'tmp/frame-execution/connected-board-header-ram1'
meta=json.loads((base/'inputs.json').read_text());prior=json.loads((base/'result.json').read_text())
assert sha(base/'inputs.json')==prior['inputs_sha256']
tested=ROOT/'tmp/frame-execution/row-pair-guard1'
proof=json.loads((tested/'result.json').read_text())
assert proof['status']=='INTEGRATED_DATAPATH_GUARD_NATIVE_PASS'
assert sha(tested/'inputs.json')==proof['inputs_sha256']
replace={}
for r in json.loads((tested/'inputs.json').read_text())['inputs']:
 f=Path(r['snapshot']);assert sha(f)==r['sha256']
 if f.suffix=='.sv' and not f.name.startswith('tb_'):replace[f.name]=f
# The old cluster embeds header helpers now carried by row_pair_compute_cluster.
omit={'compute_column_cluster.sv'}
root.mkdir()
for name in ('rtl','assembly'):shutil.copytree(base/name,root/name)
for f in base.iterdir():
 if f.is_file() and f.suffix in ('.xml','.sdc','.bin','.v','.sv','.vh','.svh'):shutil.copy2(f,root/f.name)
records=[];names=set()
for r in meta['inputs']:
 old=Path(r['snapshot']);assert sha(old)==r['sha256']
 if old.name in omit:continue
 src=replace.get(old.name,old);dst=root/old.relative_to(base)
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst),base_sha256=r['sha256']));names.add(old.name)
paths=[root/Path(f).relative_to(base) for f in meta['source_files'] if Path(f).name not in omit]
for name,src in replace.items():
 if name in names:continue
 dst=root/'assembly'/name;shutil.copy2(src,dst);paths.append(dst)
 records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)));names.add(name)
ns={'e':'http://www.efinixinc.com/enf_proj'};ET.register_namespace('efx',ns['e'])
xml=root/'ti60f225_oob.xml';tree=ET.parse(xml);design=tree.getroot().find('e:design_info',ns)
for e in list(design.findall('e:design_file',ns)):design.remove(e)
for i,f in enumerate(paths,1):design.insert(i,ET.Element('{'+ns['e']+'}design_file',name=f.relative_to(root).as_posix(),version='default',library='default'))
ET.indent(tree,space='    ');tree.write(xml,encoding='utf-8',xml_declaration=False)
for r in records:
 if Path(r['snapshot'])==xml:r.update(source='reviewed row-pair source assembly',sha256=sha(xml))
meta.update(inputs=records,base=str(base),base_inputs_sha256=sha(base/'inputs.json'),source_files=list(map(str,paths)),
            row_pair_guard_sha256=sha(tested/'result.json'),override=None,overrides=[],board_release=False)
(root/'inputs.json').write_text(json.dumps(meta,indent=2)+'\n')
for r in records:assert sha(r['snapshot'])==r['sha256']
print('ROW_PAIR_BOARD_FROZEN',len(records))
