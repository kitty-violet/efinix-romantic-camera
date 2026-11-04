"""Freeze a video-only AVI auxiliary replacement, with exact short gates."""
import argparse,json,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,HERE,sha
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--proof',type=Path,required=True);p.add_argument('--mode',choices=['speed','cong'],default='speed');a=p.parse_args()
base=a.base.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
proof=json.loads((a.proof/'result.json').read_text());pm=json.loads((a.proof/'inputs.json').read_text());assert proof['status']=='FIXED_VIDEO_CODEC_AB_PASS' and sha(a.proof/'inputs.json')==proof['inputs_sha256']
aux=HERE/'variants/fixed_video_aux/tx_aux_video_only.sv';assert any(x.get('source')==str(aux) and x['source_sha256']==sha(aux) for x in pm['inputs'])
meta=json.loads((base/'inputs.json').read_text());prior=json.loads((base/'result.json').read_text());assert sha(base/'inputs.json')==prior['inputs_sha256']
root.mkdir();records=[];changes=[]
for folder in ('rtl','assembly'):shutil.copytree(base/folder,root/folder)
for old in meta['inputs']:
 f=Path(old['snapshot']);assert sha(f)==old['sha256'];dst=root/f.relative_to(base);dst.parent.mkdir(parents=True,exist_ok=True)
 if f.name=='tx_aux.v_encrypted.v':
  dst.write_text(aux.read_text().replace('module tx_aux_video_only(','module tx_aux(',1));source=aux;changes.append(str(dst))
 else:shutil.copy2(f,dst);source=f
 records.append(dict(source=str(source),snapshot=str(dst),sha256=sha(dst),base_sha256=old['sha256']))
assert len(changes)==1
if a.mode!='speed':
 ns={'e':'http://www.efinixinc.com/enf_proj'};ET.register_namespace('efx',ns['e']);xml=root/'ti60f225_oob.xml';tree=ET.parse(xml)
 param=tree.getroot().find("e:synthesis/e:param[@name='mode']",ns);assert param.get('value')=='speed';param.set('value',a.mode);ET.indent(tree,space='    ');tree.write(xml,encoding='utf-8',xml_declaration=False)
 for rec in records:
  if Path(rec['snapshot'])==xml:rec['sha256']=sha(xml)
 changes.append(str(xml))
meta.update(inputs=records,base=str(base),base_inputs_sha256=sha(base/'inputs.json'),source_files=[str(root/Path(f).relative_to(base)) for f in meta['source_files']],fixed_video_aux=dict(proof=str(a.proof),result_sha256=sha(a.proof/'result.json'),source_sha256=sha(aux)),synthesis_mode=a.mode,runner_sha256=sha(__file__),board_release=False)
(root/'inputs.json').write_text(json.dumps(meta,indent=2)+'\n')
assert sum(x['sha256']!=x['base_sha256'] for x in records)==len(changes)
print('FIXED_VIDEO_AUX_BOARD_FROZEN',len(records),a.mode,flush=True)
