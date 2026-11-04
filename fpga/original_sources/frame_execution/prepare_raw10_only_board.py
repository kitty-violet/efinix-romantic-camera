"""Freeze only the documented, A/B-tested Pack40 CSI specialization."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--proof',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();base=a.base.resolve();proof=a.proof.resolve()
 assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 meta=json.loads((base/'inputs.json').read_text());mapped=json.loads((base/'result.json').read_text())
 assert sha(base/'inputs.json')==mapped['inputs_sha256']
 result=json.loads((proof/'result.json').read_text());tested=json.loads((proof/'inputs.json').read_text())
 assert result['status']=='CSI_RAW10_SPECIALIZATION_PASS' and result['inputs_sha256']==sha(proof/'inputs.json')
 assert sha(base/'rtl/csi_rx_controller.sv')==tested['board_csi_sha256']
 fresh=proof/'csi_rx_controller_raw10.sv';assert sha(fresh)==result['specialized_csi_sha256']
 old=(base/'rtl/csi_rx_controller.sv').read_text();assert old.count('.PACK_TYPE (15)')==1
 assert fresh.read_text()==old.replace('.PACK_TYPE (15)','.PACK_TYPE (1)',1)
 root.mkdir();records=[]
 for rec in meta['inputs']:
  src=Path(rec['snapshot']);assert sha(src)==rec['sha256'];dst=root/src.relative_to(base)
  dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(fresh if src.name=='csi_rx_controller.sv' else src,dst)
  records.append(dict(source=str(src),snapshot=str(dst),base_sha256=rec['sha256'],sha256=sha(dst),transformed=src.name=='csi_rx_controller.sv'))
 # Earlier manifests enumerate compilation units but omit nested textual
 # includes (notably I2C timescale/defines). Freeze and hash these as well.
 for folder in ('rtl','assembly'):
  for src in (base/folder).rglob('*'):
   if not src.is_file():continue
   dst=root/src.relative_to(base)
   if not dst.exists():
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst),support_file=True))
 for src in base.iterdir():
  if src.is_file() and src.suffix in ('.xml','.sdc','.bin','.v','.sv','.vh','.svh') and not (root/src.name).exists():
   dst=root/src.name;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
 sources=[root/Path(f).relative_to(base) for f in meta['source_files']]
 meta.update(inputs=records,base=str(base),base_inputs_sha256=sha(base/'inputs.json'),source_files=list(map(str,sources)),csi_pack_type=1,
  csi_raw10_proof=str(proof),csi_raw10_proof_sha256=sha(proof/'result.json'),runner_sha256=sha(__file__),board_download_allowed=False)
 (root/'inputs.json').write_text(json.dumps(meta,indent=2))
 print('RAW10_ONLY_BOARD_FROZEN',len(records),flush=True)
if __name__=='__main__':main()
