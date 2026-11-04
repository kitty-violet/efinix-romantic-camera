"""Reuse five completed tests only for byte-identical actual mapped circuits."""
import argparse,json
from pathlib import Path
from run_units import ROOT,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--tested',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 prepared,tested,root=map(Path.resolve,(a.prepared,a.tested,a.root));assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 old=json.loads((tested/'inputs.json').read_text());result=json.loads((tested/'result.json').read_text());current=json.loads((prepared/'inputs.json').read_text())
 assert result['status']=='SENSOR73_MAPPED_I2C_PINS_PASS' and result['inputs_sha256']==sha(tested/'inputs.json')
 assert [r['mode'] for r in result['cases']]==[0,10,4,6,8]
 original=Path(old['prepared']);assert old['prepared_sha256']==sha(original/'inputs.json')
 for f,h in old['files'].items():assert sha(f)==h
 files={str(prepared/'inputs.json'):sha(prepared/'inputs.json')}
 for f,h in current['files'].items():
  path=Path(f);assert sha(path)==h==sha(original/path.name),('actual circuit changed; rerun tests',path.name)
  files[str(path)]=h
 for f in (tested/'inputs.json',tested/'result.json',original/'inputs.json'):files[str(f)]=sha(f)
 root.mkdir();meta=dict(prepared=str(prepared),prepared_sha256=sha(prepared/'inputs.json'),files=files,reference_test=str(tested),runner_sha256=sha(__file__),hardware_access=False)
 (root/'inputs.json').write_text(json.dumps(meta,indent=2))
 output=dict(status='SENSOR73_IDENTICAL_MAPPED_I2C_PINS_EVIDENCE_PASS',inputs_sha256=sha(root/'inputs.json'),cases=result['cases'],reexecuted=False,
  reason='Actual mapped sensor circuit, testbench, expected sequence and every official primitive model are byte-identical to the completed five-case test.',hardware_access=False,board_release=False)
 (root/'result.json').write_text(json.dumps(output,indent=2));print(output['status'])

if __name__=='__main__':main()
