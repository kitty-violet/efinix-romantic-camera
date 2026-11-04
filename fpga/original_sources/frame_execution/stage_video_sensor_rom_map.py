"""Stage and synthesize the pin-tested ROM verification candidate."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
from run_units import ROOT,HERE,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--pins',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 n=a.native.resolve();b=a.root.resolve();pins=a.pins.resolve();assert b.is_relative_to(ROOT/'tmp/frame-execution') and not b.exists()
 proof=json.loads((n/'inputs.json').read_text());pr=json.loads((pins/'result.json').read_text());pi=json.loads((pins/'inputs.json').read_text())
 assert proof['gain_register']==0x34 and proof['source_sha256']==sha(n/'native_source.v')
 assert pr['status']=='ROM_BACKED_SENSOR_VERIFY_PIN_PROTOTYPE_PASS_NOT_PHYSICAL_RELEASE' and [r['mode'] for r in pr['cases']]==[0,10,8]
 assert pr['inputs_sha256']==sha(pins/'inputs.json') and pi['prepared_sha256']==sha(n/'inputs.json')
 for f,h in proof['files'].items():assert sha(f)==h
 for f,h in pi['files'].items():assert sha(f)==h
 parent=ROOT/'tmp/frame-execution/connected-board-video-service-probe60-input-isp-explicit'
 pm=json.loads((parent/'inputs.json').read_text());b.mkdir();rows=[]
 for row in pm['inputs']:
  old=Path(row['snapshot']);assert sha(old)==row['sha256'];rel=old.relative_to(parent)
  src=n/'native_source.v' if rel.as_posix()=='native_source.v' else old;dst=b/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
  rows.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
 shutil.copy2(parent/'map.cmd',b/'map.cmd')
 meta=dict(inputs=rows,parent=str(parent),native=str(n),prototype_pin_test=str(pins),prototype_pin_test_sha256=sha(pins/'result.json'),
  builder_sha256=sha(__file__),original_sdc_unchanged=True,hardware_access=False,board_release=False)
 (b/'inputs.json').write_text(json.dumps(meta,indent=2))
 cmd=[sys.executable,str(HERE/'map_video_service_candidate.py'),'--native',str(n),'--root',str(b),'--run-existing']
 subprocess.run(cmd,check=True)

if __name__=='__main__':main()
