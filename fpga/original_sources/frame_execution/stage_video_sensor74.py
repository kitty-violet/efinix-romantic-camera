"""Synthesize only after the minimal sensor's complete pin regression passes."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from run_units import ROOT,HERE,sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--pins',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();n=a.native.resolve();pin=a.pins.resolve();b=a.root.resolve()
    assert b.is_relative_to(ROOT/'tmp/frame-execution') and not b.exists()
    proof=json.loads((n/'inputs.json').read_text())
    tested=json.loads((pin/'result.json').read_text());pi=json.loads((pin/'inputs.json').read_text())
    assert proof['status']=='SENSOR74_MINIMAL_CANDIDATE_NOT_SYNTHESIZED'
    assert tested['status']=='SENSOR74_MINIMAL_PIN_PASS_NOT_PHYSICAL_RELEASE'
    assert [r['mode'] for r in tested['cases']]==[0,10,4,6,8]
    assert tested['inputs_sha256']==sha(pin/'inputs.json') and pi['prepared_sha256']==sha(n/'inputs.json')
    assert Path(pi['prepared'])==n
    assert proof['new_luts']==proof['new_flip_flops']==proof['new_clocks']==proof['changed_ram_address_pins']==0
    assert proof['script_sha256']==sha(HERE/'prepare_video_sensor74_minimal.py')
    assert proof['source_sha256']==sha(n/'native_source.v')
    assert proof['all_unlisted_source_byte_equal']
    for record in (proof,pi):
        for f,h in record['files'].items():assert sha(f)==h
    parent=ROOT/'tmp/frame-execution/connected-board-video-service-probe60-input-isp-explicit'
    assert proof['original_synthesis_input_sha256']==sha(parent/'native_source.v')
    pm=json.loads((parent/'inputs.json').read_text());b.mkdir();rows=[]
    for row in pm['inputs']:
        old=Path(row['snapshot']);assert sha(old)==row['sha256'];rel=old.relative_to(parent)
        src=n/'native_source.v' if rel.as_posix()=='native_source.v' else old
        dst=b/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        rows.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
    shutil.copy2(parent/'map.cmd',b/'map.cmd')
    meta=dict(inputs=rows,parent=str(parent),native=str(n),prototype_pin_test=str(pin),
              prototype_pin_test_sha256=sha(pin/'result.json'),builder_sha256=sha(__file__),
              original_synthesis_input_sha256=sha(parent/'native_source.v'),
              original_sdc_unchanged=True,hardware_access=False,board_release=False)
    (b/'inputs.json').write_text(json.dumps(meta,indent=2))
    subprocess.run([sys.executable,str(HERE/'map_video_service_candidate.py'),'--native',str(n),'--root',str(b),'--run-existing'],check=True)


if __name__=='__main__':main()
