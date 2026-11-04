"""Map the proved dangling-cell cleanup with the unchanged sensor pin evidence."""
import json,shutil,subprocess,sys
from pathlib import Path
from run_units import ROOT,HERE,sha

def main():
    n=ROOT/'tmp/frame-execution/video-sensor75-pruned1'
    b=ROOT/'tmp/frame-execution/connected-board-video-sensor75-map1'
    parent=ROOT/'tmp/frame-execution/connected-board-video-service-probe60-input-isp-explicit'
    proof=json.loads((n/'inputs.json').read_text())
    assert proof['status']=='SENSOR75_DANGLING_CELLS_SOURCE_PROVED_NOT_SYNTHESIZED'
    assert proof['script_sha256']==sha(HERE/'prepare_video_sensor75.py')
    assert proof['source_sha256']==sha(n/'native_source.v') and proof['no_surviving_loads'] and proof['existing_camera_689_cells_unchanged']
    for field in ('files','evidence'):
        for f,h in proof[field].items():assert sha(f)==h,f
    pin=Path(proof['prototype_test']);pr=json.loads((pin/'result.json').read_text())
    assert sha(pin/'result.json')==proof['prototype_test_sha256'] and pr['status']=='SENSOR74_MINIMAL_PIN_PASS_NOT_PHYSICAL_RELEASE'
    assert [r['mode'] for r in pr['cases']]==[0,10,4,6,8] and all(r['exit_code']==0 for r in pr['cases'])
    assert not b.exists();b.mkdir();rows=[]
    for row in json.loads((parent/'inputs.json').read_text())['inputs']:
        old=Path(row['snapshot']);assert sha(old)==row['sha256'];rel=old.relative_to(parent)
        src=n/'native_source.v' if rel.as_posix()=='native_source.v' else old
        dst=b/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        rows.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
    shutil.copy2(parent/'map.cmd',b/'map.cmd')
    (b/'inputs.json').write_text(json.dumps(dict(inputs=rows,parent=str(parent),native=str(n),prototype_pin_test=str(pin),prototype_pin_test_sha256=sha(pin/'result.json'),builder_sha256=sha(__file__),original_synthesis_input_sha256=sha(parent/'native_source.v'),original_sdc_unchanged=True,hardware_access=False,board_release=False),indent=2))
    subprocess.run([sys.executable,str(HERE/'map_video_service_candidate.py'),'--native',str(n),'--root',str(b),'--run-existing'],check=True)

if __name__=='__main__':main()
