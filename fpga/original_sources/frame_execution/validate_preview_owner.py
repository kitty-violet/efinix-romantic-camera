"""Bounded 512+3 frame owner/admission and display-drain integration."""
import argparse,json,os,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
from continuous_control_native import PARENT,VARIANT

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    files=[PARENT/'assembly/frame_pair_owner.sv',VARIANT/'cnn_frame_admission.sv',VARIANT/'tb_preview_owner.sv']
    records={str(f):sha(f) for f in [Path(__file__),*files]}
    for f in files:shutil.copy2(f,root/f.name);records[str(root/f.name)]=sha(root/f.name)
    (root/'inputs.json').write_text(json.dumps(records,indent=2))
    q=Path('E:/quatra/questa_fse/win64');env=dict(os.environ);env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
    try:
        for name,cmd in [('vlib',[q/'vlib.exe','work']),('compile',[q/'vlog.exe','-sv','-timescale','1ns/1ps',*root.glob('*.sv')]),
                         ('simulate',[q/'vsim.exe','-c','work.tb_preview_owner','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'])]:
            with (root/(name+'.log')).open('w') as out:
                done=subprocess.run(list(map(str,cmd)),cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=60)
            body=(root/(name+'.log')).read_text()
            assert done.returncode==0 and not re.search(r'\*\* (Error|Fatal)',body),(name,body[-2500:])
        body=(root/'simulate.log').read_text();markers=re.findall(r'^# (PASS PREVIEW_OWNER.*)$',body,re.M);assert len(markers)==1
        for f,h in records.items():assert sha(f)==h,f
        r=dict(status='CONTINUOUS_OWNER_ADMISSION_DRAIN_PASS',markers=markers,inputs_sha256=sha(root/'inputs.json'),
               log_sha256=sha(root/'simulate.log'),scope='Real current owner plus admission; capture/compute/display handshake agents, no image numeric or PHY claim',hardware_access=False,board_release=False)
        (root/'result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
    except Exception as e:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
