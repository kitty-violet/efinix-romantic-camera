"""Bounded input-ISP test, no hardware. Table generation is explicit and reproducible."""
import argparse,json,os,shutil,time,re,zlib
from pathlib import Path
from run_units import ROOT,HERE,run,sha

def tables(root):
    # Neutral-channel gamma 2.2; no guessed white-balance gain from bad CFA data.
    values=bytes(int(255*(i/255)**(1/2.2)+0.5) for i in range(256))
    for c in 'rgb':(root/f'isp_{c}.hex').write_text(''.join(f'{v:02x}\n' for v in values))
    return dict(gamma=2.2,gains=[1,1,1],black_subtraction=0,default_rgb_swap=True,
                crc32=f'{zlib.crc32(values*3):08x}',sha256={f.name:sha(f) for f in root.glob('isp_*.hex')},
                optical_acceptance=False,scope='camera input before 1080-to-720; not A-output styling')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    records=[]
    for n in ['video_input_lut.sv','tb_video_input_lut.sv']:
        f=HERE/'variants/video_service'/n;shutil.copy2(f,root/n);records.append(dict(path=str(f),sha256=sha(f)))
    meta=dict(sources=records,tables=tables(root),runner_sha256=sha(__file__))
    (root/'inputs.json').write_text(json.dumps(meta,indent=2))
    ms=Path('E:/quatra/questa_fse/win64');os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
    start=time.monotonic()
    try:
        run([ms/'vlib.exe','work'],root,'vlib.log',20)
        run([ms/'vlog.exe','-sv','video_input_lut.sv','tb_video_input_lut.sv'],root,'compile.log',40)
        body,_=run([ms/'vsim.exe','-c','work.tb_video_input_lut','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',40)
        assert 'PASS INPUT_LUT' in body
        result=dict(status='PASS',lines=re.findall(r'^# PASS.*$',body,re.M))
    except Exception as e:result=dict(status='FAIL',error=str(e))
    result.update(seconds=time.monotonic()-start,inputs_sha256=sha(root/'inputs.json'),hardware_access=False)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    return result['status']=='PASS'
if __name__=='__main__':raise SystemExit(0 if main() else 1)
