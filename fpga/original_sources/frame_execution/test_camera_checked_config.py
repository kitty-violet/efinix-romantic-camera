"""Bounded pin-level SC431HAI tests. No hardware access or environment installation."""
import argparse,concurrent.futures,hashlib,json,os,re,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--modes',default='0,1,2,3,4,5,6,7,8,9');a=p.parse_args()
 out=a.out.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists();out.mkdir(parents=True)
 old=ROOT/'tmp/frame-execution/connected-board-local-drivers1/rtl/cam_i2c_ctrl'
 files=[HERE/'camera_checked_config.sv',HERE/'tb_camera_checked_config.sv',old/'i2c/i2c_master_reg_rom.v',old/'i2c_master/i2c_master_byte_ctrl.v',old/'i2c_master/i2c_master_bit_ctrl.v']
 for f in files:shutil.copy2(f,out/f.name)
 for f in (old/'i2c_master').glob('*.v'):
  if f.name in ('timescale.v','i2c_master_defines.v'):shutil.copy2(f,out/f.name)
 rom=re.sub(r'/\*.*?\*/|//[^\n]*','',files[2].read_text(encoding='utf-8-sig'),flags=re.S)
 rows=re.findall(r"ROM\[8'h([0-9a-f]+)\]\s*=\s*\{16'h([0-9a-f]+),\s*8'h([0-9a-f]+),\s*1'b([01])\}",rom,re.I)
 assert len(rows)==164 and all(int(x[3])==0 for x in rows)
 (out/'expected.hex').write_text(''.join(f'60{r[1]}{r[2]}\n' for r in rows))
 compiler=ROOT/'tools/iverilog/ucrt64/bin/iverilog.exe'
 env=os.environ.copy();env['PATH']=str(compiler.parent)+os.pathsep+env.get('PATH','');env['TEMP']=env['TMP']=env['TMPDIR']=str(out)
 inputs={str(f):sha(f) for f in files};inputs[str(Path(__file__))]=sha(__file__)
 for speed in (4,32):
  cmd=[str(compiler),'-g2012','-I',str(out),'-s','tb_camera_checked_config',f'-Ptb_camera_checked_config.PRESCALE={speed}','-o',f'sim{speed}.vvp',*[f.name for f in files]]
  r=subprocess.run(cmd,cwd=out,env=env,capture_output=True,text=True,timeout=30)
  (out/f'compile{speed}.log').write_text(r.stdout+r.stderr,encoding='utf-8');assert r.returncode==0,(r.returncode,r.stdout,r.stderr)
 def case(mode):
  started=time.monotonic();speed=32 if mode==0 else 4
  r=subprocess.run([str(compiler.with_name('vvp.exe')),f'sim{speed}.vvp',f'+MODE={mode}'],cwd=out,env=env,capture_output=True,text=True,timeout=65)
  output=r.stdout+r.stderr;(out/f'mode{mode}.log').write_text(output,encoding='utf-8')
  return dict(mode=mode,prescale=speed,seconds=round(time.monotonic()-started,3),passed=r.returncode==0 and 'PASS CHECKED_CAMERA' in output,output=output)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:cases=list(pool.map(case,map(int,a.modes.split(','))))
 for f,h in inputs.items():assert sha(f)==h
 result=dict(status='CHECKED_CAMERA_PIN_TESTS_PASS' if all(x['passed'] for x in cases) else 'FAIL',hardware_access=False,inputs=inputs,cases=cases)
 (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
 assert result['status']=='CHECKED_CAMERA_PIN_TESTS_PASS'
if __name__=='__main__':main()
