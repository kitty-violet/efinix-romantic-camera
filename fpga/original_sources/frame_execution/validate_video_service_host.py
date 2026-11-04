"""Record the current readout/control tests and exact sources, without USB."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from run_units import ROOT,HERE,sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 out=a.root.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists();out.mkdir()
 names=['video_service_control','video_service_observation','read_video_service_pages','video_service_board','test_video_service_board','test_video_service_control','test_video_service_observation','test_video_service_pages','test_video_service_words','test_video_service_compact']
 files={str(HERE/(n+'.py')):sha(HERE/(n+'.py')) for n in names}
 tests=[n for n in names if n.startswith('test_')];command=[sys.executable,'-m','unittest',*tests]
 start=time.monotonic();r=subprocess.run(command,cwd=HERE,capture_output=True,text=True,timeout=30)
 log=r.stdout+r.stderr;(out/'test.log').write_text(log)
 result=dict(status='PASS' if r.returncode==0 and 'Ran 59 tests' in log else 'FAIL',command=command,exit_code=r.returncode,
  seconds=time.monotonic()-start,files=files,log_sha256=sha(out/'test.log'),hardware_access=False,board_release=False)
 assert all(sha(f)==h for f,h in files.items())
 (out/'result.json').write_text(json.dumps(result,indent=2));print(result['status'],log)
 assert result['status']=='PASS'
if __name__=='__main__':main()
