"""Persist finite no-USB host-reader regression evidence."""
import json,subprocess,sys
from run_units import ROOT,HERE,sha
def main():
 root=ROOT/'tmp/board-test/camera-reader-offline1';assert not root.exists();root.mkdir()
 files=['test_camera_snapshot_reader.py','read_camera_checked_snapshot.py','read_camera_checked.py','read_startup_cause.py','read_first_fault.py']
 evidence={str(HERE/n):sha(HERE/n) for n in files}
 r=subprocess.run([sys.executable,'-B','-m','unittest','test_camera_snapshot_reader','-v'],cwd=HERE,capture_output=True,text=True,timeout=30)
 f=root/'simulation.log';f.write_text(r.stdout+r.stderr);assert r.returncode==0 and 'Ran 5 tests' in f.read_text() and f.read_text().rstrip().endswith('OK')
 for p,h in evidence.items():assert sha(p)==h
 result=dict(status='CAMERA_HOST_FRESHNESS_TIMEOUT_FORMAT_PASS',tests=5,evidence={**evidence,str(f):sha(f),str(__file__):sha(__file__)},hardware_access=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(result['status'])
if __name__=='__main__':main()
