"""Freeze actual host deadline and existing freshness test receipts, offline."""
import json,subprocess,sys
from run_units import ROOT,HERE,sha
out=ROOT/'tmp/board-test/camera-reader-budget1';out.mkdir(exist_ok=False)
with (out/'simulation.log').open('x',encoding='utf8') as log:
    r=subprocess.run([sys.executable,'-m','unittest','test_camera_snapshot_reader','test_camera_snapshot_budgets','-v'],cwd=HERE,stdout=log,stderr=subprocess.STDOUT,timeout=30)
assert r.returncode==0,(out/'simulation.log').read_text()
text=(out/'simulation.log').read_text();assert 'Ran 9 tests' in text and text.rstrip().endswith('OK')
files=[HERE/n for n in ('read_camera_checked_snapshot_v2.py','read_camera_checked_snapshot.py','read_camera_checked.py','test_camera_snapshot_budgets.py','test_camera_snapshot_reader.py','verify_camera_reader_budgets.py','windows_process_tree.py')]+[out/'simulation.log']
result=dict(status='CAMERA_READER_SEPARATE_BUDGET_TESTS_PASS',tests=9,evidence={str(p):sha(p) for p in files},hardware_access=False)
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(status=result['status'],tests=9)))
