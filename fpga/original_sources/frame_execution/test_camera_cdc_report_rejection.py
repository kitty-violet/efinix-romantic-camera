"""Reject unknown crossings, missing payload paths and excessive settling.

Read-only mutations in memory; immutable failed-route evidence is not edited.
"""
import json,re
from pathlib import Path
from unittest.mock import patch
from run_units import ROOT,HERE,sha
from camera_checked_cdc_v2 import inventory

def main():
    root=ROOT/'tmp/frame-execution/camera-failed-route-gray1'
    dest=ROOT/'tmp/board-test/camera-cdc-report-rejection1.json';assert not dest.exists()
    reference=json.loads((root/'camera_inventory_reaudit.json').read_text())
    original=Path.read_text;results=[]
    held=reference['camera_structure']['held'][0]['end']
    def change_unknown(text):
        a='cell_probe/ddr_cells/release_pipe[0]~FF|SR';assert a in text
        return text.replace(a,'unreviewed/foreign_ff~FF|D')
    def change_missing(text):
        parts=text.split('++++ Path ');selected=[i for i,p in enumerate(parts) if 'Path End' in p and held in p]
        assert len(selected)==1;parts.pop(selected[0]);return '++++ Path '.join(parts)
    def change_delay(text):
        parts=text.split('++++ Path ');selected=[i for i,p in enumerate(parts) if 'Path End' in p and held in p]
        assert len(selected)==1
        i=selected[0];parts[i],n=re.subn(r'(^Data Path Delay\s*:\s*)[\d.]+',r'\g<1>14.000',parts[i],flags=re.M);assert n==1
        return '++++ Path '.join(parts)
    cases=[('unknown_ddr_endpoint','ti60f225_oob.user_to_ddr_crossings.rpt',change_unknown),
           ('missing_camera_endpoint','ti60f225_oob.camera_cfg_to_system.rpt',change_missing),
           ('excess_camera_settling','ti60f225_oob.camera_cfg_to_system.rpt',change_delay)]
    for name,filename,transform in cases:
        target=root/'outflow'/filename;body=transform(original(target));calls=[]
        def read(path,*args,**kwargs):
            if path==target:calls.append(path);return body
            return original(path,*args,**kwargs)
        try:
            with patch.object(Path,'read_text',read):inventory(root)
        except AssertionError as exc:
            assert calls;results.append(dict(case=name,rejected=True,error=repr(exc)))
        else:raise AssertionError(('Unsafe evidence accepted',name))
    evidence={str(root/'camera_inventory_reaudit.json'):sha(root/'camera_inventory_reaudit.json'),
        str(HERE/'camera_checked_cdc.py'):sha(HERE/'camera_checked_cdc.py'),str(HERE/'camera_checked_cdc_v2.py'):sha(HERE/'camera_checked_cdc_v2.py'),
        str(Path(__file__).resolve()):sha(__file__)}
    evidence.update(reference['camera_bundled_settling']['reports'])
    for r in reference['reports']:evidence[r['path']]=r['sha256']
    for f,h in evidence.items():assert sha(f)==h
    dest.write_text(json.dumps(dict(status='CDC_REPORT_NEGATIVE_GUARDS_PASS',cases=results,evidence=evidence,hardware_access=False),indent=2))
    print('CDC_REPORT_NEGATIVE_GUARDS_PASS',len(results))

if __name__=='__main__':main()
