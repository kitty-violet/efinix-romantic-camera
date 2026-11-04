"""Check native release adapters and reject a known failing STA fixture."""
import json,os,subprocess,sys
from run_units import ROOT,HERE,sha
import finish_native_split_release as release

def main():
    root=ROOT/'tmp/board-test/native-package-preflight1';assert not root.exists();root.mkdir()
    files={}
    for name,provider in (('finish',release.finish_code),('bundle',release.package_code)):
        source,code=provider();p=root/(name+'_preview.py');compile(code,str(p),'exec')
        p.write_text(code,encoding='utf8');files[name]=dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(p))
    body=(root/'bundle_preview.py').read_text()
    for gate in ('current_timing_pass','warning_audit(board)','command_sync',
                 "g['endpoints']==70",'snapshot_cdc(gray)','functional_cdc(gray)',
                 "b['setup_lower_ns']>=2.5",'required_evidence=evidence',
                 'user_supervision_required=True','flash_write_allowed=False',
                 'extra_evidence(board,batch,physical,serial,bind,verify)',
                 'native_transform_sha256','Ran 40 tests','OFP5','1026位'):
        assert gate in body,gate
    assert 'registered_datapath_boundaries=True' not in body
    fixture=root/'failed-route';fixture.mkdir()
    (fixture/'result.json').write_text(json.dumps(dict(status='ORIGINAL_STA_FAIL',wns_ns=-.001)))
    out=root/'must-not-exist';env=os.environ.copy();env['PYTHONPATH']=str(HERE)
    cmd=[sys.executable,'-B',str(root/'finish_preview.py'),'--board',str(release.BOARD),
         '--batch',str(fixture),'--paired',str(release.SERIAL),'--root',str(out),
         '--output',str(ROOT/'output/deployment/must-not-exist-native-preflight')]
    r=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=15)
    (root/'reject.log').write_text(r.stdout+r.stderr,encoding='utf8')
    assert r.returncode!=0 and 'AssertionError' in r.stderr and not out.exists()
    result=dict(status='NATIVE_RELEASE_STATIC_AND_FAILED_STA_REJECTION_PASS',files=files,
        rejection_log_sha256=sha(root/'reject.log'),checker_sha256=sha(__file__),
        release_script_sha256=sha(release.__file__),bitstream_generated=False,hardware_access=False)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(result['status'])

if __name__=='__main__':main()
