"""Actual mapped serial regression with the host's paired reply scans."""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import ROOT,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-driver',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();source=a.source_driver.resolve()
    assert root.parent==ROOT/'tmp/frame-execution' and not root.exists()
    code=source.read_text()
    assert code.count(' reg found;')==1
    code=code.replace(' reg found;',' reg found;reg [385:0] first_reply;')
    before=' scan(1,36,raw);\n if(raw[2]===1\'b1)begin'
    after=' scan(1,36,first_reply);scan(1,36,raw);\n if(first_reply[2]===1\'b1 && first_reply[35:2]===raw[35:2])begin'
    assert code.count(before)==1;code=code.replace(before,after)
    driver=root.parent/(root.name+'-driver.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code)
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),wrapper_sha256=sha(__file__)),indent=2))
    sys.argv=[str(driver),root.name,'new']
    try:runpy.run_path(str(driver),run_name='__main__')
    except SystemExit as e:
        if e.code not in (None,0):raise
    r=json.loads((root/'new/result.json').read_text());assert r['status']=='MAPPED_SERIAL_APB_OFS_RESET_PASS'
    result=dict(status='MAPPED_PAIRED_REPLY_STARTUP_FAULT_RESET_PASS',source_map_sha256=json.loads((root/'new/inputs.json').read_text())['source_map_sha256'],result_sha256=sha(root/'new/result.json'),hardware_access=False)
    (root/'paired_reply_result.json').write_text(json.dumps(result,indent=2));print(result['status'])

if __name__=='__main__':main()
