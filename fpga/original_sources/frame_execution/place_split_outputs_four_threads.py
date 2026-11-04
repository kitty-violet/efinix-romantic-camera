"""Same native placement gates with a recorded official four-thread setting."""
import json,runpy,sys
from pathlib import Path
from run_units import HERE,sha

def main():
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    assert not root.exists()
    source=HERE/'probe_local_decode_place.py'
    code=source.read_text()
    before='from local_control_decode import audit_delta'
    assert code.count(before)==1
    code=code.replace(before,'from split_logic_outputs_board import audit_delta')
    before="source=HERE/'probe_fourpll_reset_physical.py';code=source.read_text()"
    inject='''# Restrict only the official P&R worker count, before physical input hashing.
xml=root/'ti60f225_oob.xml';before_threads=xml.read_text()
needle='<efx:param name="max_threads" value="-1" value_type="e_integer" />'
assert before_threads.count(needle)==1
after_threads=before_threads.replace(needle,needle.replace('value="-1"','value="4"'))
xml.write_text(after_threads,encoding='utf8')
for rec in records:
 if Path(rec['snapshot'])==xml:rec['sha256']=sha(xml)
(root/'pnr_threads.json').write_text(json.dumps(dict(official_max_threads=4,
 before_xml_sha256=sha(base/xml.name),after_xml_sha256=sha(xml),
 scope='Host P&R workers only; no RTL, map, clock, pin or constraint change'),indent=2))
optimization_change=None'''
    assert code.count(before)==1
    code=code.replace(before,before+"\n    code=once(code,'optimization_change=None',"+repr(inject)+")")
    driver=root.parent/(root.name+'-four-thread-wrapper.py')
    assert not driver.exists()
    compile(code,str(driver),'exec')
    driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),
        source_sha256=sha(source),wrapper_sha256=sha(__file__),driver_sha256=sha(driver),
        max_threads=4,hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]]
    runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
