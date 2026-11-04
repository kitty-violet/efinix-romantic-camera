"""Pack/place stop only; ordinary logic stays free, no routing/bitstream."""
import json,runpy,sys
from pathlib import Path
from run_units import HERE,sha
from control_command_repair import once

def main():
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();assert not root.exists()
    source=HERE/'probe_local_decode_place.py';code=source.read_text()
    code=once(code,'from local_control_decode import audit_delta','from registered_boundary_board import audit_delta')
    inject='''xml=root/'ti60f225_oob.xml';before_threads=xml.read_text()
needle='<efx:param name="max_threads" value="-1" value_type="e_integer" />'
assert before_threads.count(needle)==1
xml.write_text(before_threads.replace(needle,needle.replace('value="-1"','value="8"')),encoding='utf8')
for rec in records:
 if Path(rec['snapshot'])==xml:rec['sha256']=sha(xml)
(root/'pnr_threads.json').write_text(json.dumps(dict(official_max_threads=8,
 before_xml_sha256=sha(base/xml.name),after_xml_sha256=sha(xml),
 scope='Host P&R workers only; unchanged RTL/map/clocks/pins/SDC'),indent=2))
optimization_change=None'''
    anchor="source=HERE/'probe_fourpll_reset_physical.py';code=source.read_text()"
    code=once(code,anchor,anchor+"\n    code=once(code,'optimization_change=None',"+repr(inject)+")")
    driver=root.parent/(root.name+'-place-wrapper.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        driver_sha256=sha(driver),wrapper_sha256=sha(__file__),max_threads=8,
        native_pack_place_only=True,no_generic_logic_locked=True,hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
