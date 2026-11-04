"""Complete exported-input sensitivity check; retain private vendor warnings.

The previous attempt preserved equivalence but discovered one additional
group after the diagnostic cap was removed. Do not rewrite its FAIL receipt.
"""
import inspect,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once

def main():
    source=HERE/'probe_native_open_inputs.py';code=source.read_text()
    previous=ROOT/'tmp/frame-execution/native-open-inputs1'
    warnings=(previous/'outflow/ti60f225_oob.warn.log').read_text()
    import re
    extra=set(re.findall(r"net '(Open_\d+)' does not have a driver",warnings))
    assert extra=={'Open_1186'}
    code=once(code,"        assert len(reported)==175", "        assert len(reported)==175\n        reported.add('Open_1186')")
    code=once(code,"        assert not remaining and '[EFX-0654' not in warnings,remaining",
        "        assert '[EFX-0654' not in warnings\n"
        "        assert remaining and all(\"in module 'protected_cell'\" in n and 'inst_efx_csi2_rx/' in n for n in remaining),remaining\n"
        "        assert not re.findall(r\"net '(Open_\\d+)' does not have a driver\",warnings)\n")
    code=once(code,"status='ALL_EXPORTED_OPEN_INPUT_CONSTANT_SENSITIVITY_PASS'",
        "status='EXPORTED_OPEN_INPUTS_INSENSITIVE_PRIVATE_VENDOR_WARNINGS_RETAINED'")
    code=once(code,'resource_counts_unchanged=True,protected_boundary=boundary,',
        "resource_counts_unchanged=True,protected_boundary=boundary,private_vendor_warnings=remaining,private_warning_list_may_be_capped=True,")
    out=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    driver=out.parent/(out.name+'-driver.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        driver_sha256=sha(driver),wrapper_sha256=sha(__file__),previous_warning_sha256=sha(previous/'outflow/ti60f225_oob.warn.log'),
        previous_public_audit_sha256=sha(previous/'public_connectivity_audit.json'),hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
