"""Bounded implementation of the same-map, small complete-tile repair."""
import json,runpy,sys
from pathlib import Path
from run_units import HERE,ROOT,sha
from control_ddr_combined import once
if __name__=='__main__':
    args=sys.argv[1:];root=Path(args[args.index('--root')+1]).resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    source=HERE/'probe_control_ddr_reuse_physical.py';code=source.read_text()
    code=once(code,'from fixed_control_ddr_reuse_locations import generate as reuse_locations',
        'from fixed_control_ddr_local_locations import generate as reuse_locations')
    wrapper=root.parent/(root.name+'-local-probe-wrapper.py');assert not wrapper.exists()
    compile(code,str(wrapper),'exec');wrapper.write_text(code)
    wrapper.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        generated_sha256=sha(wrapper),adapter_sha256=sha(__file__),placement_sha256=sha(HERE/'fixed_control_ddr_local_locations.py')),indent=2))
    sys.argv=[str(wrapper),*args];runpy.run_path(str(wrapper),run_name='__main__')
