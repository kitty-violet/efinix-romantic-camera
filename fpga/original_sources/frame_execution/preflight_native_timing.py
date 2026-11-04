"""Compile both generated wrappers before the single route comparison."""
import json
from run_units import ROOT,sha
import reroute_native_timing as r

if __name__=='__main__':
    root=ROOT/'tmp/board-test/native-timing-preflight1';assert not root.exists();root.mkdir()
    source,code=r.driver_code();compile(code,'native-timing-wrapper','exec')
    prefix=code[:code.index("if __name__=='__main__':")]
    anchor=" root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()"
    assert prefix.count(anchor)==1
    prefix=prefix.replace(anchor,' return code\n'+anchor)
    scope={'__file__':str(source)};exec(prefix,scope)
    inner=scope['main']();compile(inner,'native-timing-driver','exec')
    for gate in ('optimization_level=TIMING_1',"assert selected['current_timing_pass']",
        'audit_native_split_gray.py',"('source_files','copied_files')",'source_map_sha256',
        'CONTROL_DDR_MAPPED_SERIAL_IDLE_RESET_PASS',"int(iterations[-1][1])==0",
        "assert sha(path)==h",'stop_launched_tree(proc)',"'hardware_access': False"):
        if gate=="'hardware_access': False":continue
        assert gate in inner,gate
    (root/'wrapper_preview.py').write_text(code,encoding='utf8')
    (root/'driver_preview.py').write_text(inner,encoding='utf8')
    result=dict(status='NESTED_DRIVER_SYNTAX_AND_ALL_EXISTING_ROUTE_GATES_PASS',
        generated={n:sha(root/n) for n in ('wrapper_preview.py','driver_preview.py')},
        generator_sha256=sha(r.__file__),checker_sha256=sha(__file__),hardware_access=False)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(result['status'])
