"""C1-placement/C2-router control, preserving all existing release gates."""
import json,runpy,sys
from pathlib import Path
from audit_legacy_cache_recovery import install
install()
from control_command_repair import once
from run_units import ROOT,sha
import cnn_minimal_route as parent


def adapt(fn):
    source,code=fn()
    code=once(code,"{'before':'CONGESTION_2','after':'CONGESTION_2'}",
              "{'before':'CONGESTION_1','after':'CONGESTION_2'}")
    return source,code


def prepare_code():return adapt(parent.prepare_code)
def route_code():return adapt(parent.route_code)


def main():
    action=sys.argv.pop(1)
    if action=='--check-only':
        for name,fn in [('prepare',prepare_code),('route',route_code)]:compile(fn()[1],name,'exec')
        print('C1_PLACE_C2_ROUTE_WITH_ORIGINAL_GATES_CHECKED');return
    if action=='prepare':
        source,code=prepare_code();review=Path(sys.argv[sys.argv.index('--review')+1]).resolve()
        r=json.loads(review.read_text());root=Path(r['physical']);suffix='-prepare-cnn-budget'
        # Establish the actual placement recipe instead of relabelling C2.
        xml=(Path(r['placement'])/'ti60f225_oob.xml').read_text()
        assert '<efx:param name="optimization_level" value="CONGESTION_1"' in xml
    else:
        assert action=='route';source,code=route_code()
        root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();suffix='-local-driver-route'
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    driver=root.parent/(root.name+suffix+'.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        generated_sha256=sha(driver),driver_sha256=sha(driver),adapter_sha256=sha(__file__),
        wrapper_sha256=sha(__file__),hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')


if __name__=='__main__':main()
