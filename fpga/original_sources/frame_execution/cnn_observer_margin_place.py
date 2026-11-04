"""Placement-only counterpart to the measured process-domain routing deficit."""
import contextlib,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once
from audit_legacy_cache_recovery import install
from cnn_observer_process_margin import EXTRA,PROBE,verify_probe
install()

@contextlib.contextmanager
def optimization_scope(root,flow):
    if flow!='pnr':
        yield
        return
    proof=verify_probe()
    sdc=root/'ti60f225_oob.sdc';original=sdc.read_bytes()
    canonical=root/'placement_acceptance.sdc';optimized=root/'placement_optimization.sdc'
    assert not canonical.exists() and not optimized.exists()
    canonical.write_bytes(original);optimized.write_text(sdc.read_text()+EXTRA,encoding='utf8')
    receipt=root/'placement_optimization_inputs.json'
    row=dict(status='PLACEMENT_OPTIMIZATION_RUNNING',extra=EXTRA,canonical_sha256=sha(canonical),optimization_sha256=sha(optimized),
             proof=str(PROBE/'result.json'),proof_sha256=sha(PROBE/'result.json'),adapter_sha256=sha(__file__),
             hardware_clock_changed=False,ordinary_anchors=0,full_original_sta_required=True)
    def save():
        receipt.write_text(json.dumps(row,indent=2),encoding='utf8')
        p=root/'physical_inputs.json';m=json.loads(p.read_text())
        assert m['setup_margin'] is None
        m['placement_optimization_inputs_sha256']=sha(receipt)
        p.write_text(json.dumps(m,indent=2),encoding='utf8')
    save()
    sdc.write_bytes(optimized.read_bytes())
    try:yield
    finally:
        sdc.write_bytes(original);assert sha(sdc)==row['canonical_sha256']
        row['status']='ORIGINAL_SDC_RESTORED_AFTER_PLACEMENT';save()

def driver_code():
    from cnn_minimal_repack import driver_code as parent
    source,code=parent('free-arena')
    code=once(code,'from cnn_minimal_board import BOARD,audit_delta','from cnn_observer_trim_board import BOARD,audit_delta')
    needle="    driver=root.parent/(root.name+'-place-driver.py');assert not driver.exists();driver.parent.mkdir(parents=True,exist_ok=True)"
    extra="\n    code=once(code,\"  with (root/(flow+'.log')).open('w') as log:\",\"  from cnn_observer_margin_place import optimization_scope\\n  with optimization_scope(root,flow),(root/(flow+'.log')).open('w') as log:\")"
    code=once(code,needle,extra+'\n'+needle)
    compile(code,__file__+':placement-only','exec')
    return source,code

def main():
    source,code=driver_code()
    if sys.argv[1:]==['--check-only']:
        print('PROCESS_MARGIN_PLACEMENT_ONLY_GENERATOR_COMPILES');return
    assert not sys.argv[1:]
    verify_probe()
    from cnn_observer_trim_board import BOARD,audit_delta
    audit_delta(BOARD)
    base=ROOT/'tmp/frame-execution';root=base/'cnn-observer-margin-place1'
    driver=root.parent/(root.name+'-place-wrapper.py')
    assert not root.exists() and not driver.exists()
    driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),driver_sha256=sha(driver),
        wrapper_sha256=sha(__file__),probe_sha256=sha(PROBE/'result.json'),ordinary_anchors=0,hard_anchors=401,seed=3,
        host_threads=4,route_started=False,hardware_access=False,
        rationale='Same mapped full system; measured 64 negative process-domain endpoints and -.845ns routed deficit. Tighten process setup optimization by .900ns only before packing/placement; do not tighten other domains or hold. Restore original SDC byte-for-byte and review full congestion before route.'),indent=2))
    sys.argv=[str(driver),'--root',str(root),'--base',str(BOARD),'--cnn-ram-reference',str(base/'native-local-driver-route1/canonical-seed3'),
        '--pnr-budget','1800','--logic-limit','60800','--fixed-dsp-layout','--clock-profile-proof',str(base/'axi-valid-payload-physical1/clock_crossing_inventory.json'),
        '--optimization','CONGESTION_2','--seed','3']
    runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
