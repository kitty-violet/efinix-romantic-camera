"""Stricter optimization for routing only; exact original SDC restored always."""
import contextlib,json
from pathlib import Path
from run_units import sha
from cnn_observer_process_margin import BASE,EXTRA,PROBE,verify_probe

@contextlib.contextmanager
def optimization_scope(physical):
    physical=Path(physical);proof=verify_probe()
    sdc=physical/'ti60f225_oob.sdc';original=sdc.read_bytes()
    # This experiment is only the already surveyed C2 placement, not a
    # blanket relaxation or a transfer of the probe to a different netlist.
    assert original==(BASE/'ti60f225_oob.sdc').read_bytes()
    geometry={}
    for suffix in ('.map.v','.place','.vdb'):
        name='outflow/ti60f225_oob'+suffix
        assert sha(physical/name)==sha(BASE/name)
        geometry[str(physical/name)]=sha(physical/name)
    acceptance=physical/'routing_acceptance.sdc';optimized=physical/'routing_optimization.sdc'
    receipt=physical/'routing_goal_inputs.json'
    assert not any(p.exists() for p in (acceptance,optimized,receipt))
    acceptance.write_bytes(original);optimized.write_text(sdc.read_text()+EXTRA,encoding='utf8')
    row=dict(status='ROUTING_OPTIMIZATION_RUNNING',extra=EXTRA,
             canonical_sha256=sha(acceptance),optimization_sha256=sha(optimized),
             proof=str(PROBE/'result.json'),proof_sha256=sha(PROBE/'result.json'),
             adapter_sha256=sha(__file__),geometry=geometry,hardware_clock_changed=False,
             acceptance_rules_changed=False,full_original_sta_required=True)
    receipt.write_text(json.dumps(row,indent=2));sdc.write_bytes(optimized.read_bytes())
    try:yield
    finally:
        sdc.write_bytes(original)
        assert sdc.read_bytes()==original
        row['status']='ORIGINAL_SDC_RESTORED_AFTER_ROUTING'
        receipt.write_text(json.dumps(row,indent=2))

def verify_restored(physical):
    physical=Path(physical);p=physical/'routing_goal_inputs.json';r=json.loads(p.read_text())
    assert r['status']=='ORIGINAL_SDC_RESTORED_AFTER_ROUTING' and r['extra']==EXTRA
    assert not r['hardware_clock_changed'] and not r['acceptance_rules_changed']
    assert r['adapter_sha256']==sha(__file__) and r['proof_sha256']==sha(PROBE/'result.json')
    proof=verify_probe()
    acceptance=physical/'routing_acceptance.sdc';optimized=physical/'routing_optimization.sdc'
    assert sha(acceptance)==r['canonical_sha256'] and sha(optimized)==r['optimization_sha256']
    assert (physical/'ti60f225_oob.sdc').read_bytes()==acceptance.read_bytes()==(BASE/'ti60f225_oob.sdc').read_bytes()
    assert optimized.read_text()==acceptance.read_text()+EXTRA
    for f,h in r['geometry'].items():assert sha(f)==h,f
    return {**proof['evidence'],**r['geometry'],**{str(f):sha(f) for f in (p,acceptance,optimized,PROBE/'result.json',Path(__file__).resolve())}}
