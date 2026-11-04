"""Real-map preflight: exact binding plus late/mutated-SDC rejection."""
import json,shutil
from run_units import ROOT,HERE,sha
import native_prebound_constraints as check

def main():
    root=ROOT/'tmp/board-test/native-prebound-preflight1';assert not root.exists();root.mkdir()
    fixture=root/'fixture';(fixture/'outflow').mkdir(parents=True)
    for n in ('outflow/ti60f225_oob.map.v','outflow/ti60f225_oob.vdb','ti60f225_oob.sdc'):
        shutil.copy2(check.SOURCE/n,fixture/n)
    check.original_profile=lambda *_:dict(after_sha256=sha(fixture/'ti60f225_oob.sdc'))
    text=(fixture/'ti60f225_oob.sdc').read_text()
    (fixture/'ti60f225_oob.sdc').write_text(text+'\n# unexpected edit\n',encoding='utf8')
    try:check.apply_profile(fixture,root)
    except AssertionError:pass
    else:raise AssertionError('Modified SDC accepted')
    assert not (fixture/'prebound_constraints.json').exists()
    (fixture/'ti60f225_oob.sdc').write_text(text,encoding='utf8')
    (fixture/'outflow/ti60f225_oob.place').write_text('late-binding fixture')
    try:check.apply_profile(fixture,root)
    except AssertionError:pass
    else:raise AssertionError('Late binding accepted')
    assert not (fixture/'prebound_constraints.json').exists()
    (fixture/'outflow/ti60f225_oob.place').unlink()
    profile=check.apply_profile(fixture,root)
    (fixture/'physical_inputs.json').write_text(json.dumps(dict(clock_profile=profile)))
    record=check.audit(fixture)
    assert len(record['pins'])==140 and record['destination_bits']==70
    (root/'result.json').write_text(json.dumps(dict(
        status='PREPLACEMENT_EXACT_BINDING_AND_TWO_REJECTIONS_PASS',
        before_sdc_sha256=sha(check.SOURCE/'ti60f225_oob.sdc'),
        after_sdc_sha256=sha(fixture/'ti60f225_oob.sdc'),
        original_map_sha256=sha(check.SOURCE/'outflow/ti60f225_oob.map.v'),
        checker_sha256=sha(check.__file__),runner_sha256=sha(__file__),
        modified_sdc_rejected=True,late_binding_rejected=True,
        hardware_access=False,physical_run=False),indent=2))
    print('PREPLACEMENT_EXACT_BINDING_AND_TWO_REJECTIONS_PASS',flush=True)

if __name__=='__main__':main()
