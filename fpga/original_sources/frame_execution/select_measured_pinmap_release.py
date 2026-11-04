"""Consolidate four-seed evidence with the prechecked, balanced seed1 candidate."""
import json, shutil
from run_units import ROOT,sha
from audit_canonical_multiseed import audit
from audit_measured_pinout import audit as pinout

def main():
    batch=ROOT/'tmp/frame-execution/measured-pinmap-physical1'
    out=ROOT/'tmp/frame-execution/measured-pinmap-selection1'
    terminal=json.loads((batch/'result.json').read_text())
    assert terminal['status']=='OFFLINE_PHYSICAL_EVIDENCE_COMPLETE_AWAIT_REVIEW'
    rows=[audit(ROOT/'tmp/frame-execution/measured-pinmap-physical1'/f'canonical-seed{i}') for i in range(1,5)]
    assert all(r['current_timing_pass'] for r in rows)
    assert all(r['identity']==rows[0]['identity'] for r in rows)
    chosen=rows[0]
    assert chosen['wns_ns']>=0.14 and chosen['whs_ns']>=0.02
    pinout(ROOT/'tmp/frame-execution/measured-pinmap-physical1/pnr-seed1')
    gray=ROOT/'tmp/frame-execution/measured-pinmap-gray-seed1'
    bit=ROOT/'tmp/frame-execution/measured-pinmap-bit-seed1'
    g=json.loads((gray/'result.json').read_text());gm=json.loads((gray/'inputs.json').read_text())
    b=json.loads((bit/'result.json').read_text());bm=json.loads((bit/'generation_inputs.json').read_text())
    assert g['status']=='NEW_ROUTE_GRAY_SKEW_PASS_NOT_FULL_CDC_RELEASE'
    assert g['inputs_sha256']==sha(gray/'inputs.json') and g['endpoints']==70 and g['max_skew_ns']<2.5
    assert gm['base']==chosen['physical']
    assert b['status']=='OFFLINE_BITSTREAM_GENERATED_NOT_RELEASED'
    assert b['inputs_sha256']==sha(bit/'generation_inputs.json')
    assert bm['base']==chosen['physical'] and bm['canonical']==chosen['canonical']
    assert not out.exists();out.mkdir()
    shutil.copy2(batch/'host-regression3.log',out/'host-regression3.log')
    terminal.update(selected=next(r for r in terminal['seeds'] if r['seed']==1),gray=str(gray),bitstream=str(bit),
                    original_batch=str(batch),original_batch_result_sha256=sha(batch/'result.json'),
                    selection_reason='Seed1 has both setup >=0.140ns and hold >=0.020ns; prefer it over seed3 setup 0.006ns.',
                    consolidation_runner_sha256=sha(__file__))
    (out/'result.json').write_text(json.dumps(terminal,indent=2))
    print(json.dumps(dict(status=terminal['status'],selected=terminal['selected'],output=str(out))))

if __name__=='__main__':main()
