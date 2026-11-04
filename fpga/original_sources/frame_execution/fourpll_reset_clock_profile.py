"""Retain established real PLL-domain constraints for the reset-only variant."""
import json
from pathlib import Path
from run_units import sha
from measured_pinmap_clock_profile import DDR,GROUP
from prepare_four_pll_reset_board import BASE,audit_reset_periphery


def apply_profile(root,proof):
    data=json.loads(proof.read_text());assert data['status']=='CROSS_PLL_ENDPOINT_INVENTORY_NOT_CDC_RELEASE'
    skew_file=proof.parent/'gray_skew_audit.json';skew=json.loads(skew_file.read_text())
    assert skew['status']=='ROUTED_GRAY_SKEW_PASS_NOT_RESET_OR_CDC_RELEASE'
    assert skew['inventory_sha256']==sha(proof) and skew['buses']==10 and skew['endpoints']==70
    assert skew['max_skew_ns']<=skew['limit_ns']==2.5
    for r in data['reports']:assert sha(r['path'])==r['sha256']
    # Exact GPIO corrections were already independently measured. The only
    # additional periphery property is exposing PLL_BR0's LOCKED output.
    delta=audit_reset_periphery(root)
    from board_measured_pinmap import BASE as gpio_parent
    assert sha(gpio_parent/'ti60f225_oob.peri.xml')==data['peri_sha256']
    assert sha(BASE/'ti60f225_oob.sdc')==data['sdc_sha256']
    for name,key in [('ddr516_reference.sv','ddr_source_sha256'),('ddr516_v4_controller.sv','wrapper_sha256')]:
        assert sha(root/name)==data[key]
    sdc=root/'ti60f225_oob.sdc';before=sdc.read_text();before_sha=sha(sdc)
    assert GROUP not in before
    extra='\n# DDR PLL_TR0 and user PLL_BR0 have no guaranteed relative phase.\n'
    extra+='# Retain related DDR200/DDR400 phase requirements and every same-domain check.\n'+GROUP+'\n'
    extra+='# Explicit Gray first-stage bus skew: stricter than the slow 150MHz period.\n'
    for bus in data['gray_buses']:
        starts=' '.join(sorted({a for a,b in bus['pins']}));ends=' '.join(sorted({b for a,b in bus['pins']}))
        extra+=f'set_bus_skew -from [get_pins {{{starts}}}] -to [get_pins {{{ends}}}] 2.500\n'
    sdc.write_text(before+extra,encoding='utf-8');assert sdc.read_text()==before+extra
    return dict(profile='independent_user_ddr_plls',proof=str(proof),proof_sha256=sha(proof),
        before_sha256=before_sha,after_sha256=sha(sdc),related_ddr_clocks_preserved=sorted(DDR),
        added_clock_groups=1,gray_bus_constraints=10,gray_evidence=str(skew_file),gray_evidence_sha256=sha(skew_file),
        current_periphery_sha256=sha(root/'ti60f225_oob.peri.xml'),reset_periphery_audit=delta,
        note='All original timing commands retained; new asynchronous PLL lock signal classified only. New-route reset CDC and Gray review required.',
        cdc_release=False,reset_safety_proven=False)
