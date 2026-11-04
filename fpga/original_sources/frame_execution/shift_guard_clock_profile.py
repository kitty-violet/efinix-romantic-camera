"""Same verified clock constraints, with the exact PLL public-interface delta."""
import json
from run_units import sha
from measured_pinmap_clock_profile import DDR,GROUP
from prepare_four_pll_reset_board import BASE as RESET_BASE,audit_reset_periphery
from prepare_ddr_shift_guard_board import BASE,CAL,proof

def apply_profile(root,evidence):
 proof()
 data=json.loads(evidence.read_text());assert data['status']=='CROSS_PLL_ENDPOINT_INVENTORY_NOT_CDC_RELEASE'
 skew_file=evidence.parent/'gray_skew_audit.json';skew=json.loads(skew_file.read_text())
 assert skew['status']=='ROUTED_GRAY_SKEW_PASS_NOT_RESET_OR_CDC_RELEASE'
 assert skew['inventory_sha256']==sha(evidence) and skew['buses']==10 and skew['endpoints']==70
 assert skew['max_skew_ns']<=skew['limit_ns']==2.5
 for r in data['reports']:assert sha(r['path'])==r['sha256']
 from board_measured_pinmap import BASE as gpio_parent
 assert sha(gpio_parent/'ti60f225_oob.peri.xml')==data['peri_sha256']
 assert sha(RESET_BASE/'ti60f225_oob.sdc')==data['sdc_sha256']
 assert sha(root/'ddr516_reference.sv')==data['ddr_source_sha256']
 # Retain the original wrapper binding; independently bind its sole repair.
 assert sha(BASE/'ddr516_v4_controller.sv')==data['wrapper_sha256']
 for n in ('ddr516_v4_controller.sv','ddr516_pll_shift_guard.sv'):
  assert sha(root/n)==sha(CAL/n)
 delta=audit_reset_periphery(root)
 sdc=root/'ti60f225_oob.sdc';before=sdc.read_text();before_sha=sha(sdc)
 assert GROUP not in before and sha(sdc)==sha(BASE/sdc.name)
 extra='\n# DDR PLL_TR0 and user PLL_BR0 have no guaranteed relative phase.\n'
 extra+='# Retain related DDR200/DDR400 phase requirements and every same-domain check.\n'+GROUP+'\n'
 extra+='# Explicit Gray first-stage bus skew: stricter than the slow 150MHz period.\n'
 for bus in data['gray_buses']:
  starts=' '.join(sorted({a for a,b in bus['pins']}));ends=' '.join(sorted({b for a,b in bus['pins']}))
  extra+=f'set_bus_skew -from [get_pins {{{starts}}}] -to [get_pins {{{ends}}}] 2.500\n'
 sdc.write_text(before+extra,encoding='utf-8');assert sdc.read_text()==before+extra
 return dict(profile='independent_user_ddr_plls',proof=str(evidence),proof_sha256=sha(evidence),before_sha256=before_sha,after_sha256=sha(sdc),related_ddr_clocks_preserved=sorted(DDR),added_clock_groups=1,gray_bus_constraints=10,gray_evidence=str(skew_file),gray_evidence_sha256=sha(skew_file),current_periphery_sha256=sha(root/'ti60f225_oob.peri.xml'),reset_periphery_audit=delta,
  pll_shift_delta=dict(old_wrapper_sha256=data['wrapper_sha256'],new_wrapper_sha256=sha(root/'ddr516_v4_controller.sv'),guard_sha256=sha(root/'ddr516_pll_shift_guard.sv'),vendor_result_sha256=sha(CAL/'result.json')),
  note='Same original clock relationships and Gray constraints; only the verified PLL shift setup wrapper differs.',cdc_release=False,reset_safety_proven=False)
