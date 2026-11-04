"""Preserve the released clock model and add only the asynchronous SW0 input."""
import json,re,xml.etree.ElementTree as ET
from run_units import ROOT,sha
from complete_native_logic_split import protected_blocks

PARENT=ROOT/'tmp/frame-execution/connected-board-continuous-srl-register1'
PLACED=ROOT/'tmp/frame-execution/continuous-srl-register-place1'
CANONICAL=ROOT/'tmp/frame-execution/continuous-srl-register-route1/canonical-seed3'

def periphery_audit(root):
 original=(PARENT/'ti60f225_oob.peri.xml').read_text();candidate=(root/'ti60f225_oob.peri.xml').read_text()
 gpio=re.search(r'<efxpt:comp_gpio name="i_arstn".*?</efxpt:comp_gpio>',original,re.S)[0]
 new=gpio.replace('i_arstn','service_key_n').replace('GPIOR_22','GPIOL_07')
 assert candidate==original.replace(gpio,gpio+'\n'+new,1),'Unexpected periphery delta'
 assert 'GPIOL_07' not in original and candidate.count('GPIOL_07')==1
 return dict(existing_configuration_byte_preserved=True,added_input='service_key_n SW0 C4/GPIOL_07',clocks_and_existing_pins_unchanged=True)

def apply_profile(root,evidence):
 delta=periphery_audit(root)
 before=root/'ti60f225_oob.sdc';assert sha(before)==sha(PARENT/before.name)
 # Reuse the complete pre-placement canonical model from the actual released
 # implementation, including all ten nonempty Gray constraints and the exact
 # CLK_5M/process clock relationship. Do not copy routed endpoint spellings.
 meta=json.loads((PLACED/'physical_inputs.json').read_text())
 original=(PLACED/'ti60f225_oob.sdc').read_text()
 entry=next(r for r in meta['inputs'] if r['snapshot']==str(PLACED/'ti60f225_oob.sdc'))
 assert sha(PLACED/'ti60f225_oob.sdc')==entry['sha256']
 # Same opaque INPUT and tool flow as the released implementation. Reuse
 # its expected native paths before place; require current tool nonempty
 # checks again. Re-encrypted output bytes are not an equivalence proof.
 assert protected_blocks((root/'native_source.v').read_text())==protected_blocks((PARENT/'native_source.v').read_text())
 bound=(CANONICAL/'ti60f225_oob.sdc').read_text()
 assert re.sub(r'(?:protected_inst/)+','',bound)==original,'Changed timing values beyond native name binding'
 old_result=json.loads((CANONICAL/'result.json').read_text())
 assert old_result['status']=='CANONICAL_ROUTED_TIMING_PASS_NOT_RELEASE'
 old_inputs=json.loads((CANONICAL/'inputs.json').read_text())
 assert old_inputs['canonical_sdc_sha256']==sha(CANONICAL/'ti60f225_oob.sdc')
 original=bound
 assert len(re.findall(r'^set_bus_skew ',original,re.M))==10
 assert 'create_clock -period 6.666 -name process_clk_150' in original
 assert 'create_clock -period 199.980 -name CLK_5M' in original
 assert not re.search(r'^set_clock_uncertainty',original,re.M)
 # SW0 is an asynchronous human button with two actual retained sync FFs.
 # Only the external-pin path is excluded; sync-stage and all normal paths
 # remain timed. Existing PLL/Gray/IO constraints are unchanged.
 extra='\n# Asynchronous SW0, actual two-FF debounce synchronizer audited separately.\nset_false_path -from [get_ports {service_key_n}]\n'
 oldsha=sha(before);before.write_text(original+extra,encoding='utf8')
 result=dict(profile='released_original_clock_model_plus_async_SW0',before_sha256=oldsha,after_sha256=sha(before),
  original_sdc=str(PLACED/'ti60f225_oob.sdc'),original_sdc_sha256=sha(PLACED/'ti60f225_oob.sdc'),
  original_physical_inputs_sha256=sha(PLACED/'physical_inputs.json'),added_false_path_ports=['service_key_n'],
  bound_sdc=str(CANONICAL/'ti60f225_oob.sdc'),bound_sdc_sha256=sha(CANONICAL/'ti60f225_oob.sdc'),native_ip_input_byte_identical=True,current_nonempty_endpoints_pending=True,
  unchanged_original_constraints=True,gray_bus_constraints=10,reset_periphery_audit=delta,
  cdc_release=False,new_pixel_snapshot_settling_requires_routed_check=True,hardware_access=False)
 (root/'service_clock_profile.json').write_text(json.dumps(result,indent=2));return result
