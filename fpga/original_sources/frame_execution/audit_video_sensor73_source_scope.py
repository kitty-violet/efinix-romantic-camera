"""Bind the exact source edit and validate the fresh mapped sensor interface.

This does not claim a whole-netlist formal equivalence proof. The mapper
re-factors unrelated inferred LUTs; those functions retain the exact released
source and tests. Physical timing and affected actual primitive tests are new.
"""
from pathlib import Path
wrapper=Path(__file__).with_name('audit_video_sensor_rom_map.py')
text=wrapper.read_text(encoding='utf8')
marker="exec(compile(code,str(source)+':rom-backed-eight-muxes','exec'),globals())"
assert text.count(marker)==1
exec(compile(text.replace(marker,''),str(wrapper)+':build-source-scope','exec'),globals())
begin=code.index(' assert expected.keys()==g.keys()')
end=code.index(' # Validate the existing qualifier',begin)
code=code[:begin]+''' from complete_native_logic_split import protected_blocks
 from probe_native_open_ports import protected_pins
 assert proof['all_unlisted_source_byte_equal']
 assert proof['script_sha256']==sha(HERE/'prepare_video_sensor73_original_input.py')
 assert proof['base_generator_sha256']==sha(HERE/'prepare_video_sensor_rom_verify.py')
 assert proof['original_synthesis_input_sha256']==sha(parent/'native_source.v')
 assert protected_blocks((board/'native_source.v').read_text())==protected_blocks((parent/'native_source.v').read_text())
 before_protected=protected_pins((parent/'native_source.v').read_text())
 after_protected=protected_pins((board/'native_source.v').read_text())
 assert before_protected==after_protected
 nonlut={n for n,(k,_) in expected.items() if k!='EFX_LUT4'}
 assert nonlut=={n for n,(k,_) in g.items() if k!='EFX_LUT4'}
 for n in nonlut:
  assert expected[n][0]==g[n][0] and ep[n]==ps[n],('non-LUT mapped parameters',n)
 boundary=dict(source_protected_ports=len(before_protected),source_protected_ports_and_blob_unchanged=True,
  mapped_nonlut_parameters_checked=len(nonlut),whole_mapped_netlist_equivalence_claimed=False,
  changed_scope='Only camera ROM contents/bank address and eight expected-byte muxes; generator reverses every edit and asserts released original input text equality.',
  limit='Whole-map BDD comparison hit its bounded node budget; no whole-map equivalence PASS is claimed. Actual mapped sensor pin tests and new implementation timing remain mandatory.')
''' +code[end:]
code=code.replace('SENSOR73_ROM_VERIFY_ALL_RETAINED_MAP_PASS','SENSOR73_EXACT_SOURCE_SCOPE_FRESH_MAP_PASS')
code=code.replace('retained=len(old),new_luts=8,expected_ff_d_changes=8,ram_address_pin_changes=1,',
 "source_added_luts=8,net_added_luts=len(g)-len(old),expected_ff_d_changes=8,ram_address_pin_changes=1,")
code=code.replace('other_parameters_connections_equivalent=True','all_other_source_text_unchanged=True')
code=code.replace("source,native/'inputs.json',board/'inputs.json'","source,wrapper,HERE/'prepare_video_sensor73_original_input.py',HERE/'prepare_video_sensor_rom_verify.py',parent/'native_source.v',native/'inputs.json',board/'inputs.json'")
exec(compile(code,str(wrapper)+':exact-source-scope','exec'),globals())
