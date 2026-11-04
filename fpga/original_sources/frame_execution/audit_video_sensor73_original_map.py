"""Original-input map audit, allowing proved synthesis LUT factoring only."""
from pathlib import Path
wrapper=Path(__file__).with_name('audit_video_sensor_rom_map.py')
text=wrapper.read_text(encoding='utf8')
marker="exec(compile(code,str(source)+':rom-backed-eight-muxes','exec'),globals())"
assert text.count(marker)==1
exec(compile(text.replace(marker,''),str(wrapper)+':build-audit','exec'),globals())
begin=code.index(' assert expected.keys()==g.keys()')
end=code.index(' renamed={tokens(',begin)
code=code[:begin]+''' from sensor73_comb_equivalence import compare
 forward,comb_proof=compare(expected,g,ep,ps)
 (board/'sensor73-comb-equivalence.json').write_text(json.dumps(comb_proof,indent=2))
''' +code[end:]
code=code.replace("retained=len(old),new_luts=8,expected_ff_d_changes=8,ram_address_pin_changes=1,",
 "retained_nonlut=comb_proof['retained_nonlut_cells'],source_added_luts=8,net_added_luts=len(g)-len(old),expected_ff_d_changes=8,ram_address_pin_changes=1,comb_equivalence=comb_proof,")
code=code.replace("source,native/'inputs.json',board/'inputs.json'","source,wrapper,HERE/'sensor73_comb_equivalence.py',board/'sensor73-comb-equivalence.json',native/'inputs.json',board/'inputs.json'")
code=code.replace('SENSOR73_ROM_VERIFY_ALL_RETAINED_MAP_PASS','SENSOR73_ORIGINAL_INPUT_EQUIVALENT_MAP_PASS')
exec(compile(code,str(wrapper)+':original-input-equivalence','exec'),globals())
