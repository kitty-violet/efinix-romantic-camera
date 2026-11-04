"""Exact source-scope audit and actual mapped camera extraction for sensor75.

The full design is not claimed formally equivalent. Unchanged source and
protected ports are bound; all mapped sensor primitives are extracted for
fresh tests/equivalence. New physical timing is mandatory.
"""
from pathlib import Path
source=Path(__file__).with_name('audit_video_sensor72.py')
code=source.read_text(encoding='utf8')
code=code.replace("proof=json.loads((native/'result.json').read_text())", "proof=json.loads((native/'inputs.json').read_text())")
code=code.replace("proof['candidate_native_sha256']", "proof['source_sha256']")
begin=code.index(" for patch in proof['native_init_patches']:")
end=code.index(' # Validate the existing qualifier',begin)
code=code[:begin]+''' from prepare_video_service_color_exposure import read_rom
 from complete_native_logic_split import protected_blocks
 from probe_native_open_ports import protected_pins
 assert proof['status']=='SENSOR75_DANGLING_CELLS_SOURCE_PROVED_NOT_SYNTHESIZED'
 assert proof['all_unlisted_source_byte_equal'] and proof['new_luts']==proof['new_flip_flops']==proof['new_clocks']==0
 assert proof['script_sha256']==sha(HERE/'prepare_video_sensor75.py')
 assert proof['original_synthesis_input_sha256']==sha(parent/'native_source.v')
 for f,h in proof['files'].items():assert sha(f)==h
 edits=json.loads((native/'search.json').read_text())
 assert edits['combined_exhaustive_cases']==131072
 for n,r in edits['selected'].items():
  expected[n]=('EFX_LUT4',{p:[tuple(t) for t in ts] for p,ts in r['pins'].items()})
  ep[n]['LUTMASK']=(16,r['mask'])
 _,blocks=read_rom((native/'native_source.v').read_text(encoding='utf8'))
 blob=next(x[4] for x in blocks if x[3]==20)
 for i in range(40):ep['camera_checked/config_rom/ROM__D$2']['INIT_'+format(i,'X')]=(256,(blob>>(256*i))&((1<<256)-1))
 assert protected_blocks((board/'native_source.v').read_text())==protected_blocks((parent/'native_source.v').read_text())
 before_protected=protected_pins((parent/'native_source.v').read_text())
 assert before_protected==protected_pins((board/'native_source.v').read_text())
 removed=set(proof['removed_cells']);assert len(removed)==91 and proof['no_surviving_loads']
 for n in removed:del expected[n];del ep[n]
 nonlut={n for n,(k,_) in expected.items() if k!='EFX_LUT4'}
 assert nonlut=={n for n,(k,_) in g.items() if k!='EFX_LUT4'}
 for n in nonlut:assert expected[n][0]==g[n][0] and ep[n]==ps[n],('nonlut_params',n)
 common=set(expected)&set(g)
 literal_differences=[]
 for n in common:
  if expected[n]!=g[n] or ep[n]!=ps[n]:literal_differences.append(n)
 assert all(expected[n][0]=='EFX_LUT4' for n in set(expected)-set(g))
 assert all(g[n][0]=='EFX_LUT4' for n in set(g)-set(expected))
 assert proof['unchanged_remaining_source_graph'] and proof['primary_ports_assignments_unchanged']
 for f,h in proof['evidence'].items():assert sha(f)==h
 # Dead source cells can change synthesis factoring of unchanged RTL. Bind
 # the precise source edit and every remaining state element's parameters;
 # do not describe renamed/refactored LUTs as an exact mapped graph match.
 boundary=dict(declared_dangling_removed=sorted(removed),source_protected_ports=len(before_protected),source_protected_ports_and_blob_unchanged=True,
  mapped_nonlut_parameters_checked=len(nonlut),whole_mapped_netlist_equivalence_claimed=False,
  changed_scope='Five existing camera expectation LUTs plus three configuration ROM bytes. 91 proved dangling source cells removed; all surviving source connectivity unchanged. Fresh synthesis refactored inferred LUTs; all retained non-LUT parameters match the expected source change.',
  mapped_removed=sorted(set(expected)-set(g)),mapped_added=sorted(set(g)-set(expected)),mapped_literal_differences=sorted(literal_differences),
  limit='No whole mapped-netlist equivalence claim. Exact source scope, retained state parameters and fresh actual camera primitive equivalence are separate gates; new route/STA and supervised hardware tests remain required.')
''' +code[end:]
code=code.replace('sensor72-map-parse.pkl','sensor75-map-parse.pkl').replace('sensor72-map-audit.json','sensor75-map-audit.json')
code=code.replace('SENSOR72_ALL_RETAINED_MAP_PASS','SENSOR75_EXACT_SOURCE_SCOPE_FRESH_MAP_PASS')
code=code.replace('retained=len(old),new_luts=5,expected_ff_d_changes=4,','retained_nonlut=len(nonlut),source_added_luts=0,net_cell_delta=len(g)-len(old),expected_ff_d_changes=0,')
code=code.replace('other_parameters_connections_equivalent=True','all_other_original_source_byte_equal=True')
code=code.replace("Path(__file__),native/'result.json',board/'inputs.json'", "Path(__file__),source,HERE/'prepare_video_sensor75.py',parent/'native_source.v',native/'inputs.json',board/'inputs.json'")
begin=code.index(" rom=(native/'candidate_i2c_master_reg_rom.v').read_text()")
end=code.index(" tb=(HERE/'tb_camera_checked_config.sv').read_text()",begin)
code=code[:begin]+" shutil.copy2(native/'expected.hex',prepared/'expected.hex')\n"+code[end:]
code=code.replace('ACTUAL_SENSOR72_CONE_PREPARED','ACTUAL_SENSOR75_CONE_PREPARED')
exec(compile(code,str(source)+':minimal-sensor75','exec'),globals())
