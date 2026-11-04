"""Classify optimization notices located exactly at unchanged encrypted IP.

This only classifies a vendor message. Public connectivity and current-map
boot remain mandatory; it does not assert encrypted internal equivalence.
"""
from pathlib import Path
source=Path(__file__).with_name('audit_video_service_rq_warnings.py')
code=source.read_text()
anchor="anchor = '  if norm(line) in inherited:'"
assert code.count(anchor)==1
extra='''  if line.startswith('[EFX-0200 WARNING] Removing redundant signal : '):
   opaque=re.search(r'Removing redundant signal : (.*?)\\. \\(.*native_source\\.v:(\\d+)\\)',line)
   if opaque:
    opaque_name=opaque[1].replace('protected_inst/','')
    if opaque_name.startswith(('dvi_encoder_m0/','inst_efx_csi2_rx/','u_ddr3_top/')):
     origin=int(opaque[2])
     if 'protected_end_lines' not in locals():
      protected_end_lines=set();inside=False
      for number,source_line in enumerate(text.splitlines(),1):
       if source_line.strip()=='`pragma protect begin_protected':assert not inside;inside=True
       if source_line.strip()=='`pragma protect end_protected':assert inside;inside=False;protected_end_lines.add(number)
      assert not inside and protected_end_lines
     assert origin in protected_end_lines,('not at an exact encrypted block boundary',origin)
     assert proof['protected_input_unchanged'] and proof['protected_boundary']['other_connections_equal']
     assert proof['protected_boundary']['ports']==1201 and proof['protected_boundary']['bits']==4128
     reason='official_redundancy_notice_at_exact_unchanged_encrypted_block_end_not_internal_equivalence'
     counts[reason]+=1;new.append(dict(warning=line,reason=reason));continue
'''
code=code.replace(anchor,'extra+='+repr(extra)+'\n'+anchor)
exec(compile(code,str(source)+':exact-encrypted-origin','exec'),globals())
