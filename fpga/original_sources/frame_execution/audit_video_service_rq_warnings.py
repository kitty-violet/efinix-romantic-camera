"""RQ variant's one additional official opaque-IP optimization notice.

The existing checker still requires identical encrypted input, all 1201/4128
boundary ports/bits and the current actual-map boot proof. This classifies a
tool notice; it does not claim to inspect the encrypted internal circuitry.
"""
from pathlib import Path
source = Path(__file__).with_name('audit_video_service_narrow_warnings.py')
code = source.read_text()
needle = "prefix='u_ddr3_top/"
assert code.count(needle) == 1
prefix = 'u_ddr3_top/controller/u_efx_ddr3_soft_controller/genblk1.inst_ddr3_axi/'
names = [prefix + n for n in (
 'n1899_c[8:0]', 'u_wr_addr_fifo/Data_Num_c[4:0]',
 'inst_arburst_fifo/Data_Num[5]', 'inst_efx_ddr3/cmp_sub_cnt[0]',
 'inst_efx_ddr3/top_mc/fifo_ar/popCC_popPtr[5:3]',
 'inst_efx_ddr3/top_mc/fifo_ar/popCC_popPtrGray[6]',
 'inst_efx_ddr3/top_mc/fifo_ar/popCC_popPtr_c[6]',
 'inst_efx_ddr3/top_mc/fifo_ar/popCC_popPtr_c[2:0]',
 'inst_efx_ddr3/top_mc/fifo_aw/pushCC_pushPtrGray[6]')]
code = code.replace(needle, 'extra.extend('+repr(names)+')\n'+needle)
needle = "exec(compile(code,str(source)+':schema5-current-warning-list','exec'),globals())"
extra = '''  unused=re.search(r"Instantiated LUT '([^']+)'.*does not depend on connected input '(I[0-3])'",line)
  if line.startswith('[EFX-0300') and unused:
   n,pin=unused.groups();assert g[n][0]=='EFX_LUT4' and g[n][1].get(pin)
   mask=ps[n]['LUTMASK'][1];bit=int(pin[1])
   assert all((mask>>v&1)==(mask>>(v^(1<<bit))&1) for v in range(16)),(n,pin)
   counts['proved_cofactor_unused_input']+=1;continue
  if line.startswith('[EFX-0200') and any('Removing redundant signal : '+n+'. (' in line for n in ('independent_ddr_debug[35:14]','independent_ddr_debug[12:0]')):
   # These are declaration-only bits of an old alias. The protected DDR
   # output is still connected to its original Open_* wires; bit13 remains
   # live. Reject a new whole-bus or other-bit reference instead of assuming
   # that every signal carrying a debug name can be discarded.
   decl=r'wire\\s*\\[35:0\\]\\s*independent_ddr_debug\\s*;'
   assert len(re.findall(decl,text))==1
   without_decl=re.sub(decl,'',text)
   refs=re.findall(r'(?<![\\w.])independent_ddr_debug\\s*(\\[[^\\]]+\\])?',without_decl)
   assert refs and set(refs)=={'[13]'},refs
   assert proof['protected_input_unchanged'] and proof['protected_boundary']['other_connections_equal']
   counts['proved_declaration_only_debug_alias_bits']+=1;continue
'''
assert code.count(needle) == 1
anchor = '  if norm(line) in inherited:'
code = code.replace(needle, 'assert code.count('+repr(anchor)+')==1\ncode=code.replace('+repr(anchor)+','+repr(extra+anchor)+')\n'+needle)
exec(compile(code, str(source)+':rq-current-notice', 'exec'), globals())
