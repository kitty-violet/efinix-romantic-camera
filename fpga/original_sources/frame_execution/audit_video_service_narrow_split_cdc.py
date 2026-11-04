"""Keep schema5 JTAG-synchronous inputs separate from asynchronous crossings."""
from pathlib import Path
source=Path(__file__).with_name('audit_video_service_compact_routed_cdc.py')
code=source.read_text()
code=code.replace("c,_=parsed(text);compact=structure(text)","c,ps=parsed(text);compact=structure(text)\n "
    "rst=[f'service_serializer/dff_6/i8_rst_{i}~FF' for i in range(5)]\n "
    "assert {n for n in c if n.startswith('service_serializer/dff_') and '/i8_rst_' in n}==set(rst)\n "
    "for i,n in enumerate(rst):\n  "
    "assert c[n][0]=='EFX_FF'\n  "
    "pins=c[n][1]\n  "
    "assert pins['D']==(\"1'b0\" if i==0 else c[rst[i-1]][1]['Q'])\n  "
    "assert pins['CLK']=='jtag_inst1_TCK' and pins['CE']=='cpu/system_hardJtag_debug_logic_jtagBridge/n8'\n  "
    "assert pins['SR']=='cpu/system_hardJtag_debug_logic_jtagBridge/observation/n2987'\n  "
    "assert all(ps[n][k]==(1,1) for k in ('CLK_POLARITY','CE_POLARITY','SR_POLARITY','D_POLARITY','SR_SYNC','SR_SYNC_PRIORITY','SR_VALUE'))")
for old,new in [
 ('from audit_video_service_compact_jtag import structure,parsed',
  'from audit_video_service_narrow_jtag import structure,parsed'),
 ('from audit_video_service_routed_cdc import inventory as service_inventory',
  'from audit_video_service_narrow_routed_cdc import inventory as service_inventory'),
 ('(1352 if compact.get(\'dualclock_sample_ram\') else 1274)',
  '(776 if compact.get(\'dualclock_sample_ram\') else 1274)'),
 ('service_serializer/dff_11/i8_rst_','service_serializer/dff_6/i8_rst_'),
]:
 assert code.count(old)==1,old
 code=code.replace(old,new)
exec(compile(code,str(source)+':schema5','exec'),globals())
