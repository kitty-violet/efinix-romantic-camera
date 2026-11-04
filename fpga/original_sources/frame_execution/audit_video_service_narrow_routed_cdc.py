"""Schema5 exact 32-bit mailbox endpoints, with unchanged finite timing bounds."""
from pathlib import Path
source=Path(__file__).with_name('audit_video_service_routed_cdc.py')
code=source.read_text()
code=code.replace('import collections, json, re','import collections, json, re\nfrom video_service_ram_timing_pinmap import prepare as prepare_pinmap, logical_end')
code=code.replace('    measured={};reports={}','    measured={};reports={};pinmap=prepare_pinmap(root,structure)')
code=code.replace("            end=field('Path End')","            physical_end=field('Path End');end=logical_end(pinmap,block,physical_end)")
code=code.replace("measured[end]=dict(start=start,end=end,kind=entry['kind']","measured[end]=dict(start=start,end=end,physical_end=physical_end,kind=entry['kind']")
code=code.replace("endpoint_counts=dict(counts),mailbox_bits=width","physical_ram_pin_bindings=pinmap[2],endpoint_counts=dict(counts),mailbox_bits=width")
for old,new in [
 ('assert len(ram)==17 and not held_ends\n        width=128',
  'assert len(ram)==7 and not held_ends\n        width=32'),
 (r'service_pixel_snapshot/storage/banks\[[0-6]\]\.sample\|WDATA\[\d+\]',
  r'service_pixel_snapshot/storage/banks\[[0-1]\]\.sample\|WDATA\[\d+\]'),
 ('bank*20+bit<128','bank*20+bit<32'),
 ('dict(held_part_selector_ram=115,two_ff_first_stage=5)',
  'dict(held_part_selector_ram=32,two_ff_first_stage=5)'),
]:
 assert code.count(old)==1,old
 code=code.replace(old,new)
exec(compile(code,str(source)+':schema5','exec'),globals())
