"""Bind packed RAM timing pins through the actual final driving primitive.

Efinity packing may permute physical RAM lanes. Logical WDATA indices must
not be equated with timing-report physical pin indices merely by their name.
Every active logical bit must still have exactly one measured physical pin.
"""
import re
from video_service_sample_ram import bit_tokens
from audit_video_service_narrow_jtag import parsed

def prepare(root,structure):
    text=(root/'outflow/ti60f225_oob.map.v').read_text();cells,_=parsed(text)
    endpoints={}
    for row in structure['dualclock_sample_ram']:
        if row['owner']!='service_pixel_snapshot':continue
        by_net={}
        for bit,t in enumerate(reversed(bit_tokens([tuple(x) for x in row['wdatas']]))):
            if t[0]=='literal':assert t[1]=="1'b0";continue
            label=t[1]+(f'[{t[2]}]' if t[2] is not None else '')
            assert label not in by_net
            by_net[label]=row['cell']+f'|WDATA[{bit}]'
        endpoints[row['cell']]=by_net
    assert sum(map(len,endpoints.values()))==32
    return cells,endpoints,{}

def logical_end(state,block,physical):
    cells,endpoints,seen=state
    if '|WDATA[' not in physical or physical.split('|')[0] not in endpoints:return physical
    cell=physical.split('|')[0]
    # The last named LUT output is connected directly to this RAM pin in
    # the measured route; match its actual mapped output, not a guessed net alias.
    data=block.split('\nData Path\n')[-1].split('Capture Clock Path')[0]
    outputs=re.findall(r'^\s*\([^\n]+\)\s+EFX_LUT4\s+[\d.]+\s+[\d.]+\s+(\S+)\|out\s*$',data,re.M)
    assert outputs,('missing RAM driving primitive',physical)
    name=outputs[-1];assert cells[name][0]=='EFX_LUT4'
    net=cells[name][1]['O'];assert net in endpoints[cell],(physical,name,net)
    logical=endpoints[cell][net]
    assert physical not in seen and logical not in [x['logical'] for x in seen.values()]
    seen[physical]=dict(logical=logical,driving_cell=name,mapped_net=net)
    return logical
