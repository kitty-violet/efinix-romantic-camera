"""Check the actual mapped six-bank response predicates, without STA exceptions."""
import argparse, json, re
from pathlib import Path
from run_units import sha

p=argparse.ArgumentParser()
p.add_argument('--netlist',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert not a.output.exists()
s=a.netlist.read_text()
cells=[]
for m in re.finditer(r'EFX_(LUT4|COMB4)\s+(\S+)\s*\((.*?)\)\s*(?:/\*.*?\*/)?\s*;',s,re.S):
    kind,name,body=m.groups()
    match=re.search(r'/router/BANK\[(\d)\]\.CLIENT\[(\d)\]\.response_decode$',name)
    if not match:continue
    bank,client=map(int,match.groups())
    pins={k:v.strip() for k,v in re.findall(r'\.(I[0-3]|O|P)\(([^)]*)\)',body)}
    mask=re.search(r'defparam '+re.escape(name)+r"\s*\.LUTMASK = 16'h([0-9a-fA-F]+);",s)
    assert mask is not None,name
    value=int(mask[1],16)
    assert value==(0x1000<<client),(name,value)
    assert pins['I0'].endswith(f'/owner[{bank}] [0]'),(name,pins)
    assert pins['I1'].endswith(f'/owner[{bank}] [1]'),(name,pins)
    # The connected cnn_frame_datapath binds both arena/dut aux_response to bs.
    # row_pair_arena6 gates aux_full with its active/reset qualification.
    assert (pins['I2']==f'aux_response[{bank}]' or
            pins['I2'].endswith(f'/arena/aux_full[{bank}]') or
            pins['I2'].endswith(f'/arena/aux_full [{bank}]') or
            pins['I2'].endswith(f'/core/bs [{bank}]')),(name,pins)
    assert pins['I3'].endswith(f'/router/BANK[{bank}].response_enabled'),(name,pins)
    sensitivities=[]
    for bit in range(4):
        sensitivities.append(any(((value>>i)&1)!=((value>>(i^(1<<bit)))&1)
                                 for i in range(16)))
    assert all(sensitivities),(name,sensitivities)
    if kind=='COMB4':
        mode=re.search(r'defparam '+re.escape(name)+r'\s*\.MODE = "([^"]+)";',s)
        assert mode and mode[1]=='LOGIC'
    cells.append(dict(bank=bank,client=client,cell=name,primitive=kind,pins=pins,
                      mask=f'{value:04x}',all_four_inputs_functionally_used=True))
assert sorted((c['bank'],c['client']) for c in cells)==[(b,m) for b in range(6) for m in range(3)]
result=dict(status='MAPPED_18_RESPONSE_PREDICATES_PASS_NOT_STA',netlist_sha256=sha(a.netlist),
            cells=cells,timing_exceptions_added=False,physical_timing_pass=False)
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],cells=len(cells),netlist_sha256=result['netlist_sha256'])))
