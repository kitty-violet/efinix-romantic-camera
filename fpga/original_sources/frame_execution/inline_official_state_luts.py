"""Textually inline 23 public official LUT models for simulation only.

The exact function, weak defaults, init value and sensitivity list remain from
the bound vendor source. No truth-table substitute, time scale change or RTL
edit is used. Starter's recommended-instance slowdown is not a license limit.
"""
import json,re
from run_units import sha

def inline(text,model,predicate):
    source=model.read_text()
    assert 'pragma protect' not in source
    header=source.index('output O');start=source.index(');',header)+2
    body=source[start:source.rindex('endmodule')]
    assert 'always @(I0_net or I1_net or I2_net or I3_net)' in body
    params={n:v.strip() for n,v in re.findall(r'defparam\s+(\S+)\s*\.LUTMASK\s*=\s*([^;]+);',text)}
    selected=[]
    def change(m):
        name,rawports=m.groups()
        if not predicate(name):return m[0]
        ports=dict(re.findall(r'\.(\w+)\(([^()]*)\)',rawports));assert set(ports)<=set(('I0','I1','I2','I3','O')) and 'O' in ports
        assert name in params
        selected.append(name)
        lines=['generate if(1) begin : '+name+' ',f' localparam [15:0] LUTMASK={params[name]};',
               ' wire I0,I1,I2,I3,O;']
        # A missing original input remains undriven and receives exactly the
        # official weak-zero default through the unchanged model body.
        lines.extend(f' assign {pin}={value};' for pin,value in ports.items() if pin!='O')
        lines.append(f' assign {ports["O"]}=O;')
        return '\n'.join(lines)+body+'\nend endgenerate\n'
    changed=re.sub(r'\bEFX_LUT4\s+(\S+)\s*\((.*?)\)\s*;',change,text,flags=re.S)
    for name in selected:
        changed,n=re.subn(r'defparam\s+'+re.escape(name)+r'\s*\.LUTMASK\s*=\s*[^;]+;','',changed);assert n==1
    assert len(selected)==23,len(selected)
    return changed,dict(status='OFFICIAL_MODEL_BODY_TEXTUALLY_INLINED_ONLY',instances=selected,
        official_model=str(model),official_model_sha256=sha(model),primitive_instances_removed=23,
        delays_clock_stimulus_assertions_unchanged=True,hardware_netlist_unchanged=True)
