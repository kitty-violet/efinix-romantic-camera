"""Audit reported LOGIC COMB4 arcs against LUTMASK and constant inputs.

Diagnostic only: never writes SDC, suppresses timing checks, or releases a build.
"""
import argparse, itertools, json, re
from pathlib import Path
from run_units import sha

p=argparse.ArgumentParser()
p.add_argument('--board', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--report', default='ti60f225_oob.place.timing.rpt')
a=p.parse_args()
net=a.board/'outflow/ti60f225_oob.map.v'
assert Path(a.report).name==a.report
report=a.board/'outflow'/a.report
text=net.read_text()
cells={}
masks=dict(re.findall(r"defparam (\S+)\.LUTMASK = 16'h([0-9a-fA-F]+);",text))
modes=dict(re.findall(r'defparam (\S+)\.MODE = "([^"]+)";',text))
for m in re.finditer(r'EFX_COMB4\s+(\S+)\s*\((.*?)\)\s*(?:/\*.*?\*/)?\s*;',text,re.S):
    name,body=m.groups()
    mask=masks.get(name);mode=modes.get(name)
    if not mask or mode!='LOGIC':continue
    pins=dict(re.findall(r'\.(I[0-3]|[OP])\(([^)]*)\)',body))
    cells[name]=(int(mask,16),pins)
arcs=[]
lines=report.read_text().splitlines()
for i,line in enumerate(lines[:-1]):
    m=re.match(r'\s*(\S+)\|I([0-3])\s+EFX_COMB4\b',line)
    if not m:continue
    name,bit=m[1],int(m[2])
    n=re.match(r'\s*'+re.escape(name)+r'\|([OP])\s+EFX_COMB4\b',lines[i+1])
    if not n or name not in cells:continue
    port=n[1];mask,pins=cells[name]
    constants={j:int(pins['I'+str(j)][-1]) for j in range(4) if pins.get('I'+str(j)) in ("1'b0","1'b1")}
    pairs=0;sensitive=False
    for values in itertools.product((0,1),repeat=4):
        if any(values[j]!=v for j,v in constants.items() if j!=bit):continue
        if values[bit]!=0:continue
        other=list(values);other[bit]=1
        def evaluate(v):
            index=sum(v[j]<<j for j in range(4)) if port=='O' else 8+2*v[1]+v[0]
            return (mask>>index)&1
        pairs+=1
        sensitive|=evaluate(values)!=evaluate(other)
    arcs.append(dict(cell=name,input='I'+str(bit),output=port,mask=f'{mask:04x}',pins=pins,
                     constant_inputs=constants,boolean_sensitive=sensitive,enumerated_pairs=pairs,
                     report_line=i+1))
paths=[]
for m in re.finditer(r'\+\+\+\+ Path \d+ \+[^\n]*\n(.*?)(?=\+\+\+\+ Path|\Z)',report.read_text(),re.S):
    body=m[1]
    slack=re.search(r'^Slack\s*:\s*(-?\d+\.\d+)',body,re.M)
    if not slack:continue
    start=report.read_text()[:m.start()].count('\n')+1
    end=start+m[0].count('\n')
    selected=[x for x in arcs if start<=x['report_line']<end]
    paths.append(dict(slack_ns=float(slack[1]),begin=re.search(r'^Path Begin\s*:\s*(.*)',body,re.M)[1].strip(),
                      end=re.search(r'^Path End\s*:\s*(.*)',body,re.M)[1].strip(),
                      start_line=start,known_insensitive_cells=sorted({x['cell'] for x in selected if not x['boolean_sensitive']})))
result=dict(status='DIAGNOSTIC_ONLY_NOT_STA_RELEASE',board=str(a.board),map_sha256=sha(net),paths=paths,
            report_sha256=sha(report),arcs=arcs,insensitive_arcs=[x for x in arcs if not x['boolean_sensitive']],
            timing_exceptions_added=False)
assert not a.output.exists()
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(output=str(a.output),arcs=len(arcs),
                      insensitive_cells=sorted({x['cell'] for x in result['insensitive_arcs']}),
                      negative_paths=sum(x['slack_ns']<0 for x in paths),
                      negative_without_known_insensitive_arc=[x for x in paths if x['slack_ns']<0 and not x['known_insensitive_cells']][:8]),indent=2))
