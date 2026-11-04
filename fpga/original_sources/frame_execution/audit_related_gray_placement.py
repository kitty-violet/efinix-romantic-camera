"""Verify actual source placements and measure, rather than assume, sync gaps."""
import argparse,json,re
from pathlib import Path
from run_units import ROOT,sha
p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);a=p.parse_args()
b=a.board.resolve();assert b.is_relative_to(ROOT/'tmp')
proof=b/'related_gray_placement.json';d=json.loads(proof.read_text())
assert d['sources_only']
net=b/'outflow/ti60f225_oob.map.v';assert sha(net)==d['reference_map_sha256']
public_ff=set(re.findall(r'^\s*EFX_FF\s+\\(\S+)\s',net.read_text(),re.M))
f=b/'outflow/ti60f225_oob.out.placeloc';cells={}
for line in f.read_text().splitlines():
    w=line.split()
    if len(w)==4 and all(v.isdigit() for v in w[1:]):cells[w[0]]=tuple(map(int,w[1:]))
rows=[]
for r in d['pairs']:
    assert r['source'] in public_ff and r['head'] not in public_ff
    src=cells[r['source']];head=cells[r['head']]
    assert list(src)==r['source_location'],('Ignored assignment',r['source'],src)
    rows.append(dict(bit=r['bit'],source=src,head=head,manhattan_gap=abs(src[0]-head[0])+abs(src[1]-head[1])))
out=b/'related_gray_placement_audit.json';assert not out.exists()
result=dict(status='PUBLIC_SOURCE_ASSIGNMENTS_VERIFIED_NOT_TIMING_PASS',sources=7,
    protected_heads_are_automatic=True,pairs=rows,placement_sha256=sha(f),
    proof_sha256=sha(proof),board_release=False)
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
