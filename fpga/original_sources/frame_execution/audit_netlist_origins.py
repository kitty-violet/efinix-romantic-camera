"""Inspect primitive origins in a frozen map; source-line attribution is diagnostic."""
import argparse, collections, csv, json, re
from pathlib import Path
from run_units import ROOT, sha

p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
board=a.board.resolve();out=a.output.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists()
result=json.loads((board/'result.json').read_text());net=board/'outflow/ti60f225_oob.map.v'
assert sha(net)==result['map_sha256'];out.mkdir()
origins=collections.defaultdict(collections.Counter);active=None;body=[]
with net.open() as f:
 for line in f:
  m=re.match(r'\s+EFX_(LUT4|ADD|FF|COMB4|SRL8)\s+([^\s]+)',line)
  if m:active=m.groups();body=[line]
  elif active:body.append(line)
  if active and ';' in line:
   text=''.join(body);src=re.search(r'//\s+(.+\.(?:sv|v))\((\d+)\)',text)
   key=(Path(src[1]).name,int(src[2])) if src else ('unattributed',0)
   origins[key][active[0]]+=1;active=None
rows=[dict(source=k[0],line=k[1],**{kind:v[kind] for kind in ('LUT4','ADD','FF','COMB4','SRL8')}) for k,v in origins.items()]
rows.sort(key=lambda x:-(x['LUT4']+x['ADD']+x['FF']+x['SRL8']))
with (out/'origins.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['source','line','LUT4','ADD','FF','COMB4','SRL8']);w.writeheader();w.writerows(rows)
(out/'result.json').write_text(json.dumps(dict(status='DIAGNOSTIC_SOURCE_ATTRIBUTION',map_sha256=sha(net),totals={k:sum(r[k] for r in rows) for k in ('LUT4','ADD','FF','COMB4','SRL8')},caveat='Mapper may move logic across source/module boundaries. Not physical XLR attribution.'),indent=2))
print(json.dumps(rows[:35],indent=2))
