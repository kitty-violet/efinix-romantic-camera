"""Reconcile mapped area attribution; no full-system fit claim."""
import json,re
from pathlib import Path
from run_units import ROOT,sha
def parse(path):
 rows=[]
 for line in path.read_text(encoding='utf-8').splitlines():
  fields=line.split('\t')
  if len(fields)<8 or not re.fullmatch(r'\s*\d+\(\d+\)\s*',fields[1]):continue
  name=fields[0];values=[tuple(map(int,re.fullmatch(r'\s*(\d+)\((\d+)\)\s*',v).groups())) for v in fields[1:8]]
  rows.append(dict(name=name.strip(),depth=len(name)-len(name.lstrip()),total=dict(zip(('FF','SRL','ADD','LUT4','COMB4','RAM','DSP_PRIMITIVES'),[v[0] for v in values])),own=dict(zip(('FF','SRL','ADD','LUT4','COMB4','RAM','DSP_PRIMITIVES'),[v[1] for v in values]))))
 return rows
def comb(v):return sum(v[k] for k in ('LUT4','ADD','SRL','COMB4'))
def main():
 root=ROOT/'tmp/frame-execution/compact-quant-map1'
 report=root/'outflow/compute_column_cluster.res.csv';res=json.loads((root/'result.json').read_text())
 rows=parse(report);top=rows[0];children=[r for r in rows if r['depth']==1]
 assert comb(top['total'])==48996
 for key in top['total']:assert top['own'][key]+sum(r['total'][key] for r in children)==top['total'][key],key
 for k in ('LUT4','ADD','FF'):assert top['total'][k]==res['resources'][k]
 assert top['total']['SRL']==sum(v for k,v in res['resources'].items() if k.startswith('SRL'))
 buckets=[dict(name=r['name'],comb_estimate=comb(r['total']),registers=r['total']['FF']) for r in children]
 buckets.append(dict(name='top-owned routing/control (tool hierarchy attribution)',comb_estimate=comb(top['own']),registers=top['own']['FF']))
 scatter_root=ROOT/'tmp/frame-execution/stride-scatter-map1'
 scatter=json.loads((scatter_root/'result.json').read_text())
 assert scatter['resources'].get('DSP24',0)==scatter['resources'].get('DSP48',0)==0
 result=dict(status='MAPPED_AREA_ATTRIBUTION_RECONCILED_NOT_SYSTEM_FIT',report=str(report),report_sha256=sha(report),map_sha256=res['map_sha256'],
  comb_estimate=comb(top['total']),registers=top['total']['FF'],physical_dsp=160,local_ram_including_external_arena_and_resident=136,
  buckets=sorted(buckets,key=lambda r:-r['comb_estimate']),scatter=dict(resources=scatter['resources'],comb_estimate=comb(dict(LUT4=scatter['resources']['LUT4'],ADD=scatter['resources']['ADD'],SRL=0,COMB4=0)),map_sha256=scatter['map_sha256']),
  cautions=['Hierarchy attribution is the mapper estimate; optimization may move logic across module boundaries.',
   'Top routing/control bucket is not an exact source-line classification.',
   'LUT+ADD+SRL is not final placed XLR utilization; FF packing and routing still matter.',
   'Local136RAM excludes full DDR/video/CPU.',
   'Isolated scatter is not integrated; do not add or subtract module maps as if final placement were measured.',
   'Single-ingress scatter does not meet four-RQ peak ingress; duplicating four complete modules is not selected.'],
  final_fit=False,multi_seed_sta=False,continuous_frame_ms=None)
 out=ROOT/'output/toolchain/compute_resource_attribution_2026-09-30.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
