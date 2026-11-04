"""Summarize every reported failing endpoint and its actual slow interconnect."""
import argparse,collections,json,re
from pathlib import Path
from run_units import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();report=root/'outflow/canonical_setup.rpt'
 rows=[];netrows=collections.defaultdict(list)
 blocks=re.split(r'(?=^Slack\s*:)',report.read_text(),flags=re.M)[1:];assert len(blocks)==2000
 for b in blocks:
  slack=float(re.search(r'^Slack\s*:\s*([\d.+-]+)',b)[1])
  if slack>=0:continue
  start=re.search(r'^Path Start\s*:\s*(.*)',b,re.M)[1].strip();end=re.search(r'^Path End\s*:\s*(.*)',b,re.M)[1].strip()
  data=re.split(r'^Data Path\s*$',b,flags=re.M)[1].split('Capture Clock Path',1)[0]
  nets=[dict(fanout=int(fo),delay_ns=float(ns),net=n.strip()) for fo,ns,n in re.findall(r'^\s*net \(fo=(\d+)\)\s+([\d.]+)\s+[\d.]+\s+(.*)',data,re.M)]
  delay=float(re.search(r'^Data Path Delay\s*:\s*([\d.]+)',b,re.M)[1]);nets.sort(key=lambda x:x['delay_ns'],reverse=True)
  row=dict(start=start,end=end,slack_ns=slack,data_ns=delay,wire_fraction=sum(x['delay_ns'] for x in nets)/delay,slow_nets=nets[:4]);rows.append(row)
  for n in nets:netrows[n['net']].append(dict(delay_ns=n['delay_ns'],fanout=n['fanout'],path_end=end,slack_ns=slack))
 groups=collections.defaultdict(list)
 for r in rows:groups[re.sub(r'\[\d+\]','[]',r['end'])].append(r)
 families=[dict(end=k,count=len(v),worst_ns=min(x['slack_ns'] for x in v),mean_wire_fraction=sum(x['wire_fraction'] for x in v)/len(v)) for k,v in groups.items()]
 families.sort(key=lambda x:x['worst_ns'])
 slow=[dict(net=k,occurrences=len(v),worst_delay_ns=max(x['delay_ns'] for x in v),fanout=max(x['fanout'] for x in v)) for k,v in netrows.items()]
 slow.sort(key=lambda x:x['worst_delay_ns'],reverse=True)
 result=dict(status='ACTUAL_NEGATIVE_SETUP_PATHS_ANALYZED_NOT_REPAIRED',reported_endpoints=2000,negative_endpoints=len(rows),worst_ns=min(x['slack_ns'] for x in rows),families=families,slow_nets=slow,rows=rows,
  evidence={str(f):sha(f) for f in (report,Path(__file__))},hardware_access=False,board_release=False)
 (root/'failure-families.json').write_text(json.dumps(result,indent=2),encoding='utf8')
 print(json.dumps({k:result[k] for k in ('negative_endpoints','worst_ns','families')}));print(json.dumps(slow[:12]))

if __name__=='__main__':main()
