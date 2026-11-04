"""Parse saved STA data arcs only; never rerun P&R or include clock-tree delay."""
import collections,json,re
from pathlib import Path
from run_units import ROOT,sha

def summarize(root):
    reports={};families=collections.defaultdict(list)
    for file in (root/'outflow').glob('survey_*.rpt'):
        rows=[]
        for block in re.split(r'(?=^Slack\s*:)',file.read_text(),flags=re.M)[1:]:
            data=re.split(r'^Data Path\s*$',block,flags=re.M)[1].split('Capture Clock Path',1)[0]
            nets=[(float(t),n.strip()) for t,n in re.findall(r'^\s*net \(fo=\d+\)\s+([\d.]+)\s+[\d.]+\s+(.*)',data,re.M)]
            row=dict(slack_ns=float(re.search(r'^Slack\s*:\s*([+\-\d.]+)',block)[1]),
                start=re.search(r'^Path Start\s*:\s*(.*)',block,re.M)[1].strip(),
                end=re.search(r'^Path End\s*:\s*(.*)',block,re.M)[1].strip(),
                data_net_delay_ns=sum(t for t,n in nets),longest_data_net=max(nets,default=None))
            rows.append(row)
            if file.name=='survey_setup.rpt':
                k=re.sub(r'\[\d+\]','[]',row['start'])+' -> '+re.sub(r'\[\d+\]','[]',row['end'])
                families[k].append(row)
        reports[file.name]=dict(sha256=sha(file),paths=rows)
    main=reports['survey_setup.rpt']['paths'];assert len(main)==256
    assert abs(main[0]['data_net_delay_ns']-7.304)<0.001
    return dict(status='ROUTED_SURVEY_DATA_ARCS_CORRECTED_NOT_RELEASE',sampled_endpoints=len(main),
      sampled_negative=sum(r['slack_ns']<0 for r in main),sum_launched=sum('/engine/sums[' in r['start'] for r in main),
      families=[dict(path=k,count=len(v),negative=sum(r['slack_ns']<0 for r in v),worst_ns=min(r['slack_ns'] for r in v)) for k,v in families.items()],
      reports=reports,correction='Separate data arcs from launch/capture clock paths. Original survey result remains historical; raw STA reports unchanged.',
      limitation='256 worst endpoints plus source filters are a bounded survey, not an exhaustive list of all paths.',runner_sha256=sha(__file__))

if __name__=='__main__':
    root=ROOT/'tmp/frame-execution/local-controls-path-survey2';dest=root/'data_arc_analysis.json';assert not dest.exists()
    result=summarize(root);dest.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('reports','families')}))
