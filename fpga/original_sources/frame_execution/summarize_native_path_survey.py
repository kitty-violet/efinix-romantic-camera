"""Group actual routed setup paths; exclude launch/capture clock-tree arcs."""
import argparse,collections,json,re
from pathlib import Path
from run_units import sha

def analyze(root):
    result=json.loads((root/'result.json').read_text())
    assert result['original_and_route_unchanged']
    path=root/'outflow/canonical_setup.rpt'
    assert sha(path)==result['reports']['canonical_setup.rpt']
    rows=[]
    for block in re.split(r'(?=^Slack\s*:)',path.read_text(),flags=re.M)[1:]:
        data=re.split(r'^Data Path\s*$',block,flags=re.M)[1].split('Capture Clock Path',1)[0]
        nets=[dict(delay_ns=float(t),fanout=int(f),net=n.strip()) for f,t,n in re.findall(
            r'^\s*net \(fo=(\d+)\)\s+([\d.]+)\s+[\d.]+\s+(.*)',data,re.M)]
        assert nets
        total=float(re.search(r'^Data Path Delay\s*:\s*([\d.]+)',block,re.M)[1])
        net_total=sum(x['delay_ns'] for x in nets)
        assert 0<net_total<=total+.005
        rows.append(dict(slack_ns=float(re.search(r'^Slack\s*:\s*([+\-\d.]+)',block)[1]),
            start=re.search(r'^Path Start\s*:\s*(.*)',block,re.M)[1].strip(),
            end=re.search(r'^Path End\s*:\s*(.*)',block,re.M)[1].strip(),
            clock=re.search(r'^Capture Clock\s*:\s*(\S+)',block,re.M)[1],
            levels=int(re.search(r'^Logic Levels\s*:\s*(\d+)',block,re.M)[1]),
            data_delay_ns=total,net_delay_ns=round(net_total,6),net_fraction=net_total/total,
            longest=max(nets,key=lambda x:x['delay_ns']),nets=nets))
    assert rows and min(r['slack_ns'] for r in rows)==result['rows']['setup']['worst_ns']
    families=collections.defaultdict(list);hot=collections.defaultdict(list)
    bad=[r for r in rows if r['slack_ns']<0]
    for row in bad:
        key=re.sub(r'\[\d+\]','[]',row['start'])+' -> '+re.sub(r'\[\d+\]','[]',row['end'])
        families[key].append(row)
        for n in row['nets']:
            if n['delay_ns']>=1:
                hot[n['net']].append((n['delay_ns'],row['slack_ns'],row['end']))
    return dict(status='READ_ONLY_ROUTED_DATA_PATH_SURVEY_NOT_RELEASE',
        report=str(path),report_sha256=sha(path),source_result_sha256=sha(root/'result.json'),
        sampled_endpoints=len(rows),sampled_negative=len(bad),
        negative_clocks=dict(collections.Counter(r['clock'] for r in bad)),
        families=sorted([dict(path=k,count=len(v),worst_ns=min(r['slack_ns'] for r in v),
            min_net_fraction=min(r['net_fraction'] for r in v),
            mean_net_fraction=sum(r['net_fraction'] for r in v)/len(v),
            max_levels=max(r['levels'] for r in v)) for k,v in families.items()],key=lambda x:x['worst_ns']),
        hot_nets=sorted([dict(net=k,count=len(v),max_delay_ns=max(x[0] for x in v),
            worst_slack_ns=min(x[1] for x in v)) for k,v in hot.items()],key=lambda x:(-x['count'],-x['max_delay_ns'])),
        paths=rows,limitation='Worst 1000 requested endpoints, not an exhaustive all-path proof; original timing failure retained.',
        analyzer_sha256=sha(__file__),hardware_access=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    dest=a.root/'data_arc_analysis.json';assert not dest.exists()
    result=analyze(a.root);dest.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('paths','families','hot_nets')}))
    print(json.dumps(dict(families=result['families'][:16],hot_nets=result['hot_nets'][:15]),indent=2))
