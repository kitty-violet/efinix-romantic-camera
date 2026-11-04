"""Measure reported routed data paths; geometry comparisons are not route proof."""
import argparse,json,re
from pathlib import Path
from run_units import ROOT,sha

def locations(folder):
    result={}
    for line in (folder/'outflow/ti60f225_oob.out.placeloc').read_text().splitlines():
        w=line.split()
        if len(w)==4 and not line.startswith('#'):result[w[0]]=list(map(int,w[1:]))
    return result

def read_paths(folder):
    report=folder/'outflow/ti60f225_oob.timing.rpt';text=report.read_text();result=[]
    for block in re.split(r'\+\+\+\+ Path \d+ \+{5,}',text)[1:]:
        slack=re.search(r'^Slack\s*:\s*(-[\d.]+)',block,re.M)
        if not slack:continue
        def field(k):return re.search(r'^'+k+r'\s*:\s*(.*)',block,re.M)[1].strip()
        data=block.split('\nData Path\n')[1].split('\nCapture Clock Path\n')[0]
        rows=[]
        for line in data.splitlines():
            m=re.match(r'^\s*(\S+)\s+(EFX_\w+|net)\s+([-+\d.]+)\s+([-+\d.]+)\s+(\d+)\s+\((\d+),(\d+)\)',line)
            if m:rows.append(dict(name=m[1],kind=m[2],delay_ns=float(m[3]),arrival_ns=float(m[4]),fanout=int(m[5]),xy=[int(m[6]),int(m[7])]))
        assert rows
        total=sum(r['delay_ns'] for r in rows)
        target=float(re.search(r'^\+ Clock To Q \+ Data Path Delay\s*:\s*([\d.]+)',block,re.M)[1])
        assert abs(total-target)<=.002,(total,target)
        segments=[]
        for i,r in enumerate(rows):
            if r['kind']=='net':
                assert i>0 and i+1<len(rows) and rows[i-1]['kind']!='net' and rows[i+1]['kind']!='net'
                segments.append(dict(**r,start=rows[i-1]['name'].split('|')[0],end=rows[i+1]['name'].split('|')[0],
                    end_xy=rows[i+1]['xy'],manhattan=sum(abs(a-b) for a,b in zip(r['xy'],rows[i+1]['xy']))))
        net_delay=sum(r['delay_ns'] for r in segments)
        result.append(dict(slack_ns=float(slack[1]),start=field('Path Begin'),end=field('Path End'),
            logic_levels=int(field('Logic Level')),data_delay_ns=target,net_delay_ns=round(net_delay,3),
            net_fraction=net_delay/target,segments=segments))
    assert len(result)==10 and min(r['slack_ns'] for r in result)==-.660
    return dict(source=str(report),source_sha256=sha(report),scope='Ten reported failing paths, not exhaustive failing endpoints; actual data net delays exclude clocks.',paths=result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--failed',type=Path,required=True)
    p.add_argument('--placements',type=Path,nargs='*',default=[]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    failed=a.failed.resolve();out=a.output.resolve();assert out.is_relative_to(ROOT) and not out.exists()
    result=read_paths(failed);comparisons=[]
    for folder in a.placements:
        folder=folder.resolve();loc=locations(folder);nets=[]
        assert sha(folder/'outflow/ti60f225_oob.map.v')==sha(failed/'outflow/ti60f225_oob.map.v')
        for path in result['paths']:
            segments=[]
            for r in path['segments']:
                assert r['start'] in loc and r['end'] in loc,(r['start'],r['end'])
                src,dst=loc[r['start']],loc[r['end']]
                segments.append(dict(name=r['name'],old_delay_ns=r['delay_ns'],old_distance=r['manhattan'],
                    new_distance=sum(abs(x-y) for x,y in zip(src[:2],dst[:2])),start_xyz=src,end_xyz=dst))
            nets.append(dict(start=path['start'],end=path['end'],segments=segments))
        comparisons.append(dict(placement=str(folder),placement_sha256=sha(folder/'outflow/ti60f225_oob.out.placeloc'),
            metrics_sha256=sha(folder/'placement_review_inputs.json'),paths=nets))
    result.update(comparisons=comparisons,geometry_only_does_not_predict_routing_delay=True,hardware_access=False,board_release=False)
    out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps([dict(slack_ns=r['slack_ns'],net_delay_ns=r['net_delay_ns'],data_delay_ns=r['data_delay_ns'],net_percent=round(100*r['net_fraction'],1)) for r in result['paths']]))

if __name__=='__main__':main()
