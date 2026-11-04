"""Read frozen STA reports; separate data arcs from clock trees explicitly."""
import collections,json,re
from run_units import ROOT,sha

def main():
    root=ROOT/'tmp/frame-execution/minimal-feedback-path-survey1'
    target=root/'data_arc_analysis.json';assert not target.exists()
    reports={};groups=collections.defaultdict(list)
    for file in (root/'outflow').glob('survey_*.rpt'):
        rows=[]
        for block in re.split(r'(?=^Slack\s*:)',file.read_text(),flags=re.M)[1:]:
            data=re.split(r'^Data Path\s*$',block,flags=re.M)[1].split('Capture Clock Path',1)[0]
            nets=[dict(fanout=int(f),delay_ns=float(t),name=n.strip()) for f,t,n in
                re.findall(r'^\s*net \(fo=(\d+)\)\s+([\d.]+)\s+[\d.]+\s+(.*)',data,re.M)]
            assert nets,'No data nets parsed'
            arcs=[float(t) for t in re.findall(r'^\s*(?:\([\d,\s]+\)\s+\S+(?:\s+\([^)]*\))?|net \(fo=\d+\))\s+([\d.]+)\s+[\d.]+\s+',data,re.M)]
            row=dict(slack_ns=float(re.search(r'^Slack\s*:\s*([+\-\d.]+)',block)[1]),
                start=re.search(r'^Path Start\s*:\s*(.*)',block,re.M)[1].strip(),
                end=re.search(r'^Path End\s*:\s*(.*)',block,re.M)[1].strip(),
                data_delay_ns=round(sum(arcs),3),net_delay_ns=round(sum(n['delay_ns'] for n in nets),3),
                longest_net=max(nets,key=lambda n:n['delay_ns']),nets=nets)
            assert row['data_delay_ns']>=row['net_delay_ns']>0
            row['net_fraction']=round(row['net_delay_ns']/row['data_delay_ns'],4)
            rows.append(row)
            if file.name=='survey_setup.rpt':
                k=re.sub(r'\[\d+\]','[]',row['start'])+' -> '+re.sub(r'\[\d+\]','[]',row['end'])
                groups[k].append(row)
        reports[file.name]=dict(sha256=sha(file),paths=rows)
    rows=reports['survey_setup.rpt']['paths'];assert len(rows)==256
    assert abs(rows[0]['net_delay_ns']-6.890)<0.003,rows[0]
    families=[dict(path=k,count=len(v),worst_ns=min(x['slack_ns'] for x in v),
        mean_net_fraction=round(sum(x['net_fraction'] for x in v)/len(v),4),
        worst= min(v,key=lambda x:x['slack_ns'])) for k,v in groups.items()]
    result=dict(status='FROZEN_ROUTED_PATH_DATA_ARCS_CORRECTED_NOT_RELEASE',
        original_result_sha256=sha(root/'result.json'),runner_sha256=sha(__file__),reports=reports,families=families,
        correction='Original summary parsed an empty Data Path section; slack/endpoints were correct but net totals were zero. Raw reports and original summary remain unchanged.',
        limitation='256 distinct worst endpoints are a bounded sample, not the total chip violation count.',
        hardware_access=False)
    target.write_text(json.dumps(result,indent=2),encoding='utf8')
    for f in families:
        w=f['worst'];print(f"{f['count']:3} {f['worst_ns']:+.3f} net={f['mean_net_fraction']:.1%} worstnet={w['longest_net']['delay_ns']:.3f} fo={w['longest_net']['fanout']} {f['path']}")

if __name__=='__main__':main()
