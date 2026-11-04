"""Block a throughput regression hidden by a passing numeric cluster test."""
import argparse,json,re
from pathlib import Path
from run_units import ROOT,sha

p=argparse.ArgumentParser()
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
out=a.output.resolve()
assert out.is_relative_to(ROOT) and not out.exists()
def load(root):
    root=root.resolve();assert root.is_relative_to(ROOT/'tmp')
    r=json.loads((root/'result.json').read_text())
    m=json.loads((root/'inputs.json').read_text())
    assert r['status']=='FIXED_CLUSTER_REAL_RESIDENT_PASS'
    assert sha(root/'inputs.json')==r['inputs_sha256']
    assert sha(root/'simulation.log')==r['log_sha256']
    for rec in m['inputs']:assert sha(rec['snapshot'])==rec['sha256']
    for name,h in m['vectors'].items():assert sha(root/'vectors'/name)==h
    pattern=r'^BLOCK case=(\d+) kind=(\d+) width=(\d+) rows=(\d+) cin=(\d+) cout=(\d+) random=0 cycles=(\d+) last_command=(\d+)$'
    rows={}
    for values in re.findall(pattern,(root/'simulation.log').read_text(),re.M):
        v=list(map(int,values));key=tuple(v[:6]);assert key not in rows
        rows[key]=v[6:]
    assert len(rows)>=10
    return root,r,m,rows
ref,rr,rm,rb=load(a.reference)
can,cr,cm,cb=load(a.candidate)
assert rm['vectors']==cm['vectors'] and rb.keys()==cb.keys()
rows=[]
for key in rb:
    before,after=rb[key][0],cb[key][0]
    rows.append(dict(case=key[0],kind=key[1],width=key[2],rows=key[3],cin=key[4],cout=key[5],
        before=before,after=after,delta=after-before,regression_pct=100*(after-before)/before))
regressions=[r for r in rows if r['delta']>0]
data=dict(status='PERFORMANCE_REVIEW_REQUIRED' if regressions else 'DETERMINISTIC_CLUSTER_PERFORMANCE_PASS',
    reference=str(ref),candidate=str(can),reference_result_sha256=sha(ref/'result.json'),
    candidate_result_sha256=sha(can/'result.json'),rows=rows,regressions=regressions,
    long_replay_auto_release=not regressions,
    scope='Same vectors and fixed geometry, unstalled cases only. Random cases have different RNG schedules after latency changes.',
    policy='Any deterministic slowdown blocks automatic full replay; justify an explicit whole-frame budget or fix the short case first. Numeric PASS alone is insufficient.')
out.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=data['status'],regressions=regressions),ensure_ascii=False))
raise SystemExit(2 if regressions else 0)
