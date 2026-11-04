"""One larger bounded route after a decreasing prefix hit its time budget."""
import inspect,json,re
import review_native_split_route as old
from run_units import ROOT,HERE,sha
from control_command_repair import once

def main():
    prior=ROOT/'tmp/frame-execution/native-split-route1'
    guard=ROOT/'tmp/board-test/bounded-native-route1/result.json'
    failure=json.loads((prior/'failure.json').read_text())
    assert 'TimeoutExpired' in failure['error']
    body=(ROOT/'tmp/frame-execution/native-split-bound-place1/outflow/ti60f225_oob.route.out').read_text()
    rows=[(int(a),int(b)) for a,b in re.findall(r'Iter\s+(\d+):\s*overlap=(\d+)',body)]
    assert rows[-6:]==[(56,40),(57,25),(58,23),(59,19),(60,18),(61,13)]
    assert json.loads(guard.read_text())['status']=='COMMAND_FAILED'
    code=inspect.getsource(old.main)
    code=once(code,"'tmp/frame-execution/native-split-bound-place1'","'tmp/frame-execution/native-split-bound-place2'")
    code=once(code,'budget_seconds=900','budget_seconds=1350')
    last="    (physical/'route_review.json').write_text"
    insertion="""    record['rationale']+=' Prior900s tool/945s wrapper budget stopped at iter61/13conflicts with last six counts40,25,23,19,18,13. This is one unchanged-map/placement replay with1350s route budget to obtain final legal route and actual timing; not a seed sweep or an assumption of timing PASS. A prebound-placement comparison was stopped early after identical global placement and first five annealer rows; no demonstrated physical optimization was claimed.'
    record['budget_reason']='One1350s route plus45s process cleanup window;1800s whole-job limit. Same frozen layout; no repeated short simulations. Stop on actual tool failure or budget.'
    for path in (prior/'failure.json',guard,
        ROOT/'tmp/board-test/bounded-native-prebound-place1/early-stop-evidence.json',
        ROOT/'tmp/frame-execution/native-split-bound-place1/outflow/ti60f225_oob.route.out'):
        record['required_evidence'][str(path)]=sha(path)
"""
    code=once(code,last,insertion+last)
    scope=dict(old.__dict__,__file__=__file__,prior=prior,guard=guard)
    exec(compile(code,__file__+':actual-prefix-review','exec'),scope)
    scope['main']()

if __name__=='__main__':main()
