"""Record terminal physical probes and their real limits, without declaring STA."""
import argparse, json, re
from pathlib import Path
from run_units import ROOT, sha

p=argparse.ArgumentParser()
p.add_argument('--roots',type=Path,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
out=a.output.resolve()
assert out.is_relative_to(ROOT/'output') and not out.exists()
rows=[]
for item in a.roots:
    root=item.resolve()
    assert root.is_relative_to(ROOT/'tmp')
    terminal=root/'result.json'
    failure=root/'failure.json'
    assert terminal.exists() or failure.exists(),('Still running',root)
    end=json.loads((terminal if terminal.exists() else failure).read_text())
    meta=json.loads((root/'physical_inputs.json').read_text())
    for rec in meta['inputs']:
        assert sha(rec['snapshot'])==rec['sha256'],rec['snapshot']
    net=root/'outflow/ti60f225_oob.map.v'
    assert sha(net)==meta['base_map_sha256']
    layout=meta.get('dsp_layout')
    if layout:
        assert sha(root/'fixed_dsp.placeloc')==layout['assignments_sha256']
        assert sha(root/'efx_pnr_settings.ini')==layout['settings_sha256']
    place=root/'outflow/ti60f225_oob.place.rpt'
    text=place.read_text()
    def pair(label):
        m=re.search(r'^'+re.escape(label)+r':\s*(\d+)\s*/\s*(\d+)',text,re.M)
        assert m,label
        return tuple(map(int,m.groups()))
    used,capacity=pair('XLRs')
    ram,ram_capacity=pair('Memory Blocks')
    dsp,dsp_capacity=pair('DSP Blocks')
    routing=int(re.search(r'XLRs needed for Routing:\s*(\d+)',text)[1])
    timing=root/'outflow/ti60f225_oob.place.timing.rpt'
    preliminary_path=None
    if timing.exists():
        tt=timing.read_text()
        first=tt.split('++++ Path 1 +',1)[1].split('++++ Path 2 +',1)[0]
        def value(label):
            return re.search(r'^'+re.escape(label)+r'\s*:\s*(.+)',first,re.M)[1].strip()
        preliminary_path=dict(begin=value('Path Begin'),end=value('Path End'),
            slack_ns=float(value('Slack').split()[0]),logic_levels=int(value('Logic Level')))
    capacity_file=root/'capacity_gate.json'
    capacity_gate=json.loads(capacity_file.read_text()) if capacity_file.exists() else None
    if capacity_gate:
        assert capacity_gate['used_xlr']==used and capacity_gate['map_sha256']==sha(net)
    capacity_stop=capacity_gate is not None and capacity_gate['status']=='PACKING_EXCEEDS_PROJECT_LIMIT'
    place_log=root/'outflow/ti60f225_oob.place.out'
    first_error=None
    if place_log.exists():
        errors=re.findall(r'^ERROR\s*:\s*(.+)',place_log.read_text(errors='replace'),re.M)
        if errors:first_error=errors[0]
    failed_before_timing=any(s['flow']=='pnr' and s['exit_code']!=0 for s in end.get('stages',[])) and first_error is not None
    assert preliminary_path is not None or capacity_stop or failed_before_timing,'Missing preliminary timing without demonstrated terminal failure'
    route=root/'outflow/ti60f225_oob.route.out'
    rt=route.read_text() if route.exists() else ''
    overlaps=re.findall(r'Iter\s+(\d+):\s*overlap=(\d+)\s+.*?cpd=(\d+)',rt)
    stages=end.get('stages',[])
    complete=any(s['flow']=='pnr' and s['exit_code']==0 for s in stages)
    cleanup=root/'pnr_timeout_cleanup.json'
    operator_file=root/'operator_stop.json'
    operator_stop=json.loads(operator_file.read_text()) if operator_file.exists() else None
    evidence=[root/'physical_inputs.json',terminal if terminal.exists() else failure,place,net]
    if timing.exists():evidence.append(timing)
    if place_log.exists():evidence.append(place_log)
    if capacity_file.exists():evidence.append(capacity_file)
    if route.exists():evidence.append(route)
    if cleanup.exists():evidence.append(cleanup)
    if operator_file.exists():evidence.append(operator_file)
    rows.append(dict(root=str(root),base=meta['base'],net_sha256=sha(net),
        terminal_status='OPERATOR_REJECTED_CANDIDATE' if operator_stop else 'CAPACITY_GATE_STOPPED' if capacity_stop else 'PNR_TOOL_COMPLETED' if complete else 'PNR_NOT_COMPLETED',
        operator_stop=operator_stop,
        failure=end.get('error'),first_tool_error=first_error,stages=stages,tool_budget_s=meta['pnr_budget_seconds'],
        actual_xlr=used,capacity=capacity,routing_xlr=routing,
        logic_90pct_limit=capacity*9//10,logic_90pct_excess=used-capacity*9//10,
        ram=ram,ram_capacity=ram_capacity,dsp=dsp,dsp_capacity=dsp_capacity,
        preliminary_path=preliminary_path,capacity_gate=capacity_gate,
        last_route_iteration=dict(zip(('iteration','overlap','cpd_ps'),map(int,overlaps[-1]))) if overlaps else None,
        strict_sta_pass=False,board_release=False,
        cleanup=json.loads(cleanup.read_text()) if cleanup.exists() else capacity_gate.get('cleanup') if capacity_gate else None,
        evidence={str(f):sha(f) for f in evidence}))
assert len({r['net_sha256'] for r in rows})==1,'Physical comparison uses different maps'
data=dict(status='TERMINAL_PHYSICAL_COMPARISON_NOT_RELEASE',rows=rows,
    note='Placement timing and routing iteration estimates are not final routed STA. A budget stop does not prove impossibility of routing.',
    scope='Complete currently implemented board only; gesture/demon/real FPS integration remains open.')
out.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({r['root']:dict(status=r['terminal_status'],xlr=r['actual_xlr'],
    overlap=r['last_route_iteration'],preliminary_slack=r['preliminary_path']['slack_ns'] if r['preliminary_path'] else None) for r in rows},ensure_ascii=False))
