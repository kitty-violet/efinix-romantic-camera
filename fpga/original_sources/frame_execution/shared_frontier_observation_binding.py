"""Bind the APB projection through the proved, unchanged sequential boundary."""
import json,pickle
from pathlib import Path
from run_units import sha

def bind_apb(board,native,g,ps):
    current=native/'shared-frontier-independent-audit.json'
    c=json.loads(current.read_text())
    assert c['status']=='SHARED_FRONTIER_COMPOSITIONAL_CEC_PASS_NOT_PHYSICAL'
    assert c['source_sha256']==sha(board/'native_source.v')
    assert c['all_original_sequential_state_exact'] and c['extra_cycles']==0
    for f,h in c['inputs'].items():assert sha(f)==h,f
    parent=Path(json.loads((native/'shared-frontier-edit.json').read_text())['parent'])
    previous=parent/'same-edge-independent-audit.json';p=json.loads(previous.read_text())
    assert p['source_sha256']==c['parent_sha256']==sha(parent/'native_source.v')
    assert p['status']=='SAME_EDGE_ENABLE_COMPOSITIONAL_EQUIVALENCE_PASS_NOT_PHYSICAL'
    assert p['legal_state_induction'] and p['external_cycles_added']==0
    for f,h in p['inputs'].items():assert sha(f)==h,f
    k,a,pa=pickle.loads((parent/'zero-source-parse.pkl').read_bytes())
    l,b,pb=pickle.loads((native/'zero-source-parse.pkl').read_bytes())
    assert k==c['parent_sha256'] and l==c['source_sha256']
    n='SERVICE62_APB_ENABLE';assert a[n]==b[n] and pa[n]==pb[n]
    mapped=board/'outflow/ti60f225_oob.map.v';retained=board/'retained-map-audit.json';r=json.loads(retained.read_text())
    assert r['status']=='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS' and r['map_sha256']==sha(mapped)
    kind,pins=g[n];assert kind=='EFX_FF' and pins['Q']==[('net','apb_enable',None)]
    assert pins['CLK']==[('net','process_clk_150',None)]
    assert ps[n]['CLK_POLARITY']==(1,1) and ps[n]['SR_VALUE']==(1,0)
    return dict(old_cell='LUT__133025',current_cell=n,token=pins['Q'][0],legal_reset_same_edge_equivalence=True,cycles_added=0,
      evidence={str(f):sha(f) for f in (current,previous,native/'shared-frontier-edit.json',retained,mapped,Path(__file__))})
