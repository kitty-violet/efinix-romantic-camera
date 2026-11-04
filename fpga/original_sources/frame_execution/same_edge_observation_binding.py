"""Bind the old APB observation cut to its independently proved same-edge FF.

Only changes the observation oracle boundary; no production selector or
transport logic is substituted. Legal-reset equivalence is proved separately.
"""
import json
from pathlib import Path
from run_units import sha

def bind_apb(board,native,g,ps):
    proof=native/'same-edge-independent-audit.json'
    p=json.loads(proof.read_text())
    assert p['status']=='SAME_EDGE_ENABLE_COMPOSITIONAL_EQUIVALENCE_PASS_NOT_PHYSICAL'
    assert p['source_sha256']==sha(board/'native_source.v')
    assert p['legal_state_induction'] and p['external_cycles_added']==0
    for f,h in p['inputs'].items():assert sha(f)==h,f
    mapped=board/'outflow/ti60f225_oob.map.v'
    retained=board/'retained-map-audit.json';r=json.loads(retained.read_text())
    assert r['status']=='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS' and r['map_sha256']==sha(mapped)
    n='SERVICE62_APB_ENABLE';kind,pins=g[n]
    assert kind=='EFX_FF' and pins['Q']==[('net','apb_enable',None)]
    assert pins['CLK']==[('net','process_clk_150',None)]
    assert ps[n]['CLK_POLARITY']==(1,1) and ps[n]['SR_VALUE']==(1,0)
    return dict(old_cell='LUT__133025',current_cell=n,token=pins['Q'][0],
        legal_reset_same_edge_equivalence=True,cycles_added=0,
        evidence={str(f):sha(f) for f in (proof,retained,mapped,Path(__file__))})
