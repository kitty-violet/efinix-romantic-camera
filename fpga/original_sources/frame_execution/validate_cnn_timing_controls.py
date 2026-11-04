"""Retest the actual reimported writer CE graph, not only pre-map primitives."""
from pathlib import Path
from run_units import ROOT,HERE
from control_command_repair import once
SOURCE=ROOT/'tmp/board-test/test-cnn-budget-writer-control.py'
code=SOURCE.read_text()
code=once(code,"root=R/'tmp/frame-execution/cnn-budget-writer-control-short1'","root=R/'tmp/frame-execution/cnn-timing-closure-controls1'")
code=once(code,"old=oldfile.read_text();new=(base/'candidate.v').read_text();proof=", "from cnn_timing_closure_board import BOARD,audit_delta\naudit_delta(BOARD)\nold=oldfile.read_text();new=(BOARD/'outflow/ti60f225_oob.map.v').read_text();proof=")
code=once(code,"oldlut={p['O'][0]", "sg=graph((BOARD/'native_source.v').read_text());mapping={}\nfor n,(k,ps) in sg.items():\n    if n not in ng:continue\n    for p in ps.keys()&ng[n][1].keys():\n        for u,v in zip(ps[p],ng[n][1][p]):\n            assert u not in mapping or mapping[u]==v,(n,p,u)\n            mapping[u]=v\nnewstops={mapping[t] for t in stops if t in mapping}\noldlut={p['O'][0]")
code=once(code,'if t in stops or t not in drivers:', 'if t in (stops if g is og else newstops) or t not in drivers:')
code=once(code,"a,b=tuple(row['old']),tuple(row['new']);", "a,b=tuple(row['old']),mapping[tuple(row['new'])];")
code=once(code,'assert nl<=ol and rootnet in ol','assert nl<={mapping[t] for t in ol if t in mapping} and rootnet in ol')
code=once(code,"wires={t:f'vector[{i}]' for i,t in enumerate(leaves)}", "wires={(t if g is og else mapping[t]):f'vector[{i}]' for i,t in enumerate(leaves) if g is og or t in mapping}")
code=once(code,"Path(__file__),base/'result.json'", "Path(__file__),SOURCE,BOARD/'result.json',BOARD/'outflow/ti60f225_oob.map.v',BOARD/'public_connectivity_audit.json',base/'result.json'")
code=code.replace('native_candidate_not_reimported=True','native_candidate_not_reimported=False')
code=code.replace('NATIVE_WRITER_ERROR_COFACTOR_OFFICIAL_LUT_SHORT_PASS_NOT_MAPPED_RELEASE','MAPPED_WRITER_ERROR_COFACTOR_OFFICIAL_LUT_SHORT_PASS')
code=code.replace('No universal four-state equivalence; no new map/physical or board pass.','No universal four-state equivalence, physical or board pass.')
exec(compile(code,str(SOURCE)+':actual-mapped-writer-controls','exec'),globals())
