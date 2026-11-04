"""Rebind only unchanged camera cells; preserve actual old-map test identity."""
from pathlib import Path
source=Path(__file__).with_name('camera_payload_reuse.py')
code=source.read_text(encoding='utf8')
code=code.replace('from camera_payload_board import BOARD,PARENT,audit_delta','from camera_ready_board import BOARD,PARENT,audit_delta\nfrom camera_checked_v2 import BOARD as CAMERA_REFERENCE')
code=code.replace("camera=json.loads((PARENT/'result.json').read_text())['new_cells']","camera=json.loads((CAMERA_REFERENCE/'result.json').read_text())['new_cells']")
code=code.replace("r['map_sha256']==sha(oldpath)","r['map_sha256']==sha(CAMERA_REFERENCE/'outflow/ti60f225_oob.map.v')")
code=code.replace("    for f,h in bound.items():assert sha(f)==h,f", "    parent_receipt=ROOT/'tmp/board-test/camera-payload-reuse1/result.json'\n    parent=json.loads(parent_receipt.read_text())\n    assert parent['status']=='UNCHANGED_CAMERA_MAP_TESTS_BOUND_TO_PAYLOAD_MAP' and parent['map_sha256']==sha(oldpath)\n    assert parent['parent_map_sha256']==sha(CAMERA_REFERENCE/'outflow/ti60f225_oob.map.v')\n    bound[str(parent_receipt)]=sha(parent_receipt);bound.update(parent['evidence'])\n    for f,h in bound.items():assert sha(f)==h,f")
code=code.replace("changed={x['cell'] for r in groups for x in r['loads']}","changed={x['cell'] for r in groups for x in r['loads']}\n    definitions=json.loads((ROOT/'tmp/frame-execution/camera-ready-native4/definitions.json').read_text())\n    changed.update(definitions)")
code=code.replace("result=dict(status='UNCHANGED_CAMERA_MAP_TESTS_BOUND_TO_PAYLOAD_MAP'","result=dict(status='UNCHANGED_CAMERA_MAP_TESTS_BOUND_TO_READY_MAP'")
code=code.replace('lexical=lexical_roundtrip(srctxt,newtxt)','from camera_ready_board import normalized_source\n    lexical=lexical_roundtrip(normalized_source(srctxt,newtxt),newtxt)')
exec(compile(code,str(source)+':ready-bound','exec'),globals())
