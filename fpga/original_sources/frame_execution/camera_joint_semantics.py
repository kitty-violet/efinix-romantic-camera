"""Compose two proven reimport name maps for old semantic probe names."""
import json
from camera_joint_board import BOARD,PARENT,TRANSFORM
from run_units import sha
def renames():
 before=json.loads((PARENT/'result.json').read_text());after=json.loads((BOARD/'result.json').read_text())
 assert before['map_sha256']==sha(PARENT/'outflow/ti60f225_oob.map.v')
 assert after['map_sha256']==sha(BOARD/'outflow/ti60f225_oob.map.v')
 proof=json.loads((TRANSFORM/'result.json').read_text())
 assert proof['inputs'][str(PARENT/'outflow/ti60f225_oob.map.v')]==before['map_sha256']
 old=json.loads((PARENT/'public_connectivity_audit.json').read_text())['renamed_nets']
 new=json.loads((BOARD/'public_connectivity_audit.json').read_text())['renamed_nets']
 return {**new,**{k:new.get(v,v) for k,v in old.items()}}
