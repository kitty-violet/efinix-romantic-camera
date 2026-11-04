"""Capacity-only alternatives; never a speed claim or replacement RTL result."""
import json
from pathlib import Path
from fourway_frame_program import ROOT,HERE,compile_tile
from model_contract import compile_contract,load_contract
from hardware_schedule_model import build_tile_manifest
from run_units import sha
def main():
 out=ROOT/'tmp/frame-execution/fourway-row4-capacity1.json';assert not out.exists()
 plan=compile_contract(load_contract());group=next(g for g in build_tile_manifest(320)['groups'] if g['name']=='residual');results=[]
 for width in (80,76,72,68,64,60,56):
  try:
   tiles=[]
   for ti,x in enumerate(range(0,320,width)):
    tile=dict(tile_index=ti,commit_interval=dict(x0=x,x1=min(x+width,320)-1))
    tiles.append(compile_tile(plan,group,tile,-1,4))
   results.append(dict(width=width,status='CAPACITY_PASS_NOT_RTL',tiles=len(tiles),commands=sum(len(t['commands']) for t in tiles),
    high=[max(t['finite_high_water'][i] for t in tiles) for i in range(3)],peak=max(t['peak_live_padded_bytes'] for t in tiles),
    words=sum(sum(v for k,v in t['word_counts'].items() if k not in ('source','commit')) for t in tiles)))
  except MemoryError as e:results.append(dict(width=width,status='CAPACITY_FAIL',error=str(e)))
 r=dict(status='RESIDUAL_ROW4_RETILE_CAPACITY_ONLY',results=results,rtl_executed=False,measured_frame_ms=None,
   evidence={str(p):sha(p) for p in (Path(__file__),HERE/'fourway_frame_program.py',HERE/'compile_frame.py',ROOT/'network_ab/model_b_contract.json')})
 out.write_text(json.dumps(r,indent=2));print(json.dumps(r))
if __name__=='__main__':main()

