"""Actual placement geometry of the measured cache-address/weight path.

Distances describe placed endpoints, never available tracks or routed delay.
"""
import argparse,json,pickle
from pathlib import Path
from run_units import ROOT,sha
from fixed_control_ddr_reuse_locations import locations
from continuous_control_native import connections

def main():
 p=argparse.ArgumentParser()
 for n in ('board','physical','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();b=a.board.resolve();phy=a.physical.resolve();out=a.output.resolve()
 assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 mapped=b/'outflow/ti60f225_oob.map.v';assert sha(mapped)==sha(phy/'outflow/ti60f225_oob.map.v')
 key,src,g,sp,mp=pickle.loads((b/'retained-audit-parse.pkl').read_bytes());assert key[1]==sha(mapped)
 xy=locations(phy/'outflow/ti60f225_oob.out.placeloc');_,loads=connections(g)
 pre='video_processing_pipeline/pipeline/cnn/core/dut/engine/mac/'
 rows=[]
 for bit in range(8):
  n=pre+f'cache/TERM[2].CHANNEL[3].BIT[{bit}].storage';assert g[n][0]=='EFX_SRL8' and n in xy
  sinks=sorted(loads[g[n][1]['Q'][0]]);assert len(sinks)==4 and all(pin=='B' for _,pin in sinks)
  dst=[]
  for cell,pin in sinks:
   assert cell in xy
   dst.append(dict(cell=cell,pin=pin,position=xy[cell],manhattan_xy=sum(abs(x-y) for x,y in zip(xy[n][:2],xy[cell][:2]))))
  rows.append(dict(cell=n,bit=bit,position=xy[n],sinks=dst))
 path=[pre+'cache/active_parts[2]~FF','LUT__139898_LUT__139900_splitO','LUT__139908','LUT__139914','LUT__139919','LUT__139920',pre+'cache/TERM[2].CHANNEL[3].BIT[1].storage',pre+'arithmetic/ROW[2].CHANNEL[3].TAP[2].mac']
 assert all(n in xy for n in path)
 result=dict(status='CACHE_WEIGHT_PLACED_GEOMETRY_ONLY',map_sha256=sha(mapped),physical=str(phy),
  measured_seed3_path_cells=[dict(cell=n,position=xy[n]) for n in path],weight_byte=rows,
  evidence={str(f):sha(f) for f in (Path(__file__).resolve(),mapped,phy/'outflow/ti60f225_oob.out.placeloc')},
  scope='Same four DSP loads and eight cache bits. XY spans are geometry only; original routed timing and full congestion/clock/control review remain necessary.',hardware_access=False,board_release=False)
 out.write_text(json.dumps(result,indent=2));print(json.dumps(dict(status=result['status'],critical_bit=rows[1])))
if __name__=='__main__':main()
