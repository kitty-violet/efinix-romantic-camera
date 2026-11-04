import json,re,collections
from run_units import ROOT

p=ROOT/'tmp/frame-execution/connected-board-continuous-srl-register1/outflow/ti60f225_oob.map.v'
s=p.read_text();rows=[]
for m in re.finditer(r'^\s*EFX_FF\s+(\S+)\s*\((.*?)\) /\*',s,re.M|re.S):
 pins=dict(re.findall(r'\.(\w+)\(([^()]*)\)',m[2]));q=pins.get('Q','')
 if any(x in q for x in ('/owner/','frame_states','committed_frames','dropped_results',
  'pipeline/compute_input','pipeline/compute_output','pipeline/display_input','pipeline/display_output','pipeline/owner_display_id')):
  rows.append(dict(cell=m[1],pins=pins))
out=ROOT/'tmp/board-test/video-service-owner-registers1.json';assert not out.exists()
out.write_text(json.dumps(rows,indent=2))
keys=collections.Counter(re.sub(r'\[[^]]*\]','',r['pins']['Q']).strip() for r in rows)
print(json.dumps(keys,indent=2))
