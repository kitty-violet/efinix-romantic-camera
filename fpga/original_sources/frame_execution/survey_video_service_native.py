"""Read-only inventory of replacement boundaries in the frozen full design."""
import collections,json
from run_units import ROOT,HERE,sha
from native_local_drivers import graph
from camera_joint_native import OUT

def main():
 src=ROOT/'tmp/frame-execution/connected-board-continuous-srl-register1/outflow/ti60f225_oob.map.v'
 text=src.read_text();g=graph(text);drivers=collections.defaultdict(list);loads=collections.defaultdict(list)
 for n,(kind,pins) in g.items():
  for pin,terms in pins.items():
   for t in terms:
    if t[0]=='net':(drivers if pin in OUT.get(kind,()) else loads)[t].append((n,pin))
 prefixes={'owner':'video_processing_pipeline/pipeline/owner/',
 'reader':'video_processing_pipeline/pipeline/reader/',
 'display':'hdmi_scheduler/','boot':'preview_boot/','capture':'camera_burst_capture/'}
 scopes={}
 for label,prefix in prefixes.items():
  seq={n for n,(k,p) in g.items() if n.startswith(prefix) and k in ('EFX_FF','EFX_RAM10','EFX_DPRAM10','EFX_SRL8')}
  scopes[label]=dict(prefix=prefix,kinds=dict(collections.Counter(g[n][0] for n in seq)),
   state_outputs=[dict(cell=n,outputs={p:ts for p,ts in g[n][1].items() if p in OUT.get(g[n][0],())}) for n in sorted(seq)])
 names=sorted({t[1] for t in set(drivers)|set(loads) if any(k in t[1] for k in
  ('frame_states','owner_','compute_start','compute_done','display_done','compute_input','compute_output',
   'display_input','display_output','capture_ready','frame_request','down_valid','down_ready','fixed_boot','camera_abort',
   'upscale','hdmi_','scheduler','key','button')) and not t[1].startswith('protected_inst/')})
 names=[n for n in names if '~' not in n]
 out=ROOT/'tmp/board-test/video-service-native-survey1.json';assert not out.exists()
 result=dict(map_sha256=sha(src),scopes=scopes,names=names,hardware_access=False)
 out.write_text(json.dumps(result,indent=2));print(json.dumps(dict(scopes={k:v['kinds'] for k,v in scopes.items()},names=names),indent=2))
if __name__=='__main__':main()
