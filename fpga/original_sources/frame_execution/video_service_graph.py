"""One hash-bound native graph cache for the current independent refactor.

Only caches locally derived parsing. Does not cache test or release outcomes.
"""
import json,pickle,time
from run_units import ROOT,sha
from native_local_drivers import raw_cells
from audit_continuous_vectors import expand_graph
from probe_full_native_split_map import params
from continuous_control_native import connections

SOURCE=ROOT/'tmp/frame-execution/connected-board-continuous-srl-register1/native_source.v'
CACHE=ROOT/'tmp/board-test/video-service-native-cache1'

def load():
 h=sha(SOURCE); signature={str(p):sha(p) for p in [SOURCE,__file__,
  ROOT/'hw/efinity/prototypes/frame_execution/native_local_drivers.py',
  ROOT/'hw/efinity/prototypes/frame_execution/audit_continuous_vectors.py',
  ROOT/'hw/efinity/prototypes/frame_execution/probe_full_native_split_map.py',
  ROOT/'hw/efinity/prototypes/frame_execution/continuous_control_native.py']}
 if (CACHE/'binding.json').exists():
  r=json.loads((CACHE/'binding.json').read_text())
  assert {str(__import__('pathlib').Path(k).resolve()):v for k,v in r['inputs'].items()}=={str(__import__('pathlib').Path(k).resolve()):v for k,v in signature.items()}
  assert r['cache_sha256']==sha(CACHE/'graph.pkl')
  with (CACHE/'graph.pkl').open('rb') as f:return pickle.load(f)
 assert not CACHE.exists();CACHE.mkdir();t=time.monotonic();s=SOURCE.read_text()
 raw=raw_cells(s);g=expand_graph(s);ps=params(s);drivers,loads=connections(g)
 data=dict(source=str(SOURCE),sha256=h,raw=raw,g=g,ps=ps,drivers=drivers,loads=loads)
 with (CACHE/'graph.pkl').open('wb') as f:pickle.dump(data,f,protocol=4)
 (CACHE/'binding.json').write_text(json.dumps(dict(inputs=signature,cache_sha256=sha(CACHE/'graph.pkl'),seconds=time.monotonic()-t),indent=2))
 return data

if __name__=='__main__':
 d=load();terms=['cpu/boot_valid','video_processing_pipeline/pipeline/display/start_fire',
 'camera_burst_capture/start_fire','video_processing_pipeline/pipeline/capture/start_fire',
 'video_processing_pipeline/pipeline/owner/n2764','video_processing_pipeline/pipeline/compute_active',
 'video_processing_pipeline/pipeline/prod_frame_valid','cap_ready','n78042_2']
 for t in terms:
  n=d['drivers'].get(('net',t,None));print(t,n,d['raw'][n][:2] if n else None,d['ps'].get(n),flush=True)
