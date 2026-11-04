"""Run the existing full service crossing inventory on the actual color map.

The service cell classification comes from the released64 graft only after the
complete public-graph delta proves that no service cell/pin/parameter changed.
The current map is inspected; an old inventory is not relabelled as current.
"""
from pathlib import Path

source=Path(__file__).with_name('audit_video_service_cdc.py')
code=source.read_text()
old="key,g=pickle.loads((board/'service-validation-graph.pkl').read_bytes());assert key[0]==sha(src)"
new="""key,g,ps=pickle.loads((board/'color-map-parse.pkl').read_bytes());assert key==sha(src)
 # Use the established vector-expansion algorithm with the hash-checked
 # already-parsed graph. A whole vector must never become a fake external pin.
 import inspect
 from audit_continuous_vectors import expand_graph
 expansion=inspect.getsource(expand_graph)
 assert expansion.count('def expand_graph(text):')==1 and expansion.count('g=graph(text)')==1
 expansion=expansion.replace('def expand_graph(text):','def expand_graph(text,cached_graph):').replace('g=graph(text)','g=cached_graph')
 import re
 scope={'re':re};exec(compile(expansion,'verified-vector-expansion','exec'),scope)
 g=scope['expand_graph'](src.read_text(),g)
 delta=json.loads((board/'color-map-audit.json').read_text())
 assert delta['status']=='COLOR_EXPOSURE_CURRENT_MAP_COMPOSITION_PASS'
 assert delta['all_other_public_cells_and_parameters_identical'] and delta['changed_public_cells']==17
 assert delta['source_sha256']==sha(a.native/'native_source.v') and delta['map_sha256']==sha(src)
 for f,h in delta['evidence'].items():assert sha(f)==h,f
 classification=ROOT/'tmp/frame-execution/video-service-native64-joint-control/graft.json'
"""
assert code.count(old)==1
code=code.replace(old,new)
old="json.loads((a.native/'graft.json').read_text())['removed']"
assert code.count(old)==1
code=code.replace(old,"json.loads(classification.read_text())['removed']")
code=code.replace("d,l=connections(g);ps=params(src.read_text());active=set()", "d,l=connections(g);active=set()")
code=code.replace('from video_service_sample_ram import banks,storage_structure,bit_tokens',
                  'from video_service_narrow_sample_ram import banks,storage_structure,bit_tokens')
old="hardware_access=False,board_release=False,checker_sha256=sha(__file__))"
new="""hardware_access=False,board_release=False,checker_sha256=sha(__file__),
  classification_evidence={str(f):sha(f) for f in (classification,board/'color-map-audit.json',Path(__file__),HERE/'audit_video_service_cdc.py',HERE/'audit_continuous_vectors.py')})"""
assert code.count(old)==1
code=code.replace(old,new)
exec(compile(code,str(source)+':color-exposure-current-map','exec'),globals())
