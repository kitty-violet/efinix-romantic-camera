"""Inventory the current ISP map and explicitly include all new RAM inputs."""
from pathlib import Path
source=Path(__file__).with_name('audit_video_service_cdc.py');code=source.read_text()
old="key,g=pickle.loads((board/'service-validation-graph.pkl').read_bytes());assert key[0]==sha(src)"
new="""key,source_graph,g,source_params,ps=pickle.loads((board/'retained-audit-parse.pkl').read_bytes());assert key[1]==sha(src)
 import inspect,re
 from audit_continuous_vectors import expand_graph
 expansion=inspect.getsource(expand_graph).replace('def expand_graph(text):','def expand_graph(text,cached_graph):').replace('g=graph(text)','g=cached_graph')
 scope={'re':re};exec(compile(expansion,'verified-vector-expansion','exec'),scope);g=scope['expand_graph'](src.read_text(),g)
 delta=json.loads((board/'input-isp-map-audit.json').read_text())
 assert delta['status']=='INPUT_ISP_SOURCE_AND_RETAINED_MAP_PASS' and delta['map_sha256']==sha(src)
 for f,h in delta['evidence'].items():assert sha(f)==h
 classification=ROOT/'tmp/frame-execution/video-service-native64-joint-control/graft.json'
"""
assert code.count(old)==1;code=code.replace(old,new)
code=code.replace("json.loads((a.native/'graft.json').read_text())['removed']","json.loads(classification.read_text())['removed']")
code=code.replace("d,l=connections(g);ps=params(src.read_text());active=set()","d,l=connections(g);active=set()")
code=code.replace('from video_service_sample_ram import banks,storage_structure,bit_tokens','from video_service_narrow_sample_ram import banks,storage_structure,bit_tokens')
anchor=" result=dict(status='VIDEO_SERVICE_NEW_CDC_INVENTORIED_NOT_RELEASE'"
extra=""" isp_ram=[]
 for n,(kind,pins) in g.items():
  if not n.startswith('service_isp/') or kind not in ('EFX_RAM10','EFX_DPRAM10'):continue
  assert kind=='EFX_RAM10' and pins['WCLK']==pins['RCLK']==[net('process_clk_150')]
  inspected=0;external=[]
  for pin in ('WDATA','WE','WADDREN','WADDR','RE','RADDREN','RADDR','RST'):
   for t in bit_tokens(pins[pin]):
    for k,s,b in sources(t):
     if k=='external':
      external.append(dict(pin=pin,source=s,bit=b));continue
     sk,sp=g[s];clocks=[sp[q] for q in (('RCLK',) if sk in ('EFX_RAM10','EFX_DPRAM10') else ('CLK',)) if q in sp]
     assert clocks and all(c==[net('process_clk_150')] for c in clocks),('ISP RAM crossing',n,pin,s,clocks)
     inspected+=1
  isp_ram.append(dict(cell=n,clock='process_clk_150',inspected_source_paths=inspected,external_inputs=external))
 assert len(isp_ram)==3
"""
assert code.count(anchor)==1;code=code.replace(anchor,extra+anchor)
code=code.replace("hardware_access=False,board_release=False,checker_sha256=sha(__file__))","hardware_access=False,board_release=False,checker_sha256=sha(__file__),isp_single_clock_ram=isp_ram,classification_evidence={str(f):sha(f) for f in (classification,board/'input-isp-map-audit.json',Path(__file__),HERE/'audit_video_service_cdc.py',HERE/'audit_continuous_vectors.py')})")
exec(compile(code,str(source)+':input-isp-current-map','exec'),globals())
