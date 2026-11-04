"""Current full-map schema6 observation oracle including all ISP status words."""
from pathlib import Path
source=Path(__file__).with_name('validate_video_service_compact_mapped_pages.py');code=source.read_text()
for x,y in [('from video_service_sample_ram import capture_roots','from video_service_narrow_sample_ram import capture_roots'),('range(4)','range(16)'),("'service_page_part',2","'service_page_part',4"),("'service_pixel_part',2","'service_pixel_part',4"),('128','32'),('192-len(roots)','96-len(roots)'),('<<160','<<64'),('<<168','<<72'),('<<176','<<80'),('<<177','<<81'),('<<178','<<82'),('(4<<72)','(6<<72)'),('All8 original legacy chunks','All32 original legacy words'),('range(2,6)','range(2,7)')]:
 assert x in code,x;code=code.replace(x,y)
old="key,g=pickle.loads((board/'service-validation-graph.pkl').read_bytes());assert key[0]==sha(src)"
new="""key,sg,g,sp,ps=pickle.loads((board/'retained-audit-parse.pkl').read_bytes());assert key[1]==sha(src)
 import inspect
 from audit_continuous_vectors import expand_graph
 ex=inspect.getsource(expand_graph).replace('def expand_graph(text):','def expand_graph(text,cached):').replace('g=graph(text)','g=cached')
 scope={'re':re};exec(compile(ex,'vector-expansion','exec'),scope);g=scope['expand_graph'](src.read_text(),g)
 lineage=ROOT/'tmp/frame-execution/video-service-native64-joint-control/graft.json'
 delta=json.loads((board/'input-isp-map-audit.json').read_text());assert delta['status']=='INPUT_ISP_SOURCE_AND_RETAINED_MAP_PASS' and delta['map_sha256']==sha(src)
 for f,h in delta['evidence'].items():assert sha(f)==h
"""
assert code.count(old)==1;code=code.replace(old,new)
code=code.replace('ps=params(src.read_text());d,l=connections(g)','d,l=connections(g)')
code=code.replace("a.native/'graft.json'","lineage")
code=code.replace(' def mapped(t):'," def mapped(t):\n  if t==net('apb_enable'):\n   assert g['SERVICE62_APB_ENABLE'][0]=='EFX_FF' and g['SERVICE62_APB_ENABLE'][1]['CLK']==[net('process_clk_150')]\n   return g['SERVICE62_APB_ENABLE'][1]['Q'][0]")
anchor=' result=dict(status='
extra=""" # RAM/valid/status FF values are driven at the actual mapped state cut.
 # INIT=1 state is represented by Q inversion in the official zero-init FF;
 # the initialization mask is independently derived from the RTL constants.
 for trial in range(48):
  values.clear();logical=rng.getrandbits(128)&~(1<<47)
  for i in range(128):
   if i==47:continue
   cell='service_isp_table_valid~FF' if i==42 else f'service_isp_status[{i}]~FF'
   assert g[cell][0]=='EFX_FF' and g[cell][1]['CLK']==[net('process_clk_150')],cell
   inverse=int(i<32 or i in (42,44) or (i>=96 and ((0xe483a10c>>(i-96))&1)))
   values[g[cell][1]['Q'][0]]=((logical>>i)&1)^inverse
  seq=rng.randrange(256);setword('service_page_sequence',8,seq);setword('service_page_phase',3,6)
  for n in ['service_pixel_valid','service_pixel_busy','service_pixel_stale']:setword(n,1,int(n=='service_pixel_stale'))
  page=(0x49535031<<480)|logical
  for part in range(16):
   setword('service_page_part',4,part)
   header=((0x56530000|(part<<3)|6)<<32)|(seq<<64)|(6<<72)
   check(header|((page>>(32*part))&0xffffffff),('input_isp',trial,part))
"""
assert code.count(anchor)==1;code=code.replace(anchor,extra+anchor)
code=code.replace("scope='All32 original legacy words;", "scope='All128 ISP status bits/zero padding/magic; All32 original legacy words;")
exec(compile(code,str(source)+':schema6-input-isp-oracle','exec'),globals())
