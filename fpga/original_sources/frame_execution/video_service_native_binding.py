"""Resolve optimized native aliases before attaching new service logic.

The public name left in a mapped declaration is not proof of a driver.  The
bindings below are checked against retained consumers of the original graph.
"""
import re
from native_local_drivers import render_token, REF

P='video_processing_pipeline/pipeline/'
def net(name,bit=None):return ('net',name,bit)

def bind(code,D):
 g,drivers=D['g'],D['drivers']
 production=net(P+'cnn/dp_error');capture=net('cap_error')
 # These are the *actual* native inputs to the old compute terminal and APB
 # pipeline status multiplexers, not guessed hierarchical RTL aliases.
 assert g['LUT__133211'][1]['I0']==[production]
 assert g['LUT__161906'][1]['I1']==[production]
 assert g['LUT__161885'][1]['I2']==[capture]
 assert production in drivers and capture in drivers
 assert net('bringup_pipeline_status',2) not in drivers
 assert net('bringup_pipeline_status',1) not in drivers
 status='{bringup_pipeline_status[3],'+render_token(production)+','+render_token(capture)+',bringup_pipeline_status[0]}'
 code=code.replace('bringup_pipeline_status[3:0]',status)
 code=code.replace('bringup_pipeline_status[2]',render_token(production))
 code=code.replace('bringup_pipeline_status[1]',render_token(capture))
 for name,width in [('hdmi_timing_x',12),('hdmi_timing_y',11)]:
  assert all(net(name,b) in drivers for b in range(width))
  assert all(net(name,b) not in drivers for b in range(width,16))
  def padded(m):
   if m['id'] and m['id'].lstrip('\\')==name and m['hi'] is None:
    return '{'+f"{16-width}'d0,"+render_token(net(name))+f'[{width-1}:0]'+'}'
   return m[0]
  code=REF.sub(padded,code)
 # Reader uses 128-bit AXI beats and 3840-byte RGB rows.  These bits were
 # constant-folded before the service graft; retain the exact aligned value.
 for name,low in [(P+'disp_araddr',4),(P+'display/line_offset',8)]:
  assert all(net(name,b) in drivers for b in range(low,32))
  assert all(net(name,b) not in drivers for b in range(low))
  old="32'("+render_token(net(name))+')'
  assert code.count(old)==1,(name,code.count(old))
  code=code.replace(old,'{'+render_token(net(name))+f"[31:{low}],{low}'d0"+'}')
 return code,dict(production_error=list(production),capture_error=list(capture),
  production_reference_consumers=['LUT__133211.I0','LUT__161906.I1'],
  capture_reference_consumer='LUT__161885.I2',address_low_zero_bits=4,row_offset_low_zero_bits=8)

def native_reads(code,source,D,supplied,removed=()):
 """Reject reads of a native declaration whose specific bits have no source."""
 source=re.sub(r'/\*.*?\*/','',source,flags=re.S)
 decl={}
 for m in re.finditer(r'^\s*(?:wire|input|output|inout)\s+(?:\[(\d+):(\d+)\]\s*)?([^;]+);',source,re.M):
  for name in m[3].split(','):
   name=name.strip().lstrip('\\')
   if not re.search(r'\s',name):decl[name]=(int(m[1]),int(m[2])) if m[1] else None
 active={t for t,n in D['drivers'].items() if n not in removed}|set(supplied)
 # This is an original, unchanged encrypted DDR-controller output, not a
 # public primitive. Its exact boundary connection is separately bound by
 # the protected-boundary audit and the existing clock/reset evidence.
 reset='u_ddr3_top/controller/u_efx_ddr3_soft_controller/genblk1.inst_ddr3_axi/inst_efx_ddr3/top_mc/clk_rstn'
 protected=source[source.index('    protected_cell protected_inst ('):]
 assert '.'+render_token(net(reset))+'('+render_token(net(reset))+')' in protected
 active.add(net(reset))
 # Existing continuous drivers (reset aliases) are explicit source statements.
 for expr in re.findall(r'^\s*assign\s+([^=]+)=',source,re.M):
  for m in REF.finditer(expr):
   if not m['id']:continue
   name=m['id'].lstrip('\\');rng=decl.get(name)
   if m['hi'] is not None:rng=(int(m['hi']),int(m['lo'] or m['hi']))
   active.update(net(name,b) for b in range(rng[1],rng[0]+1)) if rng else active.add(net(name))
 for expr in re.findall(r'^\s*(?:input|inout)\s+([^;]+);',source,re.M):
  names=re.sub(r'^\s*\[\d+:\d+\]\s*','',expr)
  for name in names.split(','):
   name=name.strip().lstrip('\\');rng=decl[name]
   active.update(net(name,b) for b in range(rng[1],rng[0]+1)) if rng else active.add(net(name))
 reads=set()
 for m in REF.finditer(re.sub(r'//[^\n]*','',code)):
  if not m['id']:continue
  name=m['id'].lstrip('\\')
  if name not in decl:continue
  rng=decl[name]
  if m['hi'] is not None:rng=(int(m['hi']),int(m['lo'] or m['hi']))
  reads.update(net(name,b) for b in range(rng[1],rng[0]+1)) if rng else reads.add(net(name))
 missing=sorted(reads-active,key=str)
 assert not missing,('undriven native graft bits',missing)
 return dict(native_referenced_bits=len(reads),missing_driver_bits=[],driver_check='native primitives, explicit assignments, top inputs or explicit service exports')
