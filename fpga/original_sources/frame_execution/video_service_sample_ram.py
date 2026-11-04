"""Exact capture-bit and dual-clock RAM structure from the actual mapped graph."""
import re

def number(x):return x[1] if isinstance(x,tuple) else int(x)

def bit_tokens(ts):
 result=[]
 for t in ts:
  if t[0]!='literal':result.append(t);continue
  m=re.fullmatch(r"(\d+)'[sS]?([hbd])([0-9a-fA-F]+)",t[1]);assert m,t
  width=int(m[1]);value=int(m[3],{'h':16,'b':2,'d':10}[m[2]])
  result.extend(('literal',f"1'b{value>>i&1}",None) for i in reversed(range(width)))
 return result

def banks(g,prefix):
 result={}
 for n,(kind,pins) in g.items():
  if not n.startswith(prefix+'/storage/banks['):continue
  m=re.fullmatch(re.escape(prefix)+r'/storage/banks\[(\d+)\][/.]sample',n)
  assert m and kind=='EFX_RAM10',(n,kind)
  result[int(m[1])]=(n,pins)
 return result

def capture_roots(g,ps):
 memory=banks(g,'service_serializer')
 if memory:
  assert set(memory)==set(range(10)),sorted(memory)
  roots={}
  for b,(n,pins) in memory.items():
   data=bit_tokens(pins['WDATA'])
   assert number(ps[n]['WRITE_WIDTH'])==20 and len(data)==20
   for offset,t in enumerate(reversed(data)):
    bit=b*20+offset
    if bit<192:roots[bit]=(t,False)
    else:assert t==('literal',"1'b0",None),(n,bit,t)
  assert len(roots)==192
  return roots
 roots={}
 for n,(k,pins) in g.items():
  m=re.fullmatch(r'service_serializer/held\[(\d+)\]~FF',n)
  if m:roots[int(m[1])]=(pins['D'][0],ps[n]['D_POLARITY']==(1,0))
 assert len(roots)==147 and set(range(128))<=set(roots)
 return roots

def storage_structure(g,ps):
 def zero(ts):
  return bool(ts) and all(t[0]=='literal' and re.fullmatch(r"\d+'[sS]?[hbd]0+",t[1]) for t in ts)
 def scalar(ts,value):return ts==[('literal',f"1'b{value}",None)]
 rows=[]
 configs=[('service_serializer',192,'process_clk_150','jtag_inst1_TCK'),
          ('service_pixel_snapshot',128,'hdmi_tx_slow_clk','process_clk_150')]
 for prefix,width,wclk,rclk in configs:
  memory=banks(g,prefix);assert set(memory)==set(range((width+19)//20)),(prefix,sorted(memory))
  for b,(n,p) in memory.items():
   v=ps[n]
   assert number(v['WRITE_WIDTH'])==number(v['READ_WIDTH'])==20
   assert number(v.get('OUTPUT_REG',(1,0)))==0
   assert v['WRITE_MODE']=='"READ_UNKNOWN"',(n,v.get('WRITE_MODE'))
   assert p['WCLK']==[('net',wclk,None)] and p['RCLK']==[('net',rclk,None)]
   for name in ('WADDR','RADDR','RST'):assert zero(p[name]),(n,name,p[name])
   for name in ('WCLKE','WADDREN','RADDREN'):assert scalar(p[name],1),(n,name,p[name])
   assert len(p['WE'])==2 and p['WE'][0]==p['WE'][1] and len(p['RE'])==1
   for name,default in [('WCLK_POLARITY',1),('RCLK_POLARITY',1),('WCLKE_POLARITY',1),('WADDREN_POLARITY',1),('RADDREN_POLARITY',1),('WE_POLARITY',3),('RE_POLARITY',1),('RST_POLARITY',1)]:
    assert number(v.get(name,(2 if name=='WE_POLARITY' else 1,default)))==default,(n,name)
   assert all(number(value)==0 for name,value in v.items() if name.startswith('INIT_'))
   rows.append(dict(cell=n,bank=b,owner=prefix,source_clock=wclk,read_clock=rclk,
    write_enable=p['WE'][0],read_enable=p['RE'][0],wdatas=p['WDATA'],rdatas=p['RDATA']))
 assert len(rows)==17
 return rows
