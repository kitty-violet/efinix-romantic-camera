"""Require real destination FF pairs, not SRLs, for every new CDC bit."""
import re
from audit_local_reset_implementation import norm

def cells(text):
 return {norm(n):(k,{x:norm(v) for x,v in re.findall(r'\.(\w+)\(([^()]*)\)',p)},a)
  for k,n,p,a in re.findall(r'^\s*(EFX_\w+)\s+(\S+)\s*\((.*?)\) /\*([^\n]*)',text,re.M|re.S)}

def audit(text,prefix='',system='system_clk',tck='tck'):
 c=cells(text);checked=[]
 for first,second,clock in [('request_meta','request_sync',system),('ack_meta','ack_sync',tck)]+[(f'raw_meta[{i}]',f'raw_sync[{i}]',tck) for i in range(8)]:
  a,b=(prefix+x+'~FF' for x in (first,second))
  for n in (a,b):
   kind,p,attrs=c[n]
   assert kind=='EFX_FF' and p['CLK']==clock and 'async_reg=' in attrs,(n,kind,p,attrs)
   assert p.get('CE')=="1'b1" and p.get('SR')=="1'b0",(n,p)
  assert c[b][1]['D']==c[a][1]['Q'],(a,b)
  checked.append(dict(first=a,second=b,clock=clock))
 return dict(status='SNAPSHOT_20_FF_SYNC_NO_SRL_PASS',synchronizer_ff=20,pairs=checked)
