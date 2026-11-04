"""Exhaust the normal-DW column-stream schedule before implementing its RTL.
Eight staggered pixels per row, period9. Main stream reads phase0/1 twice at
the first output column; owner0's kx0 uses the saved preceding column. Then
kernel rows use delays0/3/6 and no per-owner9-word reload or RAM crossbar.
"""
import argparse,json
from pathlib import Path
from run_units import sha

def stream_x(t):
 assert t>=0
 b,p=divmod(t,9)
 return b*8+max(0,p-1)

def source_x(output_x,tap):
 b,p=divmod(output_x,8);ky,kx=divmod(tap,3)
 t=b*9+p+tap
 return b*8-1 if p==0 and kx==0 else stream_x(t-3*ky)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
 checks=0;special=0
 for w in range(1,321):
  for ox in sorted({0,(320-w)//2,320-w}):
   for x in range(w):
    for tap in range(9):
     expected=max(0,min(319,ox+x+tap%3-1))
     got=max(0,min(319,ox+source_x(x,tap)))
     assert got==expected,(w,ox,x,tap,got,expected)
     checks+=1;special+=int(x%8==0 and tap%3==0)
  b=(w+7)//8;last_p=(w-1)%8
  last_cycle=9*(b-1)+last_p+8
  assert last_cycle+1==9*b+last_p
 result=dict(status='EXHAUSTIVE_NORMAL_DW_COLUMN_STREAM_ADDRESS_IDENTITY_PASS',address_checks=checks,special_left_samples=special,
  widths=[1,320],main_columns_per_group='9*ceil(width/8)+(width-1)%8, plus one warmup',
  delays=[0,3,6],saved_left_column_rule='owner p=0 and tap kx=0 uses preceding-column sample saved at phase0',
  implemented_rtl=False,real_ports=False,full_frame=False,source_sha256=sha(__file__))
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
