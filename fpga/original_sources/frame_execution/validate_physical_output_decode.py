"""Exhaustively invert every unique production output layout, independently of RTL data."""
import argparse, json, hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--vectors',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
a.root.mkdir(parents=True,exist_ok=False)
program=a.vectors/'program.hex'; layouts=set();checked=0;invalid=0
for line in program.read_text().splitlines():
    n=int(line,16);f=[(n>>(i*32))&0xffffffff for i in range(32)]
    if f[0]!=2:continue
    layouts.add((f[2],f[3],f[5],f[13],f[14]|(f[15]<<32)))
for pixels,rows,groups,planes,bases in layouts:
    destinations={}
    for y in range(rows):
        for x in range(pixels):
            for g in range(groups):
                port=(((planes>>(2*y))&3)*4+x%4)*2+1
                address=((bases>>(10*y))&1023)+(x//4)*groups+g
                assert 0<=address<1024 and (port,address) not in destinations
                destinations[port,address]=(y,x,g)
    for port in range(24):
        for address in range(1024):
            found=[]
            if port%2==1:
                for row in range(rows):
                    if ((planes>>(2*row))&3)!=port//8:continue
                    off=address-((bases>>(10*row))&1023)
                    if off>=0:
                        col=(off//groups)*4+(port//2)%4
                        if col<pixels:found.append((row,col,off%groups))
            target=destinations.get((port,address))
            assert found==([target] if target else []),(pixels,rows,groups,port,address,found,target)
            checked+=target is not None;invalid+=target is None
result=dict(status='PHYSICAL_OUTPUT_COORDINATE_EXHAUSTIVE_PASS',unique_layouts=len(layouts),valid_destinations=checked,rejected_destinations=invalid,program_sha256=hashlib.sha256(program.read_bytes()).hexdigest(),rtl_numeric=False)
(a.root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
