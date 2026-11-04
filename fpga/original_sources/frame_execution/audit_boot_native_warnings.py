"""Check every warning against the frozen Efinity primitive implementation."""
import argparse
import bisect
import collections
import json
import re
from pathlib import Path
from run_units import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();root=a.root.resolve()
    log=(root/'compile.log').read_text();net=(root/'board_ip.map.v').read_text()
    blocks=re.split(r'(?=^%Warning-)',log,flags=re.M)
    line_offsets=[0]+[m.end() for m in re.finditer('\n',net)]
    instances=list(re.finditer(r'\b(EFX_\w+)\s+(\S+)\s+\(',net))
    starts=[m.start() for m in instances]
    reviewed=collections.Counter()
    for block in blocks:
        if not block.startswith('%Warning-'):continue
        kind=re.match(r'%Warning-([^:]+)',block).group(1)
        first=block.splitlines()[0]
        if kind=='PINMISSING':
            m=re.search(r'board_ip\.map\.v:(\d+):(\d+): Instance has missing pin: \'(\w+)\'',first);assert m,first
            line,column,pin=int(m.group(1)),int(m.group(2)),m.group(3)
            index=bisect.bisect_right(starts,line_offsets[line-1]+column-1)-1
            inst=instances[index];model=(root/(inst.group(1).lower()+'.v')).read_text()
            if pin in ('I2','I3','CI'):
                # Official models weakly pull optional inputs to zero. A
                # disconnected pin therefore has a defined zero value, not X.
                assert re.search(r'assign\s*\(weak0, weak1\)\s*'+pin+r'_net\s*=\s*1\'b0',model),(inst.group(1),pin)
                reviewed[inst.group(1)+':optional_zero_'+pin]+=1
            else:
                assert pin in ('O','CO','FCO','Q7')
                assert re.search(r'output\s+(?:wire\s+)?'+pin+r'\b',model),(inst.group(1),pin)
                reviewed[inst.group(1)+':unused_output_'+pin]+=1
        elif kind=='IMPLICIT':
            m=re.search(r"implicitly: '(Open_\d+)'",first);assert m,first
            name=m.group(1)
            assert len(re.findall(r'\b'+name+r'\b',net))==1,name
            pos=net.index(name);stmt=net[net.rfind('.RDATA(',0,pos):net.index(')',pos)+1]
            assert name in stmt and len(stmt)<1000,name
            reviewed['RAM:unconsumed_RDATA_bit']+=1
        elif kind=='WIDTHEXPAND':
            if 'efx_ram10.v:' in first:
                assert "VARREF 'WRITE_MODE'" in first or "VARREF 'RESET_OUTREG'" in first
                reviewed['RAM:string_comparison_zero_extension']+=1
            else:
                assert 'sapphire_axi32_to_axi128.sv:89:' in first and 'expects 32 bits' in first and 'generates 2 bits' in first
                reviewed['adapter:unsigned_address_2_to_32']+=1
        elif kind=='WIDTHTRUNC':
            assert 'tb_real_boot.sv:' in first and "VARREF 'smoke'" in first
            reviewed['testbench:boolean_smoke_switch']+=1
        else:raise AssertionError(first)
    assert sum(reviewed.values())==len(re.findall(r'^%Warning-',log,re.M))
    result=dict(status='BOOT_NATIVE_WARNINGS_REVIEWED',reviewed=dict(reviewed),total=sum(reviewed.values()),
                log_sha256=sha(root/'compile.log'),net_sha256=sha(root/'board_ip.map.v'),
                full_four_state_equivalence=False,full_board=False)
    (root/'warning_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
