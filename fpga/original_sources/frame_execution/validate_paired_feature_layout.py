"""Independent external-memory footprint proof for paired C32 descriptor ABI 3."""
import argparse, collections, json
from pathlib import Path
from run_units import ROOT, sha

p=argparse.ArgumentParser()
p.add_argument('--program',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert not a.output.exists() and a.output.resolve().is_relative_to(ROOT/'tmp')
meta=json.loads((a.program/'result.json').read_text())
assert meta['paired_feature_ddr'] and meta['status']=='ROW_PAIR_FULL_GRAPH_ADDRESS_PASS'
assert sha(a.program/'program.json')==meta['program_sha256']
commands=json.loads((a.program/'program.json').read_text())
old=ROOT/'tmp/frame-execution/joint-reorder-vectors1/program.hex'
words=[int(x,16) for x in old.read_text().splitlines()]
def physical_byte(logical):
    return (logical//64)*64+((logical%32)//8)*16+((logical%64)//32)*8+logical%8
# Prove the lane permutation is bijective, preserves each 64-byte block and
# is its own producer/consumer convention, independently of the RTL counters.
assert sorted(physical_byte(x) for x in range(64))==list(range(64))
counts=collections.Counter();logical_bytes=padding=0;skip_phase_checks=0
for c in commands:
    if c['kind']=='compute':
        # This production skip unit specializes offsets 0/2 and has no phase
        # input. Never allow a future compiler revision to silently violate it.
        for d in c['skip']:
            assert d['phase']==0
            skip_phase_checks+=1
        continue
    d=c['row']
    if d['groups']!=4:continue
    address=(words[c['command']]>>(7*32))&0xffffffff
    assert address%32==0 and d['phase']==(address//32)%2==d['origin']%2
    row_base=address-d['origin']*32
    assert row_base%64==0 and 0<=d['origin'] and d['origin']+d['width']<=320
    lo=address//64*64;hi=(address+d['width']*32+63)//64*64
    assert row_base<=lo<hi<=row_base+320*32
    window=next((b for b in (0,0x00800000,0x02000000,0x03000000) if b<=lo<hi<=b+0x400000),None)
    assert window is not None
    assert (hi-lo)//16==((d['width']+d['phase']+1)//2)*4
    for x in range(d['width']):
        for g in range(4):
            # Match a logical C8 group's first/last byte to its actual
            # SRAM word and half; every byte in between is consecutive.
            word=d['base']+(x+d['phase'])//2*4+g
            half=(x+d['phase'])%2
            for b in (0,7):
                expected=lo+(word-d['base'])*16+half*8+b
                assert physical_byte(address+x*32+g*8+b)==expected
            assert 0<=word<1024
    if c['kind']=='store':assert lo==address and hi==address+d['width']*32
    else:padding+=hi-lo-d['width']*32
    logical_bytes+=d['width']*32;counts[c['kind']]+=1
assert counts=={'load':1620,'store':1620} and padding==meta['feature_load_padding_bytes']
result=dict(status='PAIRED_FEATURE_DDR_FOOTPRINT_PASS',commands=dict(counts),logical_bytes=logical_bytes,
            additional_read_bytes=padding,skip_phase_checks=skip_phase_checks,
            per_block_bijection=True,ddr_windows_and_rows_bounded=True,
            byte_to_sram_word_half_proved=True,program_result_sha256=sha(a.program/'result.json'),
            reference_program_sha256=sha(old),script_sha256=sha(__file__),rtl_executed=False)
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
