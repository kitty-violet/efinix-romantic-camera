"""Four-state original encoder A/B, with only public audio packet leaves changed.

400k pixel clocks: two reset/restart epochs, 32 real-width video rows, and
enough blank clocks to exercise the original 148500-clock ACR timer. Not a
full video frame or serial PHY test. Decode the actual external TMDS symbols.
"""
import argparse, collections, json, re, shutil, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from run_units import ROOT, HERE, EF, MS, sha, run
from validate_board_dvi import TB, decode

TERC = [0x29c,0x263,0x2e4,0x2e2,0x171,0x11e,0x18e,0x13c,
        0x2cc,0x139,0x19c,0x2c6,0x28e,0x271,0x163,0x2c3]
REVERSE = {s:i for i,s in enumerate(TERC)}

def ecc(word, bits):
    parity=0
    for bit in range(bits):
        feedback=(parity ^ (word>>bit)) & 1
        parity=(parity>>1) ^ (0x83 if feedback else 0)
    return parity

def inspect(path, require_avi=True):
    rows=[line.split() for line in path.read_text().splitlines()]
    assert len(rows)==400000
    valid=[i for i,r in enumerate(rows) if r[1]=='1']
    assert len(valid)==61440
    symbols=[]
    for i,r in enumerate(rows):
        if re.search('[xz]', ''.join(r[3:6]), re.I):
            assert i%200000<128, ('unknown output',i,r)
            symbols.append(None)
        else: symbols.append(tuple((~int(s,16))&1023 for s in r[3:6]))
    active=set()
    for i in valid:
        q=rows[i+58]
        got=decode(q[3]) | (decode(q[4])<<8) | (decode(q[5])<<16)
        assert got==int(rows[i][2],16), ('RGB',i,got,rows[i][2])
        active.add(i+58)
    packets=[];island_positions=set();i=0
    while i<len(rows):
        s=symbols[i]
        if i in active or not s or not all(x in REVERSE for x in s):
            i+=1;continue
        start=i
        while i<len(rows) and i not in active and symbols[i] and all(x in REVERSE for x in symbols[i]):i+=1
        if i-start<32:continue  # Isolated video/guard symbols cannot form a packet.
        # The stimulus deliberately resets mid-packet and ends mid-blanking.
        # Only those explicit boundaries may truncate an island. Complete
        # packets before the interruption still receive full BCH checking.
        assert (i-start)%32==0 or i%200000==0, ('island length',start,i)
        island_positions.update(range(start,i))
        for begin in range(start,i-31,32):
            head=0;body=[0,0,0,0]
            for bit in range(32):
                a,b,c=(REVERSE[v] for v in symbols[begin+bit])
                head |= ((a>>2)&1)<<bit
                for n in range(4):
                    body[n] |= ((b>>n)&1)<<(bit*2)
                    body[n] |= ((c>>n)&1)<<(bit*2+1)
            assert ecc(head,32)==0, ('header BCH',begin,hex(head))
            assert all(ecc(v,64)==0 for v in body), ('body BCH',begin)
            header=head & 0xffffff
            payload=b''.join((v&((1<<56)-1)).to_bytes(7,'little') for v in body)
            if header&255==0x82:
                assert header==0x0d0282
                assert payload[4]==34 and payload[1]>>5==0
                assert (sum(header.to_bytes(3,'little'))+sum(payload[:14]))%256==0
            packets.append((begin,header,payload.hex()))
    if require_avi:
        assert sum(h&255==0x82 for _,h,_ in packets)>=2, 'No AVI after reset/restart'
    return rows,symbols,active,island_positions,packets

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--replay',type=Path,help='Re-audit immutable completed symbol captures; no simulator rerun')
    p.add_argument('--reference-capture',type=Path,help='Reuse a byte-identical reference run only')
    a=p.parse_args()
    root=a.root.resolve();base=a.base.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    meta=json.loads((base/'inputs.json').read_text())
    for r in meta['inputs']:assert sha(r['snapshot'])==r['sha256']
    root.mkdir(parents=True)
    records=[];times={}
    try:
        tb=TB.replace('n<40000','n<400000').replace(
            'reset=(n<64);x=(n-64)%2200;y=(n-64)/2200;',
            'reset=((n%200000)<64);x=((n%200000)-64)%2200;y=((n%200000)-64)/2200;').replace(
            'de=(n>=64&&','de=((n%200000)>=64&&')
        capture=a.replay.resolve() if a.replay else root
        def simulate(variant):
            work=root/variant;work.mkdir()
            sources=sorted(f for f in (base/'rtl/dvi_tx').iterdir() if f.is_file() and f.suffix in ('.sv','.v','.vh'))
            # Frozen trees contain unused same-name legacy copies. The map
            # manifest, not a directory glob, chooses the actual board source.
            board_sources={Path(r['snapshot']).name:Path(r['snapshot']) for r in meta['inputs']}
            sources=[board_sources.get(f.name,f) for f in sources]
            sources.append(EF/'sim_models/verilog/efx_srl8.v')
            local=[]
            for src in sources:
                override=HERE/'variants/no_audio'/src.name
                use=override if variant=='candidate' and override.exists() else src
                dest=work/src.name;shutil.copy2(use,dest)
                records.append(dict(variant=variant,source=str(use),snapshot=str(dest),sha256=sha(dest)))
                if dest.suffix!='.vh':local.append(dest)
            tbfile=work/'tb_board_dvi.sv';tbfile.write_text(tb)
            records.append(dict(variant=variant,source=str(tbfile),snapshot=str(tbfile),sha256=sha(tbfile)))
            if variant=='reference' and a.reference_capture:
                prior=json.loads((a.reference_capture/'inputs.json').read_text())
                expected={Path(r['snapshot']).name:r['sha256'] for r in prior['inputs'] if r['variant']=='reference'}
                actual={Path(r['snapshot']).name:r['sha256'] for r in records if r['variant']=='reference'}
                assert expected==actual, 'Reference reuse source/TB mismatch'
                oldlog=(a.reference_capture/'reference/simulation.log').read_text()
                assert 'PASS DVI_STIMULUS_COMPLETE' in oldlog and 'Errors: 0, Warnings: 0' in oldlog
                for name in ('symbols.txt','simulation.log'):
                    shutil.copy2(a.reference_capture/'reference'/name,work/name)
                times[variant]='reused byte-identical reference'
                return
            run([MS/'vlib.exe','work'],work,'vlib.log')
            run([MS/'vlog.exe','-sv','-timescale','1ns/1ps','+incdir+.',*local,tbfile],work,'compile.log')
            body,seconds=run([MS/'vsim.exe','-c','work.tb_board_dvi','-do',
                 'onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],work,'simulation.log',timeout=240)
            assert 'PASS DVI_STIMULUS_COMPLETE' in body
            times[variant]=seconds
        if not a.replay:
            # Independent original/candidate short programs; no agent delegation.
            with ThreadPoolExecutor(max_workers=2) as workers:
                jobs=[workers.submit(simulate,v) for v in ('reference','candidate')]
                for job in jobs:job.result()
        if a.replay:
            prior=json.loads((capture/'inputs.json').read_text())
            assert prior['base_inputs_sha256']==sha(base/'inputs.json')
            records=prior['inputs']
            for r in records:assert sha(r['snapshot'])==r['sha256']
            for variant in ('reference','candidate'):
                log=(capture/variant/'simulation.log').read_text()
                assert 'PASS DVI_STIMULUS_COMPLETE' in log and 'Errors: 0, Warnings: 0' in log
        inputs=dict(inputs=records,base=str(base),base_inputs_sha256=sha(base/'inputs.json'),runner_sha256=sha(__file__),
          captures=[dict(path=str(capture/v/f),sha256=sha(capture/v/f))
                    for v in ('reference','candidate') for f in ('symbols.txt','simulation.log')])
        (root/'inputs.json').write_text(json.dumps(inputs,indent=2))
        ref,rs,active,ri,rp=inspect(capture/'reference/symbols.txt')
        new,ns,newactive,ni,np=inspect(capture/'candidate/symbols.txt')
        assert active==newactive and ri==ni, 'Changed video/island timing'
        for i in range(400000):
            assert ref[i][6]==new[i][6], ('TMDS clock',i)
            if i not in ri: assert rs[i]==ns[i], ('Non-island symbol changed',i)
            else:
                assert REVERSE[rs[i][0]]&0xb==REVERSE[ns[i][0]]&0xb, ('Island framing/sync changed',i)
        avi=lambda packets:[(h,b) for _,h,b in packets if h&255==0x82]
        assert avi(rp)==avi(np), 'Changed AVI sequence'
        assert set(h&255 for _,h,_ in np)<= {0,0x82}, 'Audio packet survived'
        assert any(h&255 in (1,2,0x84) for _,h,_ in rp), 'Reference audio path not exercised'
        for r in records:assert sha(r['snapshot'])==r['sha256']
        result=dict(status='NO_AUDIO_HDMI_FOUR_STATE_AB_PASS',pixel_clocks=400000,
          rgb_pixels=61440,latency_pixels=58,reset_epochs=2,
          reference_packet_types=dict(collections.Counter(h&255 for _,h,_ in rp)),
          candidate_packet_types=dict(collections.Counter(h&255 for _,h,_ in np)),
          avi_identical=True,non_island_symbols_identical=True,all_packet_bch_valid=True,
          seconds=times,capture_root=str(capture),inputs_sha256=sha(root/'inputs.json'),full_frame=False,
          full_board=False,serial_phy=False,board_download_allowed=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as e:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise

if __name__=='__main__':main()
