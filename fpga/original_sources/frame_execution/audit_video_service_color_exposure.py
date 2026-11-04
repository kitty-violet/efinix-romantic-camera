"""Independent exact delta audit for native67; no physical or board release."""
import argparse, collections, json, pickle, re, time
from pathlib import Path
from run_units import ROOT, HERE, sha
from native_local_drivers import graph
from probe_full_native_split_map import params
from complete_native_logic_split import protected_blocks
from continuous_control_native import connections, net

P = 'raw_frontend/consumer/debayer/legacy_debayer/inst_raw_to_rgb/'

def main():
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);a=p.parse_args()
    native=a.native.resolve();edit=json.loads((native/'color-exposure-edit.json').read_text())
    parent=Path(edit['parent']);out=native/'color-exposure-independent-audit.json'
    assert not out.exists();start=time.monotonic()
    assert sha(native/'native_source.v')==edit['candidate_sha256']
    assert sha(parent/'native_source.v')==edit['parent_sha256']
    for f,h in edit['evidence'].items():assert sha(f)==h,f
    before=(parent/'native_source.v').read_text();after=(native/'native_source.v').read_text()
    key,old,op=pickle.loads((parent/'zero-source-parse.pkl').read_bytes());assert key==sha(parent/'native_source.v')
    new=graph(after);np=params(after)
    assert old.keys()==new.keys() and op.keys()==np.keys()
    changed=[]
    for name in old:
        if old[name]==new[name]:continue
        kind,pins=old[name];newkind,newpins=new[name]
        assert kind==newkind=='EFX_ADD' and name.startswith(P)
        assert pins.keys()==newpins.keys()
        assert all(pins[q]==newpins[q] for q in pins if q!='O')
        assert len(pins['O'])==len(newpins['O'])==1
        t=pins['O'][0];match=re.fullmatch(re.escape(P)+r'w_([rb])_00_00_h0',t[1])
        assert match and t[2] in range(1,9)
        other='b' if match[1]=='r' else 'r'
        assert newpins['O']==[net(P+'w_'+other+'_00_00_h0',t[2])]
        changed.append(name)
    assert len(changed)==16
    param_changed=[]
    for name in op:
        if op[name]==np[name]:continue
        assert name=='camera_checked/config_rom/ROM__D$2'
        assert op[name].keys()==np[name].keys()
        keys=[k for k in op[name] if op[name][k]!=np[name][k]]
        assert set(keys)=={'INIT_8','INIT_9'}
        param_changed.extend((name,k) for k in keys)
    assert len(param_changed)==2
    # Decode executable primitive parameters independently of the edit script.
    decoded={}
    for version,ps in [('old',op),('new',np)]:
        words=[]
        for addr in range(164):
            v=0
            for suffix,width,shift in [('1',5,20),('2',20,0)]:
                pr=ps['camera_checked/config_rom/ROM__D$'+suffix]
                blob=sum(pr['INIT_'+format(i,'X')][1]<<(256*i) for i in range(40))
                v|=((blob>>(width*addr))&((1<<width)-1))<<shift
            words.append(v)
        decoded[version]=words
    expected={0x73:(0x3e00,0x00),0x74:(0x3e01,0xba),0x75:(0x3e02,0xd0)}
    for addr in range(164):
        want=(expected[addr][0]<<9|expected[addr][1]<<1) if addr in expected else decoded['old'][addr]
        assert decoded['new'][addr]==want,(addr,want,decoded['new'][addr])
    for pattern in (r'^\s*(?:input|output|inout)\b.*$',r'^\s*assign\b.*$'):
        assert re.findall(pattern,before,re.M)==re.findall(pattern,after,re.M)
    assert protected_blocks(before)==protected_blocks(after)
    for f in parent.glob('*.sv'):assert sha(f)==sha(native/f.name)
    # Identify exact changed-adder leaves; subsequent tests must drive these
    # actual mapped boundaries, not a disconnected lookalike implementation.
    drivers,_=connections(new);cones={}
    for color in 'rb':
        selected=set();leaves=set();stack=[net(P+'w_'+color+'_00_00_h0',i) for i in range(1,9)]
        while stack:
            t=stack.pop()
            if t[0]=='literal':continue
            name=drivers.get(t)
            if name is None or new[name][0] not in ('EFX_ADD','EFX_LUT4'):
                leaves.add(t);continue
            if name in selected:continue
            selected.add(name)
            stack.extend(t for pin,ts in new[name][1].items() if pin not in ('O','CO') for t in ts)
        cones[color]=dict(cells=sorted(selected),leaves=sorted(leaves,key=str))
    cache=native/'color-source-parse.pkl';cache.write_bytes(pickle.dumps((sha(native/'native_source.v'),new,np),protocol=4))
    result=dict(status='COLOR_EXPOSURE_INDEPENDENT_EXACT_DELTA_PASS',source_sha256=sha(native/'native_source.v'),
        parent_sha256=sha(parent/'native_source.v'),primitive_count=len(new),changed_connections=changed,
        parameter_changes=param_changed,rom_words_verified=164,rom_changed_addresses=sorted(expected),
        all_other_primitives_and_parameters_identical=True,all_assigns_ports_and_protected_blocks_identical=True,
        support_rtl_identical=True,cones=cones,seconds=time.monotonic()-start,
        evidence={str(f):sha(f) for f in [Path(__file__),native/'color-exposure-edit.json',parent/'native_source.v',native/'native_source.v']},
        hardware_access=False,physical_pass=False,board_release=False)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','primitive_count','rom_words_verified','seconds']}))
    print(json.dumps(cones))

if __name__=='__main__':main()
