"""Preserve native64; build an isolated two-fix candidate and synthesis input.

No hardware access, no topology/clock/constraint changes. The ROM edit and
16 Bayer sum-output reconnections are fully reversible and explicitly scoped.
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT=Path('E:/fpga')
BASE=ROOT/'tmp/frame-execution/video-service-native64-joint-control'
BOARD=ROOT/'tmp/frame-execution/connected-board-video-service-probe55-joint-control'
OUT=ROOT/'tmp/frame-execution/video-service-native67-color-exposure'
MAP=ROOT/'tmp/frame-execution/connected-board-video-service-probe58-color-exposure'
AUDIT=ROOT/'tmp/board-test/video-service64-color-audit4'
PREFIX='raw_frontend/consumer/debayer/legacy_debayer/inst_raw_to_rgb/'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def read_rom(text):
    values={}
    blocks=[]
    for m in re.finditer(r'^\s*EFX_RAM10\s+(\\camera_checked/config_rom/\S+)\s*\(',text,re.M):
        end=re.search(r'^\s*EFX_\w+\s',text[m.end():],re.M)
        hi=m.end()+end.start()
        block=text[m.start():hi]
        width=int(re.search(r'\.READ_WIDTH\s*=\s*(\d+)',block)[1])
        init=re.findall(r'defparam\s+\S+\s*\.INIT_([0-9A-F]+)\s*=\s*256\x27b([01]+)',block)
        assert len(init)==40
        blob=sum(int(bits,2)<<(int(n,16)*256) for n,bits in init)
        values[width]=[(blob>>(a*width))&((1<<width)-1) for a in range(164)]
        blocks.append((m.start(),hi,block,width,blob))
    assert set(values)=={5,20}
    return [values[5][a]<<20|values[20][a] for a in range(164)],blocks

def main():
    assert not OUT.exists() and not MAP.exists()
    assert sha(BASE/'native_source.v')=='bd03eb8c815470bde6e3039e8c5db17ed1dbf0cddbfa904cce0a6c36e7a63a87'
    test=json.loads((AUDIT/'result.json').read_text())
    assert test['checked']==792 and test['current_bad']==120 and test['demo_bad']==test['candidate_bad']==0
    original=(BASE/'native_source.v').read_bytes().decode('utf-8')
    before,blocks=read_rom(original)
    romfile=BOARD/'rtl/cam_i2c_ctrl/i2c/i2c_master_reg_rom.v'
    rom=romfile.read_text(encoding='utf-8-sig')
    entries={int(a,16):(int(reg,16)<<9)|(int(v,16)<<1)|int(last) for a,reg,v,last in re.findall(r"ROM\[8'h([0-9a-f]+)\]\s*=\s*\{16'h([0-9a-f]+),8'h([0-9a-f]+),1'b([01])\}",rom,re.I)}
    assert len(entries)==164 and all(before[a]==v for a,v in entries.items())
    exposure_changes={0x73:(0x3e00,0x01,0x00),0x74:(0x3e01,0xbb,0xba),0x75:(0x3e02,0xf0,0xd0)}
    lo,hi,block,width,blob=next(b for b in blocks if b[3]==20)
    updated=blob
    for a,(reg,old,new) in exposure_changes.items():
        assert before[a]==reg<<9|old<<1
        updated=(updated&~(((1<<20)-1)<<(a*20)))|(((reg<<9|new<<1)&((1<<20)-1))<<(a*20))
    rom_deltas=[]
    newblock=block
    for i in range(40):
        old=f'{(blob>>(i*256))&((1<<256)-1):0256b}'
        new=f'{(updated>>(i*256))&((1<<256)-1):0256b}'
        if old==new:continue
        pattern=r'(INIT_'+format(i,'X')+r"\s*=\s*256'b)"+old
        newblock,n=re.subn(pattern,lambda m:m[1]+new,newblock)
        assert n==2,(i,n) # mapped attribute and executable defparam
        rom_deltas.append(dict(parameter='INIT_'+format(i,'X'),old=old,new=new))
    assert len(rom_deltas)==2
    candidate=original[:lo]+newblock+original[hi:]
    connection_pattern=re.compile(r'(\.O\(\\'+re.escape(PREFIX)+r'w_)([rb])(_00_00_h0\s+\[([1-8])\]\))')
    changed=[]
    def swap(m):
        changed.append((m[2],int(m[4])))
        return m[1]+('b' if m[2]=='r' else 'r')+m[3]
    candidate,n=connection_pattern.subn(swap,candidate)
    assert n==16 and set(changed)=={(c,i) for c in 'rb' for i in range(1,9)}
    after,_=read_rom(candidate)
    for a in range(164):
        expected=(exposure_changes[a][0]<<9|exposure_changes[a][2]<<1) if a in exposure_changes else before[a]
        assert after[a]==expected,(a,after[a],expected)
    reverted=connection_pattern.sub(lambda m:m[1]+('b' if m[2]=='r' else 'r')+m[3],candidate)
    assert reverted[lo:lo+len(newblock)]==newblock
    reverted=reverted[:lo]+block+reverted[lo+len(newblock):]
    assert reverted==original,'Unexpected edit outside the two declared cuts'
    OUT.mkdir()
    for f in BASE.glob('*.sv'):shutil.copy2(f,OUT/f.name)
    (OUT/'native_source.v').write_bytes(candidate.encode('utf-8'))
    shutil.copy2(AUDIT/'candidate_raw_to_rgb.v',OUT/'candidate_raw_to_rgb.v')
    for a,(reg,old,new) in exposure_changes.items():
        pattern=rf"(ROM\[8'h{a:02x}\]\s*=\s*\{{16'h{reg:04x},8'h){old:02x}"
        rom,n=re.subn(pattern,lambda m:m[1]+f'{new:02x}',rom,flags=re.I)
        assert n==1
    (OUT/'candidate_i2c_master_reg_rom.v').write_text(rom,encoding='utf-8')
    proof=dict(status='COLOR_EXPOSURE_EXACT_SCOPED_NATIVE_EDIT_PASS',parent=str(BASE),
        parent_sha256=sha(BASE/'native_source.v'),candidate_sha256=sha(OUT/'native_source.v'),
        native_rom_matches_all_164_source_words=True,bayer_output_reconnections=16,
        rom_changed_addresses=[0x73,0x74,0x75],rom_init_changes=rom_deltas,
        original_exposure_half_lines=0x1bbf,candidate_exposure_half_lines=0x0bad,
        vts_assumption=1500,vts_evidence='SC431HAI manual default 0x05dc; no VTS write in this ROM; not a live register readback',
        exposure_max_from_datasheet=2*1500-11,all_other_source_bytes_reversibly_identical=True,
        cnn_weights_program_topology_clocks_resets_pins_constraints_unchanged=True,
        resource_cells_added=0,hardware_access=False,physical_pass=False,board_release=False,
        evidence={str(p):sha(p) for p in [BASE/'native_source.v',romfile,AUDIT/'inputs.json',AUDIT/'result.json',AUDIT/'simulation.log']})
    save(OUT/'color-exposure-edit.json',proof)
    meta=json.loads((BOARD/'inputs.json').read_text())
    MAP.mkdir(); rows=[]
    for row in meta['inputs']:
        old=Path(row['snapshot']);assert sha(old)==row['sha256']
        rel=old.relative_to(BOARD)
        origin=OUT/'native_source.v' if rel.as_posix()=='native_source.v' else old
        if rel.as_posix()=='rtl/debayer/raw_to_rgb.v':origin=OUT/'candidate_raw_to_rgb.v'
        if rel.as_posix()=='rtl/cam_i2c_ctrl/i2c/i2c_master_reg_rom.v':origin=OUT/'candidate_i2c_master_reg_rom.v'
        dest=MAP/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(origin,dest)
        rows.append(dict(source=str(origin),snapshot=str(dest),sha256=sha(dest)))
    shutil.copy2(BOARD/'map.cmd',MAP/'map.cmd')
    save(MAP/'inputs.json',dict(inputs=rows,parent=str(BOARD),parent_sha256=sha(BOARD/'inputs.json'),
        native=str(OUT),functional_edit_proof=str(OUT/'color-exposure-edit.json'),
        functional_edit_proof_sha256=sha(OUT/'color-exposure-edit.json'),builder_sha256=sha(__file__),
        original_sdc_unchanged=True,physical_cdc_and_pin_audit_pending=True,hardware_access=False,board_release=False))
    print(json.dumps({k:proof[k] for k in ['status','candidate_sha256','bayer_output_reconnections','original_exposure_half_lines','candidate_exposure_half_lines','resource_cells_added']}))

if __name__=='__main__':main()
