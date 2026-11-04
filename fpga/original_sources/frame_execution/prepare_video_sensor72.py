"""Scoped SC431HAI 2.503x / 1120-half-line candidate, including readback.

The released native source is immutable. Four expected-data FF D pins get
an XOR only at VERIFY/index10; all other public and protected text is exact.
This prepares official synthesis inputs, not a released FPGA image.
"""
import argparse, hashlib, json, re, shutil
from pathlib import Path
from prepare_video_sensor_gain_candidate import NATIVE, NATIVE_SHA, ROM, MANUAL
from prepare_video_service_color_exposure import ROOT, read_rom, sha


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--board',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();board=a.board.resolve()
    assert all(x.is_relative_to(ROOT/'tmp/frame-execution') and not x.exists() for x in (root,board))
    assert sha(NATIVE)==NATIVE_SHA
    original=NATIVE.read_text(encoding='utf8');words,blocks=read_rom(original)
    changes={116:(0x3e01,0xba,0x46),117:(0x3e02,0xd0,0),162:(0x3e09,0x2a,0x34)}
    lo,hi,block,width,blob=next(b for b in blocks if b[3]==20)
    updated=blob;rom=ROM.read_text(encoding='utf8')
    for index,(reg,old,new) in changes.items():
        assert words[index]==reg<<9|old<<1
        updated=(updated&~(0xfffff<<(index*20)))|(((reg<<9|new<<1)&0xfffff)<<(index*20))
        pat=rf"(ROM\[8'h{index:02x}\]\s*=\s*\{{16'h{reg:04x},8'h){old:02x}"
        rom,n=re.subn(pat,lambda m:m[1]+f'{new:02x}',rom,flags=re.I);assert n==1
    changed_block=block;patches=[]
    for i in range(40):
        old=f'{(blob>>(i*256))&((1<<256)-1):0256b}'
        new=f'{(updated>>(i*256))&((1<<256)-1):0256b}'
        if new==old:continue
        key='INIT_'+format(i,'X')
        changed_block,n=re.subn('('+key+r"\s*=\s*256'b)"+old,lambda m:m[1]+new,changed_block)
        assert n==2
        patches.append(dict(parameter=key,old=old,new=new,occurrences=n))
    candidate=original[:lo]+changed_block+original[hi:]
    rewrites=[]
    for bit in range(1,5):
        cell=f'camera_config_diagnostic[{16+bit}]~FF'
        pat=rf'(EFX_FF\s+\\{re.escape(cell)}\s+\(\.D\()([^)]*)(\))'
        match=re.search(pat,candidate);assert match
        old=match[2];assert old.strip()==rf'\camera_checked/n229 [{bit}]'
        new=f'sensor72_expected_{bit}'
        candidate,n=re.subn(pat,lambda m:m[1]+new+m[3],candidate);assert n==1
        rewrites.append(dict(cell=cell,port='D',old=old,new=new))
    # Existing phase/index-high qualifier is audited independently in the test.
    assert re.search(r'EFX_LUT4 LUT__241152\s*\(\.I0\(n80869_2\), \.I1\(n80870_2\), \.O\(n80871_2\)\)',original)
    names=['sensor72_index10']+[f'sensor72_expected_{b}' for b in range(1,5)]
    decl='\nwire '+','.join(names)+';\n'
    mask=sum((int(bool(i&1)^bool((i&2) and (i&4))))<<i for i in range(16))
    extra='\nEFX_LUT4 sensor72_index_decode(.I0(camera_config_diagnostic[48]),.I1(camera_config_diagnostic[49]),.I2(camera_config_diagnostic[50]),.I3(camera_config_diagnostic[51]),.O(sensor72_index10));\n'
    extra+="defparam sensor72_index_decode.LUTMASK=16'h0400;\n"
    for bit in range(1,5):
        extra+=rf'EFX_LUT4 sensor72_expected_xor{bit}(.I0(\camera_checked/n229 [{bit}]),.I1(n80871_2),.I2(sensor72_index10),.I3(1\'b0),.O(sensor72_expected_{bit}));'.replace("\\'","'")+'\n'
        extra+=f"defparam sensor72_expected_xor{bit}.LUTMASK=16'h{mask:04x};\n"
    # Declare before instances; no protected blocks or shared signal definitions change.
    insertion=re.search(r'\bEFX_(?:LUT4|FF|RAM10|ADD)\s',candidate).start()
    candidate=candidate[:insertion]+decl+candidate[insertion:]
    assert len(re.findall(r'^endmodule\b',candidate,re.M))==1
    candidate=re.sub(r'^endmodule\b',lambda m:extra+'endmodule',candidate,flags=re.M)
    restored=candidate.replace(decl,'').replace(extra,'')
    for row in rewrites:
        pat=rf'(EFX_FF\s+\\{re.escape(row["cell"])}\s+\(\.D\(){row["new"]}(\))'
        restored,n=re.subn(pat,lambda m:m[1]+row['old']+m[2],restored);assert n==1
    restored=restored[:lo]+block+restored[lo+len(changed_block):]
    assert restored==original,'Unlisted native edit'
    actual,_=read_rom(candidate);expected=list(words)
    for i,(reg,old,new) in changes.items():expected[i]=reg<<9|new<<1
    assert actual==expected
    checks=0
    for phase in range(16):
        for index in range(256):
            toggle=(phase==4 and index==10)
            qualifier=(phase==4 and index>>4==0)
            decoded=(0x0400>>(index&15))&1
            for old in (0,1):
                pins=old|(int(qualifier)<<1)|(decoded<<2)
                assert (mask>>pins)&1==old^toggle
                checks+=1
    root.mkdir();(root/'native_source.v').write_text(candidate,encoding='utf8')
    (root/'candidate_i2c_master_reg_rom.v').write_text(rom,encoding='utf8')
    reference=ROOT/'hw/efinity/prototypes/frame_execution/camera_checked_config.sv'
    reftext=reference.read_text(encoding='utf8');assert reftext.count("3e092aff")==1
    (root/'camera_checked_config_reference.sv').write_text(reftext.replace('3e092aff','3e0934ff'),encoding='utf8')
    proof=dict(status='SCOPED_SENSOR72_SOURCE_PASS_NOT_RELEASED',source_sha256=sha(NATIVE),
        candidate_native_sha256=sha(root/'native_source.v'),candidate_rom_sha256=sha(root/'candidate_i2c_master_reg_rom.v'),
        native_init_patches=patches,changes={str(i):dict(register=r,old=o,new=n) for i,(r,o,n) in changes.items()},
        expected_ff_d_edits=rewrites,new_luts=5,all_unlisted_source_text_byte_equal=True,
        compared_rom_words=164,qualifier_truth_cases=checks,exposure_half_lines=1120,gain=2.503,
        gain_registers=[128,52],verify_index=10,verify_expected=52,gamma_runtime_target=2.3,
        gamma_not_applied=True,hardware_access=False,board_release=False,
        inputs={str(f):sha(f) for f in (Path(__file__),NATIVE,ROM,MANUAL,reference)})
    save(root/'result.json',proof)
    parent=ROOT/'tmp/frame-execution/connected-board-video-service-probe60-input-isp-explicit'
    meta=json.loads((parent/'inputs.json').read_text());board.mkdir();rows=[]
    for row in meta['inputs']:
        old=Path(row['snapshot']);assert sha(old)==row['sha256']
        rel=old.relative_to(parent);origin=root/'native_source.v' if rel.as_posix()=='native_source.v' else old
        dst=board/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(origin,dst)
        rows.append(dict(source=str(origin),snapshot=str(dst),sha256=sha(dst)))
    shutil.copy2(parent/'map.cmd',board/'map.cmd')
    save(board/'inputs.json',dict(inputs=rows,parent=str(parent),native=str(root),original_sdc_unchanged=True,
        functional_edit_proof=str(root/'result.json'),functional_edit_proof_sha256=sha(root/'result.json'),hardware_access=False,board_release=False))
    print(proof['status'],flush=True)


if __name__=='__main__':main()
