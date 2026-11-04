"""Lower gain only relative to live76; keep exposure and geometry unchanged."""
import hashlib,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from prepare_video_service_color_exposure import read_rom

BASE=ROOT/'tmp/frame-execution/video-sensor73-bits9'
OUT=ROOT/'tmp/frame-execution/video-gain82-candidate1'
PREPARED=ROOT/'tmp/frame-execution/video-gain82-pin-prepared1'
OLD=ROOT/'tmp/frame-execution/video-sensor73-pin-prepared2'
MANUAL=ROOT/'output/toolchain/project_evidence_audit_2026-09-28/pdf_text/摄像头_摄像头SC431HAI_数据手册_V1.1_dfe1737d7ee2.txt'
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
    assert not OUT.exists() and not PREPARED.exists()
    mapped=BASE/'outflow/ti60f225_oob.map.v'
    assert sha(mapped)=='07918fc6c80ee639828ef33785ea6a138fab4b47070d067d0b48ef400b7d74de'
    text=mapped.read_text(encoding='utf8');words,blocks=read_rom(text)
    lo,hi,block,width,blob=next(b for b in blocks if b[3]==20)
    assert words[116]==0x3e01<<9|0x46<<1 and words[162]==0x3e09<<9|0x34<<1
    assert (blob>>(266*20))&0xfffff==0x34<<1
    changes={116:dict(register=0x3e01,old=0x46,new=0x54),
             162:dict(register=0x3e09,old=0x34,new=0x2a),
             266:dict(register=0,old=0x34,new=0x2a,kind='readback_expected_byte')}
    updated=blob;newwords=list(words)
    for i,c in changes.items():
        value=c['register']<<9|c['new']<<1
        updated=(updated&~(0xfffff<<(20*i)))|((value&0xfffff)<<(20*i))
        if i<164:newwords[i]=value
    newblock=block;patches=[]
    for i in range(40):
        a=(blob>>(i*256))&((1<<256)-1);b=(updated>>(i*256))&((1<<256)-1)
        if a==b:continue
        key='INIT_'+format(i,'X')
        newblock,n=re.subn('('+key+r"\s*=\s*256'b)"+f'{a:0256b}',lambda m:m[1]+f'{b:0256b}',newblock);assert n==2
        patches.append(dict(parameter=key,old=f'{a:0256b}',new=f'{b:0256b}',occurrences=n))
    effective=text[:lo]+newblock+text[hi:]
    assert read_rom(effective)[0]==newwords and effective[:lo]+block+effective[lo+len(newblock):]==text
    changed=[i for i in range(512) if (blob>>(20*i))&0xfffff!=(updated>>(20*i))&0xfffff]
    assert changed==[116,162,266]
    # Relative to76, only the gain command and its matching readback expectation change.
    v76=json.loads((ROOT/'tmp/frame-execution/video-exposure76-candidate1/result.json').read_text())
    current=list(v76['configuration_words']);assert current[162]>>1&255==0x34
    assert [i for i,(a,b) in enumerate(zip(current,newwords)) if a!=b]==[162]
    oldmeta=json.loads((OLD/'inputs.json').read_text());assert oldmeta['map_sha256']==sha(mapped)
    for f,h in oldmeta['files'].items():assert sha(f)==h
    body=(OLD/'mapped_camera.v').read_text(encoding='utf8')
    for patch in patches:
        pat=(r'(defparam \\camera_checked/config_rom/ROM__D\$2 \.'+patch['parameter']+r"=256'h)"+f"{int(patch['old'],2):x}"+r';')
        body,n=re.subn(pat,lambda m:m[1]+f"{int(patch['new'],2):x};",body);assert n==1
    tb=(OLD/'tb.sv').read_text()
    old="if(mode==10 && address==16'h3e09) send_byte=8'h2a;"
    new="if(mode==10 && address==16'h3e09) send_byte=8'h34;"
    assert tb.count(old)==1;tb=tb.replace(old,new)
    old="diagnostic[31:24]!=8'h2a || diagnostic[23:16]!=8'h34"
    assert tb.count(old)==1;tb=tb.replace(old,"diagnostic[31:24]!=8'h34 || diagnostic[23:16]!=8'h2a")
    OUT.mkdir();PREPARED.mkdir()
    (OUT/'configuration.hex').write_text(''.join(f'{w:07x}\n' for w in newwords))
    report=dict(status='GAIN82_ROM_EXACT_DELTA_PASS_NOT_RELEASED',source_sha256=sha(mapped),
        candidate_native_sha256=hashlib.sha256(effective.encode()).hexdigest(),candidate_native_written=False,
        changed_addresses=changed,changes=changes,native_init_patches=patches,configuration_words=newwords,
        compared_register_words=164,untouched_native_text_byte_equal=True,exposure_before_half_lines=1120,
        exposure_after_half_lines=1344,gain_before=2.503,gain_after=2.021,verification_bank_unchanged=False,
        verification_changed_address=266,live76_exposure_unchanged=True,live76_gain_ratio=2.021/2.503,
        clocks_pins_topology_A_unchanged=True,hardware_access=False,board_release=False,
        evidence={str(f):sha(f) for f in (Path(__file__),mapped,BASE/'native_source.v',MANUAL,OLD/'inputs.json',ROOT/'tmp/frame-execution/video-exposure76-candidate1/result.json')})
    save(OUT/'result.json',report)
    (PREPARED/'mapped_camera.v').write_text(body,encoding='utf8');(PREPARED/'tb.sv').write_text(tb,encoding='utf8')
    (PREPARED/'expected.hex').write_text(''.join(f'60{w>>9:04x}{(w>>1)&255:02x}\n' for w in newwords))
    for kind in oldmeta['kinds']:
        name=kind.lower()+'.v';assert sha(OLD/name)==sha(EF/'sim_models/verilog'/name);shutil.copy2(OLD/name,PREPARED/name)
    meta=dict(oldmeta);meta.update(candidate=str(OUT),candidate_sha256=sha(OUT/'result.json'),original_prepared=str(OLD),
        original_prepared_sha256=sha(OLD/'inputs.json'),overlay_builder_sha256=sha(__file__),
        files={str(p):sha(p) for p in PREPARED.iterdir() if p.is_file()},hardware_access=False)
    save(PREPARED/'inputs.json',meta);print(report['status'])
if __name__=='__main__':main()
