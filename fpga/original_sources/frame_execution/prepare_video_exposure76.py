"""Exposure-only overlay on the released73 circuit; no device access."""
import hashlib,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from prepare_video_service_color_exposure import read_rom

BASE=ROOT/'tmp/frame-execution/video-sensor73-bits9'
OUT=ROOT/'tmp/frame-execution/video-exposure76-candidate1'
PREPARED=ROOT/'tmp/frame-execution/video-exposure76-pin-prepared1'
OLD=ROOT/'tmp/frame-execution/video-sensor73-pin-prepared2'
MANUAL=ROOT/'output/toolchain/project_evidence_audit_2026-09-28/pdf_text/摄像头_摄像头SC431HAI_数据手册_V1.1_dfe1737d7ee2.txt'

def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
 assert not OUT.exists() and not PREPARED.exists()
 mapped=BASE/'outflow/ti60f225_oob.map.v'
 assert sha(mapped)=='07918fc6c80ee639828ef33785ea6a138fab4b47070d067d0b48ef400b7d74de'
 text=mapped.read_text(encoding='utf8');words,blocks=read_rom(text)
 lo,hi,block,width,blob=next(b for b in blocks if b[3]==20)
 assert [words[i] for i in (115,116,117,161,162)]==[0x3e00<<9,0x3e01<<9|0x46<<1,0x3e02<<9,0x3e08<<9|0x80<<1,0x3e09<<9|0x34<<1]
 # 1344 half-lines = 0x0540; only 3e01 changes, from46 to54.
 newwords=list(words);newwords[116]=0x3e01<<9|0x54<<1
 updated=(blob&~(0xfffff<<(116*20)))|((newwords[116]&0xfffff)<<(116*20))
 newblock=block;patches=[]
 for i in range(40):
  old=f'{(blob>>(i*256))&((1<<256)-1):0256b}';new=f'{(updated>>(i*256))&((1<<256)-1):0256b}'
  if old==new:continue
  key='INIT_'+format(i,'X')
  newblock,n=re.subn('('+key+r"\s*=\s*256'b)"+old,lambda m:m[1]+new,newblock)
  assert n==2
  patches.append(dict(parameter=key,old=old,new=new,occurrences=n))
 effective=text[:lo]+newblock+text[hi:]
 assert read_rom(effective)[0]==newwords
 assert effective[:lo]+block+effective[lo+len(newblock):]==text
 assert ((blob^updated)>>(256*20))==0,'verification bank must not change'
 oldmeta=json.loads((OLD/'inputs.json').read_text())
 assert oldmeta['map_sha256']==sha(mapped)
 for f,h in oldmeta['files'].items():assert sha(f)==h
 body=(OLD/'mapped_camera.v').read_text(encoding='utf8');newbody=body
 for patch in patches:
  # Extracted primitive uses hexadecimal defparams, one occurrence per INIT.
  pat=(r'(defparam \\camera_checked/config_rom/ROM__D\$2 \.'+patch['parameter']+r"=256'h)"+f"{int(patch['old'],2):x}"+r';')
  newbody,n=re.subn(pat,lambda m:m[1]+f"{int(patch['new'],2):x};",newbody);assert n==1
 OUT.mkdir();PREPARED.mkdir()
 (OUT/'configuration.hex').write_text(''.join(f'{w:07x}\n' for w in newwords))
 report=dict(status='EXPOSURE76_ROM_EXACT_DELTA_PASS_NOT_RELEASED',source_sha256=sha(mapped),
  candidate_native_sha256=hashlib.sha256(effective.encode()).hexdigest(),candidate_native_written=False,
  changed_addresses=[116],changes={'116':dict(register=0x3e01,old=0x46,new=0x54)},native_init_patches=patches,
  configuration_words=newwords,compared_register_words=164,untouched_native_text_byte_equal=True,
  exposure_before_half_lines=1120,exposure_after_half_lines=1344,relative_exposure=1.2,
  gain_before=2.503,gain_after=2.503,verification_bank_unchanged=True,clocks_pins_topology_A_unchanged=True,
  hardware_access=False,board_release=False,evidence={str(f):sha(f) for f in (Path(__file__),mapped,BASE/'native_source.v',MANUAL,OLD/'inputs.json')})
 save(OUT/'result.json',report)
 (PREPARED/'mapped_camera.v').write_text(newbody,encoding='utf8')
 (PREPARED/'expected.hex').write_text(''.join(f'60{w>>9:04x}{(w>>1)&255:02x}\n' for w in newwords))
 for kind in oldmeta['kinds']:
  name=kind.lower()+'.v';assert sha(OLD/name)==sha(EF/'sim_models/verilog'/name)
  shutil.copy2(OLD/name,PREPARED/name)
 shutil.copy2(OLD/'tb.sv',PREPARED/'tb.sv')
 meta=dict(oldmeta);meta.update(candidate=str(OUT),candidate_sha256=sha(OUT/'result.json'),
  original_prepared=str(OLD),original_prepared_sha256=sha(OLD/'inputs.json'),overlay_builder_sha256=sha(__file__),
  files={str(p):sha(p) for p in PREPARED.iterdir() if p.is_file()},hardware_access=False)
 save(PREPARED/'inputs.json',meta)
 print(report['status'],flush=True)

if __name__=='__main__':main()
