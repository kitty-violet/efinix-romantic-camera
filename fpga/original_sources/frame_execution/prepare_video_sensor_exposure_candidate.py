"""Offline exposure-only candidate; preserve gain/readback and all circuitry."""
import argparse,json,re,hashlib
from pathlib import Path
from prepare_video_sensor_gain_candidate import NATIVE,NATIVE_SHA,ROM,MANUAL
from prepare_video_service_color_exposure import ROOT,read_rom,sha


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
 assert root.is_relative_to(ROOT/'tmp/frame-execution') and not root.exists()
 assert sha(NATIVE)==NATIVE_SHA
 text=NATIVE.read_text(encoding='utf8');words,blocks=read_rom(text)
 changes={0x74:(0x3e01,0xba,0x5d),0x75:(0x3e02,0xd0,0x60)}
 lo,hi,block,width,blob=next(b for b in blocks if b[3]==20);updated=blob
 rom=ROM.read_text(encoding='utf8')
 for i,(reg,old,new) in changes.items():
  assert words[i]==reg<<9|old<<1
  updated=(updated&~(0xfffff<<(i*20)))|(((reg<<9|new<<1)&0xfffff)<<(i*20))
  pat=rf"(ROM\[8'h{i:02x}\]\s*=\s*\{{16'h{reg:04x},8'h){old:02x}"
  rom,n=re.subn(pat,lambda m:m[1]+f'{new:02x}',rom,flags=re.I);assert n==1
 newblock=block;edits=[]
 for i in range(40):
  old=f'{(blob>>(i*256))&((1<<256)-1):0256b}';new=f'{(updated>>(i*256))&((1<<256)-1):0256b}'
  if new==old:continue
  key='INIT_'+format(i,'X');newblock,n=re.subn('('+key+r"\s*=\s*256'b)"+old,lambda m:m[1]+new,newblock);assert n==2
  edits.append(dict(parameter=key,old=old,new=new,occurrences=n))
 candidate=text[:lo]+newblock+text[hi:];actual,_=read_rom(candidate)
 expected=list(words)
 for i,(r,o,n) in changes.items():expected[i]=r<<9|n<<1
 assert actual==expected and candidate[:lo]+block+candidate[lo+len(newblock):]==text
 assert words[0x73]==actual[0x73]==0x3e00<<9
 assert actual[0xa1]==0x3e08<<9|0x80<<1 and actual[0xa2]==0x3e09<<9|0x2a<<1
 root.mkdir();target=root/'candidate_i2c_master_reg_rom.v';target.write_text(rom,encoding='utf8')
 result=dict(status='OFFLINE_EXPOSURE_ROM_EXACT_DELTA_PASS_NOT_RELEASED',source_sha256=sha(NATIVE),
  candidate_native_sha256=hashlib.sha256(candidate.encode()).hexdigest(),candidate_rom_sha256=sha(target),
  candidate_native_written=False,changed_addresses=list(changes),changes={str(i):dict(register=r,old=o,new=n) for i,(r,o,n) in changes.items()},
  native_init_patches=edits,compared_register_words=164,untouched_native_text_byte_equal=True,
  exposure_before_half_lines=2989,exposure_after_half_lines=1494,relative_exposure=1494/2989,
  gain_before=2.021,gain_after=2.021,gain_readback_contract_unchanged=True,
  clocks_pins_topology_A_unchanged=True,hardware_access=False,board_release=False,
  rejected_gain_only_candidate='video-sensor-gain70-bram3: ROM updater passes but gain verify constants still 80/2a; NOT RELEASED',
  evidence={str(f):sha(f) for f in (Path(__file__),NATIVE,ROM,MANUAL)})
 (root/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(result['status'])


if __name__=='__main__':main()
