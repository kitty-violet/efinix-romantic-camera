"""Actual vendor CPU/JTAG four-state load at the new coefficient address.

Reuses a passed four-state fixture, never an emulated/replaced CPU. DDR stays
a stalled behavioral memory and frame completion is a loader test stimulus.
"""
import argparse,importlib.util,json,os,re,shutil,subprocess,time,zlib
from pathlib import Path
from .region_detail_target import ROOT,RUN
from .runtime import file_hash,write_json


def run(args,root,name,timeout):
    start=time.perf_counter()
    with (root/name).open('w',encoding='utf-8') as f:r=subprocess.run(list(map(str,args)),cwd=root,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    assert r.returncode==0,(name,r.returncode);return time.perf_counter()-start


def main():
    p=argparse.ArgumentParser();p.add_argument('--program',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args()
    ref=ROOT.parent/'tmp/frame-execution/real-boot-questa3';old=json.loads((ref/'result.json').read_text());assert old['status']=='PASS_REAL_BOOT_COMPLETE' and old['seconds']<600
    meta=json.loads((ref/'inputs.json').read_text());sim=Path(meta['simulator']);assert (sim/'vsim.exe').exists()
    m=json.loads((a.program/'program.json').read_text());assert m['fixed_header_windows'];address=m['coefficient_address'];assert address==0x04600000
    raw=(a.program/'coefficients-padded.bin').read_bytes();cm=json.loads((a.program/'coefficient_map.json').read_text());assert len(raw)==20480 and file_hash(a.program/'coefficients-padded.bin')==cm['sha256']
    root=ROOT/'runs'/RUN/a.name;root.mkdir(exist_ok=False);records=[]
    for r in meta['inputs']:
        src=Path(r['snapshot']);assert file_hash(src)==r['sha256'];dst=root/src.name;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=file_hash(dst)))
    source=ROOT.parent/'hw/efinity/prototypes/frame_execution/board_integration/boot_image.py';text=source.read_text();assert text.count('COEFFICIENTS = 0xE1200000')==1
    f=root/'boot_image_region.py';f.write_text(text.replace('COEFFICIENTS = 0xE1200000','COEFFICIENTS = 0xE1600000'),encoding='utf-8')
    spec=importlib.util.spec_from_file_location('boot_image_region',f);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);firmware=module.write_image(root/'firmware')
    for f in (root/'firmware').glob('EfxSapphireSoc*.bin'):shutil.copy2(f,root/f.name)
    crc=zlib.crc32(raw);vector=f"localparam [31:0] EXPECTED_CRC=32'h{crc:08x};\nfunction automatic [31:0] upload_word(input integer i);\nbegin case(i)\n"
    vector+='\n'.join(f"{i}:upload_word=32'h{int.from_bytes(raw[i*4:i*4+4],'little'):08x};" for i in range(5120));vector+="\ndefault:upload_word=32'hx;endcase end endfunction\n"
    (root/'boot_vectors.vh').write_text(vector,encoding='ascii')
    tb=(ref/'tb_real_boot.sv').read_text();assert file_hash(ref/'tb_real_boot.sv')==meta['tb_sha256']
    for before,after in [('04200000','04600000'),('04205000','04605000'),('04100000','04500000'),('e1100000','e1500000'),('e1200000','e1600000')]:
        assert before in tb;tb=tb.replace(before,after)
    (root/'tb_real_boot.sv').write_text(tb)
    write_json(root/'inputs.json',dict(program=str(a.program),program_sha256=file_hash(a.program/'program.json'),coefficients_sha256=file_hash(a.program/'coefficients-padded.bin'),
        reference_result_sha256=file_hash(ref/'result.json'),reference_seconds=old['seconds'],inputs=records,firmware=firmware,original_boot_source_sha256=file_hash(source),tb_sha256=file_hash(root/'tb_real_boot.sv'),
        actual_vendor_cpu=True,encrypted_cpu_unchanged=True,actual_jtag_upload=True,new_cpu_coefficient_address=0xE1600000,new_physical_coefficient_address=address,full_cnn=False,ddr_phy=False))
    run([sim/'vlib.exe','work'],root,'vlib.log',30)
    run([sim/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records if not r['snapshot'].endswith('.svh')],root/'tb_real_boot.sv'],root,'compile.log',180)
    results=[]
    for smoke in [1,0]:
        name='smoke.log' if smoke else 'simulation.log';seconds=run([sim/'vsim.exe','-c','work.tb_real_boot',f'+SMOKE={smoke}',
            '-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],root,name,1200)
        log=(root/name).read_text(errors='replace');marker='PASS REAL_BOOT_SMOKE' if smoke else 'PASS REAL_BOOT_COMPLETE';assert marker in log
        results.append(dict(smoke=smoke,seconds=seconds,lines=re.findall(r'^# (PASS .*)$',log,re.M)));write_json(root/'progress.json',dict(results=results));print('REGION_BOOT_STAGE',smoke,seconds,flush=True)
    assert all(file_hash(r['snapshot'])==r['sha256'] for r in records)
    write_json(root/'result.json',dict(status='REAL_VENDOR_CPU_JTAG_REGION_LOAD_FOURSTATE_PASS',results=results,inputs_sha256=file_hash(root/'inputs.json'),full_cnn=False,ddr_phy=False,board_fit=False,timing=False,uploaded_to_board=False))


if __name__=='__main__':main()
