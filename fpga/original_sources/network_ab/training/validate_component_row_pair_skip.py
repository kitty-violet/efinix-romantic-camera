"""C8/C16/C32 residual routing on the current six-bank physical interface."""
import argparse,os,re,shutil,subprocess,time
from pathlib import Path
from .train_luma_regions import ROOT,RUN
from .runtime import file_hash,write_json


def derivative(s):
    assert s.count('groups!=4 ||')==1
    s=s.replace('groups!=4 ||','(groups!=1 && groups!=2 && groups!=4) ||')
    old="{1'b0,sx[10:1],2'd0}";assert s.count(old)==1
    s=s.replace(old,"(groups==1 ? {3'd0,sx[10:1]} : groups==2 ? {2'd0,sx[10:1],1'b0} : {1'b0,sx[10:1],2'd0})")
    s=s.replace('tag[15:13]!=0 || y!=',"tag[15:13]!=0 || {1'b0,g}>=groups || y!=")
    return s


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='rtl-component-row-pair-skip-v1');a=p.parse_args();out=ROOT/'runs'/RUN/a.name;out.mkdir(exist_ok=False)
    hw=ROOT.parent/'hw/efinity/prototypes/frame_execution/variants/row_pair_arena';assembly=ROOT.parent/'tmp/frame-execution/connected-board-compact-feed1/assembly';ef=Path('E:/Efinity/2025.2/sim_models/verilog/efx_srl8.v')
    # Reuse the exact official model already present in the verified hardware run.
    candidates=list((ROOT.parent/'tmp/frame-execution').glob('row-pair-skip*/efx_srl8.v'));assert candidates
    ef=candidates[0];records=[]
    for src in [ef,assembly/'shallow_result_fifo.sv',hw/'row_pair_skip_fuse4.sv',hw/'tb_row_pair_skip_fuse4.sv']:
        dst=out/src.name;shutil.copy2(src,dst);records.append(dict(source=str(src),source_sha256=file_hash(src),snapshot=str(dst)))
    dst=out/'row_pair_skip_fuse4.sv';dst.write_text(derivative(dst.read_text()))
    tb=out/'tb_row_pair_skip_fuse4.sv';s=tb.read_text()
    for old,new in [('sent[b]/4','sent[b]/int\'(groups)'),('sent[b]%4','sent[b]%int\'(groups)'),("int'(pixels)*4","int'(pixels)*int'(groups)"),('off/4',"off/int'(groups)"),('off%4',"off%int'(groups)"),('rows*pixels*4','rows*pixels*groups'),('groups=4;skip_phase=0',"groups=3'(1<<((id/16)%3));skip_phase=0"),('j<16','j<48'),('1:groups=2','1:groups=3'),('jobs!=32','jobs!=64')]:
        assert old in s,old;s=s.replace(old,new)
    tb.write_text(s)
    for r in records:r['snapshot_sha256']=file_hash(r['snapshot'])
    write_json(out/'inputs.json',dict(records=records,script_sha256=file_hash(__file__),production_modified=False))
    exe=Path('E:/quatra/questa_fse/win64');env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
    def run(args,label):
        start=time.perf_counter()
        with (out/label).open('w',encoding='utf-8') as f:r=subprocess.run(list(map(str,args)),cwd=out,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
        assert r.returncode==0,(label,r.returncode);return time.perf_counter()-start
    run([exe/'vlib.exe','work'],'vlib.log');run([exe/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],'compile.log');assert 'Errors: 0, Warnings: 0' in (out/'compile.log').read_text()
    runs=[]
    for seed in [71,391,1027]:
        sec=run([exe/'vsim.exe','-c','work.tb_row_pair_skip_fuse4',f'+SEED={seed}','-do','onerror {quit -code 1}; run -all; quit -code 0'],f'simulation_{seed}.log')
        body=(out/f'simulation_{seed}.log').read_text();lines=re.findall(r'^# (PASS ROW_PAIR_SKIP.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
        runs.append(dict(seed=seed,seconds=sec,line=lines[0],log_sha256=file_hash(out/f'simulation_{seed}.log')));print(runs[-1],flush=True)
    write_json(out/'result.json',dict(status='C8_C16_C32_ROW_PAIR_SKIP_FOUR_STATE_PASS',runs=runs,inputs_sha256=file_hash(out/'inputs.json'),requantization='behavioral identity feedback; real MAC/RQ integration pending',real_ram=False,cnn=False,fit=False,timing=False))


if __name__=='__main__':main()
