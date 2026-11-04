"""Read-only source audit and bounded pin-level I2C regression; no USB access."""
import argparse, hashlib, json, os, re, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def uncomment(s): return re.sub(r'/\*.*?\*/|//[^\n]*','',s,flags=re.S)
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--simulator',choices=['modelsim','iverilog'],default='modelsim');a=p.parse_args()
    out=a.out.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists();out.mkdir(parents=True)
    base=ROOT/'tmp/frame-execution/connected-board-local-drivers1'
    original=ROOT/'tmp/board-test/vendor-camera-original-v6/Ti60f225_sc431hai2hdmi_v6'
    source=base/'rtl/cam_i2c_ctrl'
    files=[source/'i2c'/f for f in ('i2c_master_ctrl_top.v','i2c_master_reg_set.v','i2c_master_reg_rom.v','i2c_16addr_8data.v')]
    files += [source/'i2c_master'/f for f in ('i2c_master_top.v','i2c_master_byte_ctrl.v','i2c_master_bit_ctrl.v')]
    records=[]
    for f in files:
        old=original/f.relative_to(base)
        records.append(dict(path=str(f),sha256=sha(f),vendor_identical=f.read_bytes()==old.read_bytes()))
    assert all(x['vendor_identical'] for x in records)
    rom=uncomment((source/'i2c/i2c_master_reg_rom.v').read_text(encoding='utf-8-sig'))
    rows=re.findall(r"ROM\[8'h([0-9a-f]+)\]\s*=\s*\{16'h([0-9a-f]+),\s*8'h([0-9a-f]+),\s*1'b([01])\}",rom,re.I)
    values=[tuple(int(x,16) for x in row) for row in rows]
    assert len(values)==164 and [r[0] for r in values]==list(range(164)) and all(r[3]==0 for r in values)
    (out/'expected.hex').write_text(''.join(f'60{addr:04x}{value:02x}\n' for _,addr,value,_ in values),encoding='ascii')
    regs={addr:value for _,addr,value,_ in values}
    result=dict(status='RUNNING',hardware_access=False,sources=records,rom_records=164,
      configured_device_8bit=0x60,configured_device_7bit=0x30,
      configured_width=(regs[0x3208]<<8)|regs[0x3209],configured_height=(regs[0x320a]<<8)|regs[0x320b],
      unwritten_key_registers=[hex(r) for r in (0x3018,0x3031,0x3037,0x3107,0x3108) if r not in regs],cases=[])
    compiler=ROOT/'tools/iverilog/ucrt64/bin/iverilog.exe';runtime=compiler.with_name('vvp.exe')
    env=os.environ.copy();env['PATH']=str(compiler.parent)+os.pathsep+env.get('PATH','')
    env['TEMP']=env['TMP']=env['TMPDIR']=str(out)
    tb=Path(__file__).with_name('tb_camera_config_audit.sv')
    cmd=[str(compiler),'-g2012','-I',str(source/'i2c_master'),'-s','tb_camera_config_audit','-o',str(out/'sim.vvp'),str(tb),*map(str,files)]
    ms=Path('E:/altera/15.0/modelsim_ase/win32aloem')
    if a.simulator=='modelsim':
        r=subprocess.run([str(ms/'vlib.exe'),'work'],cwd=out,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=20)
        (out/'vlib.log').write_text(r.stdout,encoding='utf-8');assert r.returncode==0,r.stdout
        cmd=[str(ms/'vlog.exe'),'-sv','+incdir+'+str(source/'i2c_master'),str(tb),*map(str,files)]
    r=subprocess.run(cmd,cwd=out,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30)
    (out/'compile.log').write_text(r.stdout,encoding='utf-8')
    (out/'compile_result.json').write_text(json.dumps(dict(command=cmd,exit_code=r.returncode)),encoding='utf-8')
    assert r.returncode==0,(r.returncode,r.stdout)
    for mode in range(3):
        start=time.monotonic()
        simcmd=[str(runtime),str(out/'sim.vvp'),f'+ACK_MODE={mode}']
        if a.simulator=='modelsim':
            simcmd=[str(ms/'vsim.exe'),'-c','work.tb_camera_config_audit',f'+ACK_MODE={mode}','-do','onerror {quit -code 1}; run -all; quit -code 0']
        r=subprocess.run(simcmd,cwd=out,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        (out/f'mode{mode}.log').write_text(r.stdout,encoding='utf-8')
        result['cases'].append(dict(mode=mode,exit_code=r.returncode,seconds=round(time.monotonic()-start,3),output=r.stdout))
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        assert r.returncode==0 and 'PASS CONFIG_AUDIT' in r.stdout,r.stdout
    result['status']='CAMERA_SOURCE_AND_NACK_BLINDNESS_REPRODUCED'
    result['tb_sha256']=sha(tb);result['runner_sha256']=sha(__file__)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
