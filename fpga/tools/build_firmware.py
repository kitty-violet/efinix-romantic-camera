"""Build the included C firmware with the RV32 GCC 15.2.0 toolchain."""
from pathlib import Path
import argparse,os,subprocess
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--toolchain',type=Path,required=True);args=ap.parse_args()
 gcc=args.toolchain.resolve();fw=ROOT/'firmware';out=ROOT/'build/firmware';out.mkdir(parents=True,exist_ok=True)
 env=dict(os.environ,PATH=str(gcc/'bin')+os.pathsep+os.environ['PATH']);prefix=str(gcc/'bin/riscv-none-elf-')
 elf=out/'control.elf'
 cmd=[prefix+'gcc.exe','-march=rv32i_zicsr_zifencei','-mabi=ilp32','-Os','-g','-ffreestanding','-fno-builtin','-fno-common','-fno-stack-protector','-fdata-sections','-ffunction-sections','-msmall-data-limit=0','-mrelax','-nostdlib','-nostartfiles','-Wall','-Wextra','-Werror','-fstack-usage','-T',str(fw/'memory.ld'),'-Wl,--gc-sections','-Wl,-Map='+str(out/'control.map'),str(fw/'start.S'),str(fw/'main.c'),str(gcc/'lib/gcc/riscv-none-elf/15.2.0/rv32i/ilp32/libgcc.a'),'-o',str(elf)]
 subprocess.run(cmd,cwd=out,env=env,check=True,timeout=60)
 subprocess.run([prefix+'objcopy.exe','-O','binary',str(elf),str(out/'control.bin')],env=env,check=True,timeout=30)
 data=(out/'control.bin').read_bytes();assert len(data)<=8192;data+=bytes(8192-len(data))
 for lane in range(4):(out/f'EfxSapphireSoc.v_toplevel_system_ramA_logic_ram_symbol{lane}.bin').write_text(''.join(f'{x:08b}\n' for x in data[lane::4]),encoding='ascii')
 print(out)
if __name__=='__main__':main()
