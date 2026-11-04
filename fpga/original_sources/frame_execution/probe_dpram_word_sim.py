"""A/B the public RAM model's storage representation, never the running board.

All timing, polarity, output, reset and validation code remains verbatim.
Only the homogeneous ten-bit memory uses words instead of ten scalar lookups.
Other configurations retain the original path. No installed model is edited.
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from run_units import ROOT, EF, sha


def word_view(original):
    old = 'reg mem [MEMORY_SIZE-1:0];'
    assert original.count(old) == 1
    text = original.replace(old, old + '''
   localparam WORD10 = READ_WIDTH_A==10 && WRITE_WIDTH_A==10 &&
                        READ_WIDTH_B==10 && WRITE_WIDTH_B==10;
   reg [9:0] word_mem [0:1023];
   localparam [10239:0] WORD_INIT = {''' + ','.join(
        f'INIT_{i:X}' for i in reversed(range(40))) + '''};
   initial if (WORD10) begin
      for (integer wi=0; wi<1024; wi=wi+1)
         word_mem[wi] = WORD_INIT[wi*10+:10];
   end
''')
    for port in ('A', 'B'):
        pattern = (rf'for \(i=0; i < READ_WIDTH_{port}; i=i\+1\)\s*'
                   rf'rdata\[i\] = mem\[addr\*READ_WIDTH_{port}\+i\];')
        text, count = re.subn(pattern, lambda m: 'if (WORD10) rdata=word_mem[addr];\n'
                             '         else ' + m[0], text)
        assert count == 1
        pattern = (rf'if \(we\)\s*for \(i=0; i < WRITE_WIDTH_{port}; i=i\+1\)\s*'
                   rf'mem\[addr\*WRITE_WIDTH_{port}\+i\] = wdata\[i\];')
        replacement = f'''if (we) begin
            if (WORD10) word_mem[addr]=wdata;
            else for (i=0; i < WRITE_WIDTH_{port}; i=i+1)
               mem[addr*WRITE_WIDTH_{port}+i] = wdata[i];
         end'''
        text, count = re.subn(pattern, replacement, text)
        assert count == 1
    return text


TB = '''`timescale 1ns/1ps
module tb_ram;
 parameter MODE=0; // 0 equivalence, 1 official speed, 2 word speed
 parameter N=37, STEPS=10000;
 reg clk=0;always #5 clk=~clk;
 reg [9:0] a=0,b=0,wa=0,wb=0;
 reg wea=0,web=0,cea=1,ceb=1,aea=1,aeb=1,rsta=0,rstb=0;
 integer step,seed=12901,checks=0;
 for(genvar k=0;k<N;k=k+1)begin:G
  localparam WIDTH=(MODE==0 && k==36) ? 8:10;
  wire [WIDTH-1:0] qa0,qb0,qa1,qb1;
  __INSTANCES__
  if(MODE==0)always @(posedge clk or negedge clk)begin
   #0.001;
   if(qa0!==qa1 || qb0!==qb1)
    $fatal(1,"RAM_EQ_FAIL cfg=%0d step=%0d a=%h/%h b=%h/%h",k,step,qa0,qa1,qb0,qb1);
   checks=checks+1;
  end
 end
 initial begin
  for(step=0;step<STEPS;step=step+1)begin
   @(negedge clk);#1;
   if(step<1024)begin
    a=step;b=step;wea=0;web=0;
   end else begin
    // Non-overlapping writes; read/write collision is exercised explicitly.
    a={1'b0,step[8:0]};b={1'b1,step[8:0]};
    wea=(step%3)!=0;web=(step%5)!=0;
    if(step%7==0)begin b=a;web=0;end
    wa=$random(seed);wb=$random(seed);
    cea=(step%11)!=0;ceb=(step%13)!=0;
    aea=(step%17)!=0;aeb=(step%19)!=0;
    // A previous explicit cross-port read collision can leave B's held
    // address in A's half. Refresh B's address before enabling a write.
    // Otherwise step 2584 accidentally performs undefined dual writes.
    if(web)aeb=1;
    rsta=(step%101)==0;rstb=(step%103)==0;
   end
   if(MODE==0 && step>8000)begin
    if(step%23==0)a=10'bx000z01010;
    if(step%29==0)b=10'bz000x01010;
    if(step%31==0)wa[0]=1'bx;
    if(step%37==0)wb[8]=1'bz;
    if(step%41==0)cea=1'bx;
    if(step%43==0)ceb=1'bz;
    if(step%47==0)wea=1'bx;
    if(step%53==0)web=1'bz;
    if(step%59==0)aea=1'bx;
    if(step%61==0)aeb=1'bz;
    if(step%67==0)rsta=1'bx;
    if(step%71==0)rstb=1'bz;
   end
  end
  @(negedge clk);#1;
  $display("PASS DPRAM_WORD_PROBE mode=%0d configs=%0d steps=%0d checks=%0d",MODE,N,STEPS,checks);
  $finish;
 end
endmodule
'''

PARAMS = '''.READ_WIDTH_A(WIDTH),.WRITE_WIDTH_A(WIDTH),.READ_WIDTH_B(WIDTH),.WRITE_WIDTH_B(WIDTH),
 .OUTPUT_REG_A((k/9)%2),.OUTPUT_REG_B((k/18)%2),
 .WRITE_MODE_A(k%3==0 ? "READ_FIRST":k%3==1 ? "WRITE_FIRST":"NO_CHANGE"),
 .WRITE_MODE_B((k/3)%3==0 ? "READ_FIRST":(k/3)%3==1 ? "WRITE_FIRST":"NO_CHANGE"),
 .RESET_RAM_A(k%3==0 ? "ASYNC":k%3==1 ? "SYNC":"NONE"),
 .RESET_RAM_B((k/3)%3==0 ? "ASYNC":(k/3)%3==1 ? "SYNC":"NONE"),
 .RESET_OUTREG_A(k%2==0 ? "ASYNC":"NONE"),.RESET_OUTREG_B(k%2==0 ? "NONE":"ASYNC"),
 .CLKA_POLARITY(k%2),.CLKB_POLARITY(k%2),
 .INIT_0(256'h123456789abcdeffedcba98765432100123456789abcdef00123456789abcdef00),
 .INIT_1(256'hz123x6789abcdeffedcba98765432100123456789abcdef00123456789abcdef00),
 .INIT_27(256'h89abcdef0123456789abcdef0123456789abcdef0123456789abcdef01234567)'''
PORTS = '''.CLKA(clk),.WEA(wea),.CLKEA(cea),.RSTA(rsta),.ADDRENA(aea),.WDATAA(wa[WIDTH-1:0]),.ADDRA(a),.RDATAA(qa{n}),
 .CLKB(clk),.WEB(web),.CLKEB(ceb),.RSTB(rstb),.ADDRENB(aeb),.WDATAB(wb[WIDTH-1:0]),.ADDRB(b),.RDATAB(qb{n})'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sim-bin', type=Path, default=Path('E:/altera/15.0/modelsim_ase/win32aloem'))
    args = parser.parse_args()
    root = args.root.resolve()
    assert root.is_relative_to(ROOT / 'tmp') and not root.exists()
    root.mkdir()
    official = EF / 'sim_models/verilog/efx_dpram10.v'
    source = official.read_text()
    (root / 'official.v').write_text(source.replace('EFX_DPRAM10', 'EFX_DPRAM10_REF'))
    (root / 'word.v').write_text(word_view(source))
    instances = '\n'.join(f'if(MODE!={exclude})begin:B{n}\n{module} #({PARAMS}) u({PORTS.format(n=n)});\nend'
                           for n, module, exclude in [(0, 'EFX_DPRAM10_REF', 2), (1, 'EFX_DPRAM10', 1)])
    (root / 'tb.sv').write_text(TB.replace('__INSTANCES__', instances))
    manifest = dict(official=str(official), official_sha256=sha(official),
                    source_sha256=sha(Path(__file__)), word_sha256=sha(root/'word.v'),
                    tb_sha256=sha(root/'tb.sv'), simulator_bin=str(args.sim_bin.resolve()), hardware_change=False,
                    running_board_change=False, scope='10-bit homogeneous word storage; original path otherwise')
    (root/'inputs.json').write_text(json.dumps(manifest, indent=2))
    sim = args.sim_bin.resolve()
    env=os.environ.copy()
    env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
    def run(name, command):
        start = time.monotonic()
        with (root/f'{name}.log').open('wb') as log:
            result = subprocess.run([str(x) for x in command], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=240)
        text = (root/f'{name}.log').read_text(errors='replace')
        assert result.returncode == 0 and '** Error' not in text and '** Fatal' not in text, name
        return time.monotonic()-start, text
    run('vlib', [sim/'vlib.exe', 'work'])
    run('compile', [sim/'vlog.exe', '-sv', '-timescale', '1ns/1ps', 'official.v', 'word.v', 'tb.sv'])
    measurements = {}
    for mode, name in [(0, 'equivalence'), (1, 'official_speed'), (2, 'word_speed')]:
        elapsed, log = run(name, [sim/'vsim.exe', '-c', *(['-voptargs=-O5'] if 'questa' in str(sim).lower() else []), 'work.tb_ram', f'-gMODE={mode}',
                                '-gN=37' if mode==0 else '-gN=84',
                                '-gSTEPS=10000' if mode==0 else '-gSTEPS=30000',
                                '-do', 'onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'])
        assert f'PASS DPRAM_WORD_PROBE mode={mode}' in log, name
        measurements[name] = dict(seconds=elapsed, log_sha256=sha(root/f'{name}.log'))
        print(name, round(elapsed, 3), flush=True)
    result = dict(status='PUBLIC_DPRAM_WORD_EQUIVALENCE_PASS', inputs_sha256=sha(root/'inputs.json'),
                  measurements=measurements,
                  host_speed_ratio=measurements['official_speed']['seconds']/measurements['word_speed']['seconds'],
                  simulator_bin=str(sim),
                  limitations=['Isolated primitive result; not a whole-board speed result.',
                               'Simultaneous same-address dual writes are undefined by the official model and not claimed.',
                               'No running model, installed file, PHY, license or hardware RTL was changed.'])
    (root/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
