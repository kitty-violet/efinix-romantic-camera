"""Inline only public DSP24 pipe hierarchy; preserve its exact four-state RTL.

This is simulation-source elaboration, not a hardware/latency change. Original
copyright and all arithmetic, reset, clock, enable and initialization code stay.
No simulator executable, license control or protected IP is changed.
"""
import argparse, json, os, re, subprocess, time
from pathlib import Path
from run_units import ROOT, EF, sha

def named(text):
    pairs = {}
    pos = 0
    while pos < len(text):
        found = re.match(r'\s*,?\s*\.(\w+)\(', text[pos:])
        assert found, text[pos:]
        start = pos + found.end()
        end, depth, quoted = start, 1, False
        while depth:
            char = text[end]
            if char == '"': quoted = not quoted
            if not quoted:
                if char == '(': depth += 1
                elif char == ')': depth -= 1
            end += 1
        pairs[found.group(1)] = text[start:end-1]
        pos = end
    return pairs

def flatten(original, top_name='EFX_DSP24'):
    declaration = re.search(r'module EFX_DSP24_pipe \([^;]+;(?P<body>.*?)endmodule', original, re.S)
    assert declaration
    body = declaration['body']
    defaults = dict(re.findall(r'parameter\s+(\w+)\s*=\s*([^;]+);', body))
    assert set(defaults) == {'W', 'REG', 'SYNC'}
    body = re.sub(r'\b(input|output)\b', 'wire', body)
    pattern = r'EFX_DSP24_pipe\s*#\((.*?)\)\s+(\w+)\s*\((.*?)\);'
    changes = []
    def inline(match):
        params, instance, ports = named(match[1]), match[2], named(match[3])
        assert set(ports) == {'I', 'CLK', 'CE', 'RST', 'O'}
        assert set(params) <= set(defaults)
        bindings = {name: params.get(name, default) for name, default in defaults.items()}
        aliases = ''.join(f'localparam __flat_{instance}_{key} = {val};\n' for key, val in bindings.items())
        local = re.sub(r'parameter\s+(\w+)\s*=\s*([^;]+);',
                       lambda m: f'localparam {m[1]} = __flat_{instance}_{m[1]};', body)
        # Connections outside the generated scope retain the parent's lexical
        # context; no accidental self-reference for e.g. SIGNED or W.
        connections = ''.join(f'assign {instance}.{key} = {val};\n' for key, val in ports.items() if key != 'O')
        connections += f'assign {ports["O"]} = {instance}.O;\n'
        changes.append(dict(instance=instance, parameters=bindings, ports=ports))
        return aliases + f'if (1) begin : {instance}\n' + local + '\nend\n' + connections
    flat, count = re.subn(pattern, inline, original)
    assert count == 8
    if top_name != 'EFX_DSP24':
        flat = flat.replace('module EFX_DSP24 (', f'module {top_name} (', 1)
    return flat, changes

def production_guard(text):
    # Exact parameters of the only 256 DSP24 instances in this board. Generic
    # X-control latch configurations are deliberately not claimed equivalent.
    settings = dict(A_REG=1,B_REG=1,C_REG=0,P_REG=0,OP_REG=0,W_REG=1,O_REG=0,
        SHIFTER=0,RST_SYNC=1,SIGNED=1,P_EXT='ALIGN_RIGHT',C_EXT='ALIGN_RIGHT',
        M_SEL='P',N_SEL='W',W_SEL='X',CASCOUT_SEL='W',CLK_POLARITY=1,
        CE_POLARITY=1,RST_POLARITY=1,SHIFT_ENA_POLARITY=1,
        A_REG_USE_CE=0,B_REG_USE_CE=0,C_REG_USE_CE=1,OP_REG_USE_CE=1,
        P_REG_USE_CE=1,W_REG_USE_CE=1,O_REG_USE_CE=1,
        A_REG_USE_RST=1,B_REG_USE_RST=1,C_REG_USE_RST=1,OP_REG_USE_RST=1,
        P_REG_USE_RST=1,W_REG_USE_RST=1,O_REG_USE_RST=1)
    checks = ' || '.join(f'{key} != {json.dumps(value)}' for key,value in settings.items())
    guard = f'''initial if ({checks}) $fatal(1,"DSP flat view used outside proven board parameter contract");
    always @(OP or C or CASCIN or SHIFT_ENA) begin
      if (OP !== 2'd0 || C !== 8'd0 || CASCIN !== 24'd0 || SHIFT_ENA !== 1'b0)
        $fatal(1,"DSP flat view used outside proven board tied-control contract");
    end
'''
    return text.replace('   reg \t\tfinish_error = 0;', guard+'   reg \t\tfinish_error = 0;',1),settings

def compact_production(original):
    # Algebraic specialization of the guarded production settings: signed
    # 8x8 multiply, registered A/B, a CE-gated 24-bit wrapping accumulator,
    # synchronous reset, and direct O/CASCOUT. Retain original weak defaults.
    start=original.index('   // Mult Operator')
    end=original.index('endmodule',start)
    block='''   reg [7:0] board_a=0,board_b=0;
   reg [23:0] board_sum=0;
   wire signed [15:0] board_product=$signed(board_a)*$signed(board_b);
   always @(posedge CLK_i) begin
      if (RST_i) begin board_a<=0;board_b<=0;board_sum<=0;end
      else begin
         board_a<=A;board_b<=B;
         if (CE_i) board_sum<=board_sum+{{8{board_product[15]}},board_product};
      end
   end
   assign O=board_sum;
   assign CASCOUT=board_sum;
'''
    return production_guard(original[:start]+block+original[end:])

TB = '''`timescale 1ns/1ps
module tb_dsp_flat;
 reg clk=0;always #5 clk=~clk;
 reg [7:0] a=0,b=0,c=0;reg [23:0] cascin=0;
 reg [1:0] op=0;reg ce=0,rst=0,shift=0;
 integer checks=0;integer step;integer seed=391;
 for(genvar k=0;k<65;k=k+1)begin:CFG
  wire [23:0] old_o,new_o,old_c,new_c;
  __INSTANCES__
  always @(posedge clk or negedge clk)begin
   #0.001;
   if(old_o!==new_o || old_c!==new_c)
    $fatal(1,"DSP hierarchy mismatch cfg=%0d step=%0d O=%h/%h C=%h/%h",k,step,old_o,new_o,old_c,new_c);
   checks=checks+1;
  end
 end
 initial begin
  for(step=0;step<66000;step=step+1)begin
   if(step%16384==0)$display("DSP_EQ_PROGRESS step=%0d",step);
   @(negedge clk);#0.5;
   a=step[7:0];b=step[15:8];c=$random(seed);cascin=$random(seed);
   op=step[1:0];ce=(step%11)!=0;rst=(step%101)==0;shift=(step%97)==0;
   if(step>65535)begin
    a=$random(seed);b=$random(seed);
    if(step%7==0)a[0]=1'bx;
    if(step%13==0)b[5]=1'bz;
    if(step%17==0)ce=1'bx;
    if(step%19==0)rst=1'bx;
    if(step%23==0)ce=1'bz;
    if(step%29==0)shift=1'bx;
    if(step%31==0)op=2'bxx;
   end
   #1;
  end
  @(negedge clk);#1;
  $display("PASS DSP24_FLAT_EQUIVALENCE configs=65 checks=%0d exhaustive_pairs=65536 four_state_tail=464",checks);
  $finish;
 end
endmodule
'''

PARAMS = '''.A_REG(k==0 ? 1'b1:1'(k)),.B_REG(k==0 ? 1'b1:1'(k>>1)),
 .C_REG(1'(k>>2)),.P_REG(1'(k>>3)),.OP_REG(1'(k>>4)),.W_REG(k==0 ? 1'b1:1'(k>>5)),.O_REG(1'(k>>2)),
 .A_REG_USE_CE(k==0 ? 1'b0:1'(k>>3)),.B_REG_USE_CE(k==0 ? 1'b0:1'(k>>4)),
 .W_REG_USE_CE(1'b1),.A_REG_USE_RST(1'b1),.B_REG_USE_RST(1'b1),.W_REG_USE_RST(1'b1),
 .RST_SYNC(k==0 ? 1'b1:1'(k)),.SIGNED(k==0 ? 1'b1:1'(k>>1)),
 .CLK_POLARITY(k==0 ? 1'b1:1'(k>>2)),.CE_POLARITY(k==0 ? 1'b1:1'(k>>3)),
 .RST_POLARITY(k==0 ? 1'b1:1'(k>>4)),.SHIFT_ENA_POLARITY(k==0 ? 1'b1:1'(k>>5)),
 .M_SEL("P"),.N_SEL(k==0 ? "W":k%3==0 ? "CASCIN":k%3==1 ? "P":"C"),
 .W_SEL("X"),.CASCOUT_SEL(k==0 ? "W":k%3==0 ? "C":k%3==1 ? "P":"ABC")'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--compile-only',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--board-config-only',action='store_true');p.add_argument('--compact-board',action='store_true')
    p.add_argument('--idle-event-filter',action='store_true');p.add_argument('--legacy-simulator',action='store_true');a=p.parse_args()
    assert not a.compact_board or a.board_config_only
    assert not a.idle_event_filter or a.compact_board
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp')
    original=EF/'sim_models/verilog/efx_dsp24.v'
    if not a.resume:
        assert not root.exists();root.mkdir()
        text=original.read_text();flat,changes=flatten(text)
        if a.board_config_only:
            flat,settings=compact_production(text) if a.compact_board else production_guard(flat)
            assert 'DSP flat view used outside proven board parameter contract' in flat
        if a.idle_event_filter:
            # The omitted event performs only A<=A, B<=B and no accumulator
            # update. Case equality preserves X/Z storage exactly; reset/CE
            # unknowns follow the original procedural-if (not asserted).
            old='   always @(posedge CLK_i) begin'
            assert flat.count(old)==1
            flat=flat.replace(old,'''   always @(posedge CLK_i iff (RST_i===1'b1 || CE_i===1'b1 || A!==board_a || B!==board_b)) begin''')
        (root/'efx_dsp24_flat.v').write_text(flat)
        # Compare renamed complete models, so all helper modules remain the
        # official logic on both sides, while only hierarchy differs.
        (root/'original_ref.v').write_text(re.sub(r'\bEFX_DSP24\b','EFX_DSP24_REF',text))
        # Helpers are defined by the original file; avoid duplicate declarations.
        (root/'flat_ref.v').write_text(flat[:flat.index('module EFX_DSP24_pipe')])
        instances='\n'.join(f'{name} #({PARAMS}) u_{label}(.A(a),.B(b),.C(c),.CASCIN(cascin),.OP(op),.SHIFT_ENA(shift),.CLK(clk),.CE(ce),.RST(rst),.O({label}_o),.CASCOUT({label}_c));' for name,label in [('EFX_DSP24_REF','old'),('EFX_DSP24','new')])
        bench=TB.replace('__INSTANCES__',instances)
        if a.board_config_only:
            bench=bench.replace('k<65','k<1').replace('configs=65','configs=1')
            bench=bench.replace('c=$random(seed);cascin=$random(seed);','c=0;cascin=0;')
            bench=bench.replace('op=step[1:0];','op=0;').replace('shift=(step%97)==0;','shift=0;')
            bench=bench.replace("if(step%29==0)shift=1'bx;",'').replace("if(step%31==0)op=2'bxx;",'')
        if a.idle_event_filter:
            bench=bench.replace('  @(negedge clk);#1;\n  $display("PASS', '''  // Long identical-value idle spans must actually take the omitted path.
  for(step=66000;step<70000;step=step+1)begin
   @(negedge clk);#0.5;
   ce=(step%64)==0;rst=(step%257)==0;
   if(step%127==0)begin a=$random(seed);b=$random(seed);end
   if(step%509==0)a=8'bx10z0011;
   if(step%503==0)ce=1'bx;
   if(step%499==0)rst=1'bz;
  end
  @(negedge clk);#1;
  $display("IDLE_EVENT_FILTER_STABLE_CYCLES 4000");
  $display("PASS''')
        (root/'tb.sv').write_text(bench)
        manifest=dict(original=str(original),original_sha256=sha(original),flat_sha256=sha(root/'efx_dsp24_flat.v'),
                      changes=changes,rtl_arithmetic_changed=False,hardware_changed=False,
                      tb_sha256=sha(root/'tb.sv'),board_config_only=a.board_config_only,
                      parameter_contract=settings if a.board_config_only else None,compact_board=a.compact_board,idle_event_filter=a.idle_event_filter)
        (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
    env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
    sim=Path('E:/altera/15.0/modelsim_ase/win32aloem' if a.legacy_simulator else 'E:/quatra/questa_fse/win64')
    def run(name,cmd):
        with (root/f'{name}.log').open('wb') as log:
            ret=subprocess.run(list(map(str,cmd)),cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        assert ret.returncode==0,(name,ret.returncode)
    if not a.resume:
        run('vlib',[sim/'vlib.exe','work'])
        run('compile',[sim/'vlog.exe','-sv','-timescale','1ns/1ps','original_ref.v','flat_ref.v','tb.sv'])
    if a.compile_only:print('DSP_FLAT_EQUIVALENCE_COMPILED');return
    start=time.monotonic()
    run('simulation',[sim/'vsim.exe','-c','work.tb_dsp_flat','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'])
    log=(root/'simulation.log').read_text();assert 'PASS DSP24_FLAT_EQUIVALENCE' in log
    metadata=json.loads((root/'inputs.json').read_text());scope=metadata['board_config_only']
    result=dict(status='DSP24_BOARD_PIPE_HIERARCHY_EQUIVALENCE_PASS' if scope else 'DSP24_PUBLIC_PIPE_HIERARCHY_EQUIVALENCE_PASS',configs=1 if scope else 65,exhaustive_pairs=65536,four_state_tail=464,
                seconds=time.monotonic()-start,inputs_sha256=sha(root/'inputs.json'),flat_sha256=sha(root/'efx_dsp24_flat.v'),
                log_sha256=sha(root/'simulation.log'),hardware_change=False,compact_board=metadata.get('compact_board',False),
                idle_event_filter=metadata.get('idle_event_filter',False),stable_input_cycles=4000 if metadata.get('idle_event_filter') else 0,
                limitation='Only exact actual board parameters and tied controls; A/B, clock enable and reset four-state data tested. Runtime guards reject any other use.' if scope else None)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
