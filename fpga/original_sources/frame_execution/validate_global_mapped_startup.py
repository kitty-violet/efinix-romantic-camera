"""Actual full-board mapped control/APB startup cut, not a behavioral CPU."""
import argparse,json,os,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,EF,sha
from windows_process_tree import stop_launched_tree
Q=Path('E:/quatra/questa_fse/win64')

def main():
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    board=a.board.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    mapped=board/'outflow/ti60f225_oob.map.v'
    receipt=json.loads((board/'result.json').read_text());assert sha(mapped)==receipt['map_sha256']
    old=ROOT/'tmp/frame-execution/snapshot-control-startup2/extractor.py'
    code=old.read_text()
    oldsource="source=ROOT/'tmp/frame-execution/connected-board-control-snapshot2/outflow/ti60f225_oob.map.v'"
    oldout="out=ROOT/'tmp/frame-execution/snapshot-control-startup2'"
    assert code.count(oldsource)==code.count(oldout)==1
    code=code.replace(oldsource,f'source=Path({str(mapped)!r})').replace(oldout,f'out=Path({str(root)!r})')
    code=code.replace('02cb1d5e1f6f0a290fc2c98b7d38aac14dbc2ad116a76064a12244bf88b19ef4',sha(mapped))
    # Global replication packs both reset outputs into one COMB4. Cutting one
    # output while retaining that cell's other output would create two drivers.
    # Keep the actual reset decode; stimulate the real external reset input.
    code=code.replace("'process_clk_150','w_arstn','jtag_inst1_TCK'",
                      "'process_clk_150','jtag_inst1_TCK'")
    old_used="for c in selected:\n    used|=deps[c]\n    for n,d in drivers.items():\n        if c in d:used.add(n)"
    new_used="for c in selected:used|=deps[c]\nfor n,d in drivers.items():\n    if d & selected:used.add(n)"
    assert code.count(old_used)==1;code=code.replace(old_used,new_used)
    exec(compile(code,str(old),'exec'),{'__name__':'_extract_only','print':lambda *args,**kwargs:None})
    (root/'extractor.py').write_text(code);ex=json.loads((root/'extraction.json').read_text())
    lines=['`timescale 1ns/1ps','module tb;'];connections=[]
    for original,port in ex['inputs'].items():
        lines.append(f'reg {ex["input_widths"][original]} {port}=0;');connections.append(f'.{port}({port})')
    for n in ex['outputs']:lines.append(f'wire {n};');connections.append(f'.{n}({n})')
    lines.append('mapped_control dut('+','.join(connections)+');')
    ports=ex['inputs'];clk=ports['process_clk_150'];reset=ports['i_arstn']
    lines.append(f'always #3.333 {clk}=~{clk};')
    release=['i_arstn','i_pll_locked','pll_locked','user_pll_locked','process_pll_locked','\\startup_reset/released']
    assert all(x in ports for x in release)
    # WAIT_CMD actively polls the mailbox. Sampling IDLE at an arbitrary
    # timestamp rejects legal DECODE/MAIL_READ/RESPONSE phases. Require the
    # complete live polling cycle, a clear fault and bounded progress instead.
    lines.append(r'''
task check_wait_loop;
 integer replies;reg [9:0] seen;
 begin
  replies=0;seen=0;
  repeat(256)begin
   @(negedge DUT_CLOCK);
   if(dut.\cpu/sequencer/state !==6'd7 || dut.\cpu/fatal_bus_error !==1'b0)
    $fatal(1,"startup boot=%h fatal=%b state=%h",dut.\cpu/sequencer/state ,dut.\cpu/fatal_bus_error ,dut.\cpu/state_oh );
   case(dut.\cpu/state_oh )
    10'h001,10'h002,10'h004,10'h200:begin end
    default:$fatal(1,"unexpected wait-loop state %h",dut.\cpu/state_oh );
   endcase
   seen=seen|dut.\cpu/state_oh ;
   if(dut.\cpu/state_oh ==10'h200)replies=replies+1;
  end
  if(seen!==10'h207 || replies<32)$fatal(1,"polling made no progress seen=%h replies=%0d",seen,replies);
 end
endtask
'''.replace('DUT_CLOCK',clk))
    lines.append('initial begin')
    lines.append('#2000;'+''.join(ports[n]+'=1;' for n in release))
    for iteration in range(3):
        lines.extend(['#40000;check_wait_loop;',
          f'$display("MAPPED_STARTUP_RECOVERY_PASS round={iteration}");'])
        if iteration!=2:lines.append(f'{reset}=0;#2000;{reset}=1;')
    lines.extend(['$display("PASS GLOBAL_MAPPED_STARTUP resets=3");$finish;end',
                  'initial begin #150000;$fatal(1,"host-independent simulation limit");end','endmodule'])
    (root/'tb.sv').write_text('\n'.join(lines))
    models=[]
    for kind in ex['cells']:
        if kind=='assign':continue
        f=EF/'sim_models/verilog'/(kind.lower()+'.v');shutil.copy2(f,root/f.name);models.append(root/f.name)
    shutil.copy2(__file__,root/'runner.py')
    inputs=dict(original_map=str(mapped),original_map_sha256=sha(mapped),extractor_original_sha256=sha(old),
                files={f.name:sha(f) for f in root.iterdir() if f.is_file()},
                scope=ex['scope'],hardware_access=False)
    (root/'inputs.json').write_text(json.dumps(inputs,indent=2))
    env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat';start=time.monotonic()
    def run(args,name,budget):
        with (root/name).open('x') as log:
            child=subprocess.Popen([str(Q/args[0]),*map(str,args[1:])],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:result=child.wait(timeout=budget)
            except subprocess.TimeoutExpired:stop_launched_tree(child);raise
        text=(root/name).read_text(errors='replace')
        assert result==0 and not re.search(r'(?m)^#?\s*\*\* (Error|Fatal)',text),(name,text[-2500:])
        return text
    try:
        run(['vlib.exe','work'],'vlib.log',15)
        run(['vlog.exe','-sv','-timescale','1ns/1ps',*models,root/'mapped_control.v',root/'tb.sv'],'compile.log',40)
        text=run(['vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],'simulation.log',90)
        assert 'PASS GLOBAL_MAPPED_STARTUP resets=3' in text
        for name,h in inputs['files'].items():assert sha(root/name)==h
        assert sha(mapped)==inputs['original_map_sha256']
        result=dict(status='GLOBAL_MAPPED_STARTUP_THREE_RESETS_PASS',seconds=time.monotonic()-start,
                    inputs_sha256=sha(root/'inputs.json'),simulation_sha256=sha(root/'simulation.log'),
                    full_board=False,hardware_access=False,board_release=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as e:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
