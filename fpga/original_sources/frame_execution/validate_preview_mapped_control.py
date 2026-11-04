"""Actual mapped new control: full load/readback, 512 budget and explicit rearm."""
import argparse,inspect,json,os,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from continuous_control_native import DEST,net,connections
from continuous_control_board import BOARD
from native_local_drivers import CELL,render_token
from audit_continuous_vectors import expand_graph
from camera_joint_native import OUT
from validate_continuous_run_control import boot_tb
from control_command_repair import once
from windows_process_tree import stop_launched_tree

def extract(text,rows):
    g=expand_graph(text);drivers,loads=connections(g)
    cuts={net('process_clk_150'):'clk',net('ceg_net4871'):'!reset',net('w_arstn'):'!reset',net('cpu/boot_ready'):'cmd_ready',
          net('n78042_2'):'rsp_valid',net('cpu/reply_error'):'rsp_error',net('cpu/fatal_bus_error'):"1'b0"}
    cuts.update({net('cpu/reply_data',i):f'rsp_data[{i}]' for i in range(32)})
    owned={n for n,(k,p) in g.items() if k in ('EFX_FF','EFX_RAM10') and
           n.startswith(('preview_boot/','preview_frame_budget[','cpu/sequencer/','cpu/boot_observer_state[','cpu/boot_fault[','cpu/permit~','cpu/boot_terminal~','cpu/boot_failed~'))}
    selected=set(owned)|{drivers[net('cpu/boot_valid')]};todo=list(selected);unknown=set()
    while todo:
        n=todo.pop();kind,pins=g[n]
        for port,ts in pins.items():
            if port in OUT[kind]:continue
            for t in ts:
                if t[0]=='literal' or t in cuts:continue
                d=drivers.get(t)
                if d is None or (g[d][0] not in ('EFX_LUT4','EFX_ADD') and d not in owned):unknown.add(t);continue
                if d not in selected:selected.add(d);todo.append(d)
    assert not unknown,[render_token(t) for t in sorted(unknown,key=str)]
    actual=[]
    for row in rows:
        t=tuple(row['token'])
        if t not in drivers:
            m=re.fullmatch(r'frames\[(\d+)\]',row['expression'])
            assert m,('unmapped native export',row)
            t=net('preview_frame_budget',int(m[1]))
        assert drivers[t] in selected,(row,t,drivers[t]);actual.append(t)
    # Export high frame-budget bits too; this detects accidental 5-bit truncation.
    actual += [net('preview_frame_budget',i) for i in range(10)]
    cells='\n'.join(m[0] for m in CELL.finditer(text) if m[2].lstrip('\\') in selected)
    settings='\n'.join(m[0] for m in re.finditer(r'^\s*defparam\s+(\S+?)\s*\.\w+\s*=[^;]*;',text,re.M) if m[1].lstrip('\\') in selected)
    declarations='\n'.join(re.findall(r'^\s*wire\b[^;]*;',text,re.M))
    pins=f'input wire clk,reset,cmd_ready,rsp_valid,rsp_error,input wire [31:0] rsp_data,output wire cmd_valid,output wire [{len(actual)-1}:0] native_state'
    assigns='\n'.join(f'assign {render_token(t)} = {expr};' for t,expr in cuts.items())
    assigns+='\nassign cmd_valid=\\cpu/boot_valid ;\nassign native_state={'+', '.join(render_token(t) for t in reversed(actual))+'};\n'
    return f'module mapped_preview({pins});\n'+declarations+'\n'+assigns+cells+'\n'+settings+'\nendmodule\n',selected,actual

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    src=BOARD/'outflow/ti60f225_oob.map.v';rows=json.loads((DEST/'boundary.json').read_text())['rows'];width=len(rows)+10
    try:
        mapped,selected,actual=extract(src.read_text(),rows);(root/'mapped_preview.sv').write_text(mapped)
        shutil.copy2(DEST/'cnn_preview_boot.sv',root/'cnn_preview_boot.sv')
        t=boot_tb()
        t=once(t,'cnn_boot_sequencer #(.RUN_TIMEOUT_CYCLES(8192),.BUS_TIMEOUT_CYCLES(64)) dut(.*);',
            f'wire [{len(rows)-1}:0] native_state;wire [9:0] frame_budget;wire [{width-1}:0] mapped_state;wire mapped_valid;\n'
            ' cnn_preview_boot dut(.*);\n'
            ' mapped_preview mapped(.clk(clk),.reset(reset),.cmd_ready(cmd_ready),.rsp_valid(rsp_valid),.rsp_error(rsp_error),.rsp_data(rsp_data),.cmd_valid(mapped_valid),.native_state(mapped_state));\n'
            ' integer checks=0,reset_clocks=0;reg armed=0;\n'
            ' always @(posedge clk)begin #2;if(reset)reset_clocks=reset_clocks+1;if(reset_clocks>=3)armed=1;\n'
            ' if(armed)begin\n'
            '  if(cmd_valid!==mapped_valid)$fatal(1,"PREVIEW_MAPPED valid state=%0d rtl=%b map=%b",dut.state,cmd_valid,mapped_valid);\n'
            '  if({frame_budget,native_state}!==mapped_state)$fatal(1,"PREVIEW_MAPPED state=%0d differing_mask=%h",dut.state,{frame_budget,native_state}^mapped_state);\n'
            '  checks=checks+1;end end')
        t=once(t,'for(mode=0;mode<17;mode=mode+1)begin','for(mode=16;mode<17;mode=mode+1)begin')
        t=once(t,'success_images=4','success_images=2')
        t=once(t,'  $finish;','  $display("PASS PREVIEW_ACTUAL_MAP comparisons=%0d explicit_rearm=1 frame_budget_512=1",checks);\n  $finish;')
        (root/'tb.sv').write_text(t)
        for kind in ('EFX_FF','EFX_LUT4','EFX_ADD','EFX_RAM10'):
            shutil.copy2(EF/'sim_models/verilog'/(kind.lower()+'.v'),root/(kind.lower()+'.v'))
        evidence={str(f):sha(f) for f in [Path(__file__),src,DEST/'boundary.json',DEST/'cnn_preview_boot.sv',HERE/'validate_continuous_run_control.py',*root.glob('*.sv'),*root.glob('*.v')]}
        (root/'inputs.json').write_text(json.dumps(dict(files=evidence,map_sha256=sha(src),selected_cells=sorted(selected),exports=actual),indent=2))
        q=Path('E:/quatra/questa_fse/win64');env=dict(os.environ);env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat';timings=[]
        for name,cmd in [('vlib',[q/'vlib.exe','work']),('compile',[q/'vlog.exe','-sv','-timescale','1ns/1ps',*root.glob('*.v'),*root.glob('*.sv')]),
                         ('simulate',[q/'vsim.exe','-c','work.tb_cnn_boot_sequencer','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'])]:
            begin=time.monotonic()
            with (root/(name+'.log')).open('w') as out:
                child=subprocess.Popen(list(map(str,cmd)),cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT)
                try:rc=child.wait(timeout=240)
                except subprocess.TimeoutExpired:(root/'timeout.json').write_text(json.dumps(stop_launched_tree(child),indent=2));raise
            text=(root/(name+'.log')).read_text()
            assert rc==0 and not re.search(r'\*\* (Error|Fatal)',text),(name,text[-2000:])
            timings.append(dict(stage=name,seconds=time.monotonic()-begin))
        text=(root/'simulate.log').read_text();assert 'PASS PREVIEW_ACTUAL_MAP' in text and 'PASS PREVIEW_REARM' in text
        for f,h in evidence.items():assert sha(f)==h,f
        r=dict(status='CONTINUOUS_ACTUAL_MAP_LOAD_VERIFY_REARM_PASS',map_sha256=sha(src),inputs_sha256=sha(root/'inputs.json'),
               log_sha256=sha(root/'simulate.log'),exported_bits=width,timings=timings,markers=re.findall(r'^# (PASS.*)$',text,re.M),
               hardware_access=False,board_release=False)
        (root/'result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
    except Exception as e:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
