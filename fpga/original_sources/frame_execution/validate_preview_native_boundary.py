"""Check each exported native state bit against the ORIGINAL mapped controller.

The new behavior has separate unit tests. This gate proves the old native
connections' meaning, including optimized polarity and ROM aliases, before
reusing their shared address/data decode. No device access.
"""
import argparse,json,os,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from native_local_drivers import CELL,graph,render_token
from camera_joint_native import OUT as OUTPUT
from continuous_control_native import DEST,PARENT,connections,net
from control_command_repair import once

Q=Path('E:/quatra/questa_fse/win64')

def extract(text,rows):
    g=graph(text);drivers,loads=connections(g)
    cuts={net('process_clk_150'):'clk',net('ceg_net4871'):'!reset',net('w_arstn'):'!reset',net('cpu/boot_ready'):'cmd_ready',
          net('n78042_2'):'rsp_valid',net('cpu/reply_error'):'rsp_error',net('cpu/fatal_bus_error'):"1'b0"}
    cuts.update({net('cpu/reply_data',i):f'rsp_data[{i}]' for i in range(32)})
    own={n for n,(k,p) in g.items() if k in ('EFX_FF','EFX_RAM10') and
         (n.startswith('cpu/sequencer/') or n.startswith(('cpu/boot_observer_state[','cpu/boot_fault[','cpu/permit~','cpu/boot_terminal~','cpu/boot_failed~')))}
    own.add('cpu/first_fault/previous_flags[16]~FF')
    chosen=set(own)|{drivers[net('cpu/boot_valid')]};todo=list(chosen);unknown=set()
    while todo:
        n=todo.pop();k,p=g[n]
        for port,ts in p.items():
            if port in OUTPUT[k]:continue
            for t in ts:
                if t[0]=='literal' or t in cuts:continue
                if t==net('cpu/boot_observer_state'):
                    for i in range(6):
                        d=drivers[net('cpu/boot_observer_state',i)]
                        if d not in chosen:chosen.add(d);todo.append(d)
                    continue
                d=drivers.get(t)
                if d is None or (g[d][0] not in ('EFX_LUT4','EFX_ADD') and d not in own):unknown.add(t);continue
                if d not in chosen:chosen.add(d);todo.append(d)
    assert not unknown,[render_token(t) for t in sorted(unknown,key=str)]
    cells='\n'.join(m[0] for m in CELL.finditer(text) if m[2].lstrip('\\') in chosen)
    settings='\n'.join(m[0] for m in re.finditer(r'^\s*defparam\s+(\S+?)\s*\.\w+\s*=[^;]*;',text,re.M)
                       if m[1].lstrip('\\') in chosen)
    decl='\n'.join(re.findall(r'^\s*wire\b[^;]*;',text,re.M))
    # Original whole-map wire declarations do not include the test ports.
    ports=f'input wire clk,reset,cmd_ready,rsp_valid,rsp_error,input wire [31:0] rsp_data,output wire cmd_valid,output wire [{len(rows)-1}:0] native_state'
    assigns='\n'.join(f'assign {render_token(t)} = {expr};' for t,expr in cuts.items())
    assigns+='\nassign cmd_valid=\\cpu/boot_valid ;\nassign native_state={'
    assigns+=', '.join(render_token(tuple(x['token'])) for x in reversed(rows))+'};\n'
    return f'module mapped_reference({ports});\n'+decl+'\n'+assigns+cells+'\n'+settings+'\nendmodule\n',chosen

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    src=PARENT/'outflow/ti60f225_oob.map.v';rows=json.loads((DEST/'boundary.json').read_text())['rows']
    try:
        mapped,chosen=extract(src.read_text(),rows)
        (root/'mapped_reference.sv').write_text(mapped)
        shutil.copy2(DEST/'original_exposed.sv',root/'original_exposed.sv')
        t=(HERE/'variants/fixed_boot_sequencer/tb_cnn_boot_sequencer.sv').read_text()
        t=once(t,'cnn_boot_sequencer #(.RUN_TIMEOUT_CYCLES(1024),.BUS_TIMEOUT_CYCLES(64)) dut(.*);',
          f'wire [5:0] observer_state;wire [9:0] frame_budget;wire [{len(rows)-1}:0] native_state,mapped_state;wire mapped_valid;\n'
          ' original_exposed dut(.*);\n'
          ' mapped_reference mapped(.clk(clk),.reset(reset),.cmd_ready(cmd_ready),.rsp_valid(rsp_valid),.rsp_error(rsp_error),.rsp_data(rsp_data),.cmd_valid(mapped_valid),.native_state(mapped_state));\n'
          ' integer comparisons=0;integer startup_reset_clocks=0;reg armed=0;\n'
          ' always @(posedge clk)begin #2;if(reset)startup_reset_clocks=startup_reset_clocks+1;\n'
          ' if(startup_reset_clocks>=3)armed=1;\n'
          ' if(armed)begin\n'
          '  if(cmd_valid!==mapped_valid)$fatal(1,"MAPPED_VALID mismatch mode=%0d state=%0d rtl=%b map=%b",mode,dut.state,cmd_valid,mapped_valid);\n'
          f'  if(native_state!==mapped_state)for(integer bitno=0;bitno<{len(rows)};bitno=bitno+1)\n'
          '   if(native_state[bitno]!==mapped_state[bitno])$fatal(1,"MAPPED_BOUNDARY bit=%0d mode=%0d state=%0d rtl=%b map=%b",bitno,mode,dut.state,native_state[bitno],mapped_state[bitno]);\n'
          '  comparisons=comparisons+1;end end')
        # Boundary semantic check: one complete 2560-word load and readback,
        # success, plus three invalid mailbox inputs reaching fault/HALT.
        # All 17 new behavior cases remain in continuous-control-short2.
        # Do not repeat every complete image in this native primitive test.
        t=once(t,'for(mode=0;mode<16;mode=mode+1)begin','for(mode=0;mode<4;mode=mode+1)begin')
        t=once(t,'  $finish;',f'  $display("PASS MAPPED_PREVIEW_BOUNDARY bits={len(rows)} comparisons=%0d",comparisons);\n  $finish;')
        (root/'tb.sv').write_text(t)
        kinds=['EFX_FF','EFX_LUT4','EFX_ADD','EFX_RAM10']
        for k in kinds:shutil.copy2(EF/'sim_models/verilog'/('efx_'+k[4:].lower()+'.v'),root/(k.lower()+'.v'))
        inputs={str(f):sha(f) for f in [Path(__file__),src,DEST/'boundary.json',DEST/'original_exposed.sv',*root.glob('*.sv'),*root.glob('*.v')]}
        (root/'inputs.json').write_text(json.dumps(inputs,indent=2))
        env=dict(os.environ);env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
        cmds=[('vlib',[Q/'vlib.exe','work']),('compile',[Q/'vlog.exe','-sv','-timescale','1ns/1ps',*root.glob('*.v'),*root.glob('*.sv')]),
              ('simulate',[Q/'vsim.exe','-c','work.tb_cnn_boot_sequencer','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'])]
        timings=[]
        for name,cmd in cmds:
            start=time.monotonic()
            with (root/(name+'.log')).open('w') as log:
                proc=subprocess.run(list(map(str,cmd)),cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            body=(root/(name+'.log')).read_text()
            assert proc.returncode==0 and not re.search(r'\*\* (Error|Fatal)|^# FAIL',body,re.M),(name,body[-2500:])
            timings.append(dict(stage=name,seconds=time.monotonic()-start))
        body=(root/'simulate.log').read_text();assert 'PASS MAPPED_PREVIEW_BOUNDARY' in body
        for f,h in inputs.items():assert sha(f)==h
        result=dict(status='ORIGINAL_MAPPED_CONTROLLER_EXPORT_MEANINGS_PASS',bits=len(rows),cells=len(chosen),
                    markers=re.findall(r'^# (PASS.*)$',body,re.M),timings=timings,inputs_sha256=sha(root/'inputs.json'),
                    hardware_access=False,new_behavior_pass=False,board_release=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as e:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise

if __name__=='__main__':main()
