"""Extend the exact mapped startup cone to include both real CPU adapter FSMs.

No hardware; protected DDR reset output is an explicit four-cycle release
boundary, not a claim that the physical DDR reset has been measured.
"""
import argparse, json, runpy, sys
from pathlib import Path
from run_units import ROOT,HERE,sha

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',default='complete-transition-adapter-idle1')
    parser.add_argument('--board',type=Path,default=ROOT/'tmp/frame-execution/connected-board-complete-transition2')
    parser.add_argument('--modelsim',action='store_true')
    args=parser.parse_args()
    assert args.tag.replace('-','').isalnum()
    board=args.board.resolve()
    root=ROOT/'tmp/frame-execution'/args.tag
    source=HERE/'validate_global_mapped_startup.py'
    code=source.read_text()
    marker="exec(compile(code,str(old),'exec')"
    assert code.count(marker)==1
    addition="""code=code.replace('len(selected)<4500','len(selected)<6500')
    code=code.replace('def owned(n):', 'def owned(n):\\n    if n.startswith('+repr('\\\\video_processing_pipeline/pipeline/fabric/cpu_width_adapter/')+'):return True')
    """
    code=code.replace(marker,addition+marker)
    ofp4='fault_probe_transform_sha256' in json.loads((board/'inputs.json').read_text())
    if ofp4:
        code=code.replace('len(selected)<6500','len(selected)<9000')
        code=code.replace("    exec(compile(code,str(old),'exec')",
            "    code=code.replace('def owned(n):', \"def owned(n):\\n    if n in drivers['adapter_diagnostic'] or n in drivers['s_axi_rdata']:return True\")\n    exec(compile(code,str(old),'exec')")
    code=code.replace(r'\cpu/sequencer/state ',r'\cpu/boot_observer_state ')
    adapter=r'dut.\video_processing_pipeline/pipeline/fabric/cpu_width_adapter/'
    target='   seen=seen|dut.\\cpu/state_oh ;'
    assert code.count(target)==1
    checks='''   if(ADAPTERrstate !==3'd0 || ADAPTERwstate !==3'd0 ||
      ADAPTERrerror !==1'b0 || ADAPTERwerror !==1'b0)
    $fatal(1,"adapter idle rs=%h ws=%h re=%b we=%b",ADAPTERrstate ,ADAPTERwstate ,ADAPTERrerror ,ADAPTERwerror );
'''.replace('ADAPTER',adapter)
    code=code.replace(target,checks+target)
    if ofp4:
        for old,new in (('rstate','adapter_diagnostic[98:96]'),('wstate','adapter_diagnostic[101:99]'),
                        ('rerror','adapter_diagnostic[102]'),('werror','adapter_diagnostic[103]')):
            code=code.replace(adapter+old+' ','dut.'+new+' ')
    # Deliberately drive the protected reset boundary as documented, keeping
    # its assertion/release synchronized to the actual reset stimulus.
    marker="    lines.append('initial begin')"
    assert code.count(marker)==1
    addition="""    ddr_reset=ports['\\\\u_ddr3_top/controller/u_efx_ddr3_soft_controller/genblk1.inst_ddr3_axi/inst_efx_ddr3/top_mc/clk_rstn']
    assert not any('cpu_width_adapter/' in n for n in ports), 'Adapter state still cut'
    lines.append(f'reg [3:0] protected_reset_model=0; always @(posedge {clk} or negedge {reset}) if(!{reset}) protected_reset_model<=0; else protected_reset_model<={{protected_reset_model[2:0],1\\'b1}}; always @* {ddr_reset}=protected_reset_model[3];')
"""
    code=code.replace(marker,addition+marker)
    code=code.replace("status='GLOBAL_MAPPED_STARTUP_THREE_RESETS_PASS'",
                      "status='ACTUAL_MAPPED_CONTROL_ADAPTER_IDLE_THREE_RESETS_PASS'")
    # Starter's 5000-instance message is a recommended-capacity warning, not
    # a functional error. Keep it visible, as existing project runners do.
    if args.modelsim:
        code=code.replace("Q=Path('E:/quatra/questa_fse/win64')", "Q=Path('E:/altera/15.0/modelsim_ase/win32aloem')")
        code=code.replace("shutil.copy2(f,root/f.name);models.append(root/f.name)",
                          "assert 'pragma protect' not in f.read_text(errors='replace');shutil.copy2(f,root/f.name);models.append(root/f.name)")
    else:
        code=code.replace("['vsim.exe','-c','work.tb'",
                          "['vsim.exe','-c','-warning','14408,16154','-voptargs=-warning 14408','work.tb'")
        code=code.replace("'simulation.log',90)","'simulation.log',180)")
    driver=root.parent/(root.name+'-runner.py')
    assert not driver.exists() and not root.exists()
    # Syntax check before any simulation or evidence directory is launched.
    compile(code,str(driver),'exec')
    driver.write_text(code,encoding='utf-8')
    driver.with_suffix('.wrapper.py').write_bytes(Path(__file__).read_bytes())
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        generated_sha256=sha(driver),wrapper_sha256=sha(__file__),hardware_access=False),indent=2))
    sys.argv=[str(driver),'--board',str(board),'--root',str(root)]
    runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
