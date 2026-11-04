"""Same frozen map, complete-tile placement reuse, bounded 100-iteration P&R."""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from control_ddr_combined import once

def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cnn-ram-reference',type=Path,required=True);a,rest=p.parse_known_args()
    root=a.root.resolve();assert not root.exists() and root.is_relative_to(ROOT/'tmp')
    original=HERE/'probe_fourpll_reset_physical.py';code=original.read_text()
    code=once(code,'from fourpll_reset_clock_profile import apply_profile','from control_ddr_clock_profile import apply_profile')
    code=once(code,'related_gray_layout=None\n',
        "from fixed_control_ddr_reuse_locations import generate as reuse_locations\n"
        "assert a.fixed_dsp_layout and not a.related_gray_placement_reference\n"
        "reuse=reuse_locations(root)\ndsp_layout['assignments_sha256']=sha(root/'fixed_dsp.placeloc')\n"
        "dsp_layout['reuse_placement_sha256']=sha(root/'reuse_placement.json')\nrelated_gray_layout=None\n")
    code=once(code,"  capacity_file=root/'outflow/ti60f225_oob.place.rpt'", """  if flow=='pnr':
   direct=[str(EF/'bin/efx_pnr.exe'),'--circuit','ti60f225_oob','--family','Titanium',
    '--device','Ti60F225','--operating_conditions','I3','--vdb_file','outflow/ti60f225_oob.vdb',
    '--use_vdb_file','on','--prj',str(root/'ti60f225_oob.xml'),'--output_dir','outflow',
    '--work_dir','work_pnr','--place_file','outflow/ti60f225_oob.place',
    '--route_file','outflow/ti60f225_oob.route','--settings_file','efx_pnr_settings.ini',
    '--sync_file','outflow/ti60f225_oob.interface.csv','--generate_prevpr_netlist','off',
    '--sdc_file=ti60f225_oob.sdc','--max_router_iterations','100','--troute_options=max_iterations=100']
   script.write_text('@echo off\\ncall "'+str(EF/'bin/setup.bat')+'"\\nif errorlevel 1 exit /b 1\\n'+subprocess.list2cmdline(direct)+'\\nexit /b %errorlevel%\\n')
  capacity_file=root/'outflow/ti60f225_oob.place.rpt'""")
    driver=root.parent/(root.name+'-reuse-probe.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code)
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(original),source_sha256=sha(original),
        generated_sha256=sha(driver),wrapper_sha256=sha(__file__),placement_generator_sha256=sha(HERE/'fixed_control_ddr_reuse_locations.py'),
        changes='Placement assignments and bounded routing iterations only; no RTL/map/SDC changes.',hardware_access=False),indent=2))
    sys.argv=[str(driver),'--root',str(root),*rest];runpy.run_path(str(driver),run_name='__main__')
    result=json.loads((root/'result.json').read_text())
    assert all(row['exit_code']==0 for row in result['stages']),result
    from fixed_control_ddr_reuse_locations import verify_placed
    (root/'reuse_actual_locations.json').write_text(json.dumps(verify_placed(root),indent=2))

if __name__=='__main__':main()
