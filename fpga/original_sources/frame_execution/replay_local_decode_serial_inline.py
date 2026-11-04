"""Same serial test, native Questa; public LUT bodies inline, no reduced checks."""
import argparse,json,re,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from inline_official_state_luts import inline

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    source=a.source.resolve();root=a.root.resolve();assert not root.exists()
    model=source/'efx_lut4.v';original=source/'mapped_control.v'
    changed,proof=inline(original.read_text(),model,lambda n:n.startswith('\\cpu/ctrl_decode/'))
    # Reuse the exact replay gate, changing only the simulator and mapped
    # simulation file immediately after copying; all stimulus files match.
    base=HERE/'replay_mapped_serial_modelsim.py';code=base.read_text()
    code=code.replace('from run_units import ROOT,MS,sha','from run_units import ROOT,sha\nMS=Path(\'E:/quatra/questa_fse/win64\')')
    marker='    meta=dict(inputs,source_replay='
    assert code.count(marker)==1
    inject=("    original_files=dict(inputs['files'])\n"
      "    from inline_official_state_luts import inline\n"
      "    content,inline_proof=inline((root/'mapped_control.v').read_text(),root/'efx_lut4.v',lambda n:n.startswith('\\\\cpu/ctrl_decode/'))\n"
      "    (root/'mapped_control.v').write_text(content)\n"
      "    inputs['files']['mapped_control.v']=sha(root/'mapped_control.v')\n"
      "    inputs['simulation_only_inline']=dict(inline_proof,original_files=original_files,transform_sha256=sha(HERE/'inline_official_state_luts.py'))\n"
      "    import os\n    os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'\n")
    code=code.replace('from run_units import ROOT,sha','from run_units import ROOT,HERE,sha')
    code=code.replace(marker,inject+marker)
    code=code.replace('identical_stimulus_and_unencrypted_official_models=True','identical_stimulus_and_unencrypted_official_models=True,official_models_textually_inlined=23')
    driver=root.parent/(root.name+'-driver.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code)
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(base),source_sha256=sha(base),
        driver_sha256=sha(driver),adapter_sha256=sha(__file__),inline_proof=proof),indent=2))
    sys.argv=[str(driver),'--source',str(source),'--root',str(root)];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
