"""Read-only APB diagnostics plus real coefficient-store regression; no board I/O."""
import argparse,json,os,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha,run
MS=Path('E:/quatra/questa_fse/win64')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    root.mkdir(parents=True)
    os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
    variant=HERE/'variants/board_bringup_status'
    base=ROOT/'tmp/frame-execution/connected-board-skip-credit1'
    vectors=ROOT/'model-training-workspace/accepted/B1-A-20261008'
    words=(vectors/'load_words.hex').read_text().splitlines()
    assert len(words)==2560
    assert b''.join(int(w,16).to_bytes(8,'little') for w in words)==(vectors/'coefficients-padded.bin').read_bytes()
    paths=[EF/'sim_models/verilog/efx_ram10.v',ROOT/'tmp/frame-execution/program-map2/resident_weight_bank64.sv',
           ROOT/'tmp/frame-execution/program-map2/coefficient_bank_store_fast.sv',
           variant/'cnn_apb_control.sv',variant/'tb_cnn_apb_control.sv']
    records=[]
    for src in paths:
        dst=root/src.name;shutil.copy2(src,dst)
        records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
    # This test exercises exactly the current frozen APB, with only two new
    # read-only addresses and their input declarations. Reject other changes.
    old=(base/'assembly/cnn_apb_control.sv').read_text()
    actual=(variant/'cnn_apb_control.sv').read_text()
    stripped=actual.replace(' input wire [31:0] bringup_pipeline_status,bringup_ddr_status,\n','')
    stripped=stripped.replace('   // Read-only observations; no control-path feedback.\n'
        "   16'h000c:prdata=bringup_pipeline_status;\n   16'h006c:prdata=bringup_ddr_status;\n",'')
    assert stripped.strip()==old.strip(),'Unexpected functional APB change'
    # The CPU decoder already permits these offsets; no bus address expansion.
    soc=(base/'assembly/EfxSapphireSoc.v').read_text()
    assert "address[6:0]<=7'h6c" in soc
    inputs=dict(inputs=records,runner_sha256=sha(__file__),vectors_sha256=sha(vectors/'load_words.hex'),base_apb_sha256=sha(base/'assembly/cnn_apb_control.sv'),
                variants={f.name:sha(f) for f in variant.glob('*') if f.is_file()})
    (root/'inputs.json').write_text(json.dumps(inputs,indent=2))
    try:
        run([MS/'vlib.exe','work'],root,'vlib.log')
        body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log')
        assert 'Errors: 0, Warnings: 0' in body
        body,seconds=run([MS/'vsim.exe','-c','work.tb_cnn_apb_control',
            '+VECTORS='+vectors.as_posix(),
            '-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=120)
        assert re.search(r'^# PASS CNN_APB_CONTROL',body,re.M) and 'Errors: 0, Warnings: 0' in body
        for r in records:assert sha(r['source'])==sha(r['snapshot'])==r['sha256']
        result=dict(status='BRINGUP_STATUS_APB_FOURSTATE_PASS',seconds=seconds,
          status_bit_patterns=64,rejected_diagnostic_writes=64,actual_coefficient_writes=2560,
          actual_coefficient_readbacks=2560,existing_apb_behavior_unchanged=True,
          inputs_sha256=sha(root/'inputs.json'),simulation_sha256=sha(root/'simulation.log'),
          physical_gate=False,board_test=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise

if __name__=='__main__':main()
