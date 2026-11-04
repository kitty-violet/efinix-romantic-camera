"""Freeze a separate complete board; no synthesis or hardware side effects."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import BASE,soc,snapshot,audit_delta,audit_parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    audit_parent(BASE);root.mkdir()
    meta=json.loads((BASE/'inputs.json').read_text())
    for row in meta['inputs']:
        original=Path(row['snapshot']);assert original.is_relative_to(BASE) and sha(original)==row['sha256']
        rel=original.relative_to(BASE);dst=root/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(original,dst)
        if rel.as_posix()=='assembly/EfxSapphireSoc.v':dst.write_text(soc(original.read_text()),encoding='utf-8')
        if rel.as_posix()=='assembly/jtag_bringup_snapshot.sv':dst.write_text(snapshot(original.read_text()),encoding='utf-8')
        row.update(snapshot=str(dst),sha256=sha(dst))
    meta.update(parent=str(BASE),parent_inputs_sha256=sha(BASE/'inputs.json'),warning_reference_base=str(BASE),
                command_repair_sha256=sha(HERE/'control_command_repair.py'),runner_sha256=sha(__file__),
                hardware_access=False,board_download_allowed=False)
    (root/'runner.py').write_bytes(Path(__file__).read_bytes())
    (root/'inputs.json').write_text(json.dumps(meta,indent=2))
    (root/'delta_audit.json').write_text(json.dumps(audit_delta(root),indent=2))
    print('COMMAND_REPAIR_FULL_BOARD_FROZEN_NOT_RELEASED')
if __name__=='__main__':main()
