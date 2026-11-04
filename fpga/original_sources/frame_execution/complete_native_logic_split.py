"""Reimport the remaining 18 vendor-exported LOGIC pairs, not protected RTL.

The original encrypted block is copied byte-for-byte. No FPGA clock, pin,
timing constraint or PHY state machine is edited.
"""
import argparse
import collections
import hashlib
import json
import re
import shutil
import traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha,run
from native_logic_split import split_logic_comb
from audit_native_roundtrip import verify

PARENT=ROOT/'tmp/frame-execution/full-native-split-map3'


def resource_counts(path):
    return {n:int(v) for n,v in re.findall(r'^(EFX_\w+)\s*:\s*(\d+)\s*$',path.read_text(),re.M)}


def protected_blocks(text):
    return re.findall(r'`pragma protect begin_protected.*?`pragma protect end_protected',text,re.S)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    try:
        for file in (__file__,HERE/'native_logic_split.py',HERE/'audit_native_roundtrip.py'):
            shutil.copy2(file,root/Path(file).name)
        assert json.loads((PARENT/'public_connectivity_audit.json').read_text())['status']=='PUBLIC_PRIMITIVE_RENAME_EQUIVALENCE_PASS'
        source=PARENT/'outflow/ti60f225_oob.map.v';original=source.read_text()
        changed,rows=split_logic_comb(original)
        assert len(rows)==18
        blocks=protected_blocks(original)
        assert blocks and protected_blocks(changed)==blocks
        (root/'native_source.v').write_text(changed,encoding='utf8')
        (root/'replacements.json').write_text(json.dumps(rows,indent=2))
        all_rows=json.loads((PARENT/'replacements.json').read_text())+rows
        assert len(all_rows)==128
        (root/'all_replacements.json').write_text(json.dumps(all_rows,indent=2))
        for name in ('ti60f225_oob.xml','ti60f225_oob.sdc','ti60f225_oob.peri.xml','map.cmd'):
            shutil.copy2(PARENT/name,root/name)
        for path in PARENT.glob('*.bin'):shutil.copy2(path,root/path.name)
        meta=dict(parent=str(PARENT),parent_map_sha256=sha(source),parent_audit_sha256=sha(PARENT/'public_connectivity_audit.json'),
                  files={f.name:sha(f) for f in root.iterdir() if f.is_file()},replacement_count=18,
                  encrypted_blocks_sha256=[hashlib.sha256(b.encode()).hexdigest() for b in blocks],hardware_access=False)
        (root/'inputs.json').write_text(json.dumps(meta,indent=2))
        body,seconds=run(['cmd.exe','/d','/c','map.cmd'],root,'map.log',timeout=480)
        assert re.search(r'map\s*:\s*PASS',body)
        net=root/'outflow/ti60f225_oob.map.v'
        audit=verify(changed,net.read_text())
        (root/'public_connectivity_audit.json').write_text(json.dumps(audit,indent=2))
        assert audit['added_protected']==0, 'Unexpected additional protected expansion'
        old=resource_counts(PARENT/'outflow/ti60f225_oob.map.rpt')
        new=resource_counts(root/'outflow/ti60f225_oob.map.rpt')
        expected=dict(old);expected['EFX_COMB4']=0;expected['EFX_LUT4']+=36
        assert {k:v for k,v in new.items() if v}=={k:v for k,v in expected.items() if v},(new,expected)
        assert sha(source)==meta['parent_map_sha256']
        assert (root/'outflow/ti60f225_oob.vdb').stat().st_size>0
        result=dict(status='FULL_NATIVE_ZERO_COMB_FRESH_MAP_PUBLIC_STRUCTURE_PASS',seconds=seconds,
                    counts=new,map_sha256=sha(net),vdb_sha256=sha(root/'outflow/ti60f225_oob.vdb'),
                    inputs_sha256=sha(root/'inputs.json'),public_audit_sha256=sha(root/'public_connectivity_audit.json'),
                    local_split_total=128,physical_pass=False,hardware_access=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise


if __name__=='__main__':main()
