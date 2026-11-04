"""Offline experiment: split only LOGIC COMB4s in an existing full netlist.

Fresh official map/VDB is mandatory. The exported Verilog is never used as
a substitute for the physical database. No placement or download is run.
"""
import argparse
import collections
import json
import re
import shutil
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT, HERE, EF, sha, run
from native_logic_split import split_logic_comb
from audit_snapshot_cells import cells

PARENT = ROOT / 'tmp/frame-execution/connected-board-startup-cause4'
NS = 'http://www.efinixinc.com/enf_proj'


def params(text):
    result = {}
    for name, key, value in re.findall(r'defparam\s+(\S+?)\s*\.([A-Z_0-9]+)\s*=\s*([^;]+);', text):
        name = name.lstrip('\\')
        value = value.strip()
        match = re.fullmatch(r"(\d+)'s?([hbd])([0-9a-fA-F]+)", value)
        if match:
            value = (int(match[1]), int(match[3], {'h':16,'b':2,'d':10}[match[2]]))
        result.setdefault(name, {})[key] = value
    return result


def lut_function(pins, mask):
    # Compare actual Boolean functions, allowing the mapper to omit unused
    # inputs. The official primitive's absent inputs have weak-zero defaults.
    ins = [pins.get('I'+str(i), "1'b0") for i in range(4)]
    variables = sorted(set(ins) - {"1'b0", "1'b1"})
    table = []
    for value in range(1 << len(variables)):
        values = {n:(value >> j)&1 for j,n in enumerate(variables)}
        values.update({"1'b0":0, "1'b1":1})
        table.append((mask >> sum(values[n] << j for j,n in enumerate(ins))) & 1)
    active = [j for j in range(len(variables)) if any(table[v] != table[v^(1<<j)] for v in range(len(table)))]
    reduced = [table[sum(((v>>i)&1)<<j for i,j in enumerate(active))] for v in range(1<<len(active))]
    return pins.get('O'), tuple(variables[j] for j in active), tuple(reduced)


def canonical(text):
    parsed, settings = cells(text), params(text)
    result = {}
    for name, (kind, pins, attrs) in parsed.items():
        attributes = settings.get(name, {})
        if kind == 'EFX_LUT4':
            assert 'LUTMASK' in attributes, name
            result[name] = (kind, lut_function(pins, attributes['LUTMASK'][1]))
        else:
            result[name] = (kind, pins, attributes)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    root.mkdir()
    source = PARENT/'outflow/ti60f225_oob.map.v'
    (root/'runner.py').write_bytes(Path(__file__).read_bytes())
    try:
        original = source.read_text()
        changed, replacement = split_logic_comb(original)
        assert len(replacement) == 110
        (root/'native_source.v').write_text(changed, encoding='utf8')
        (root/'replacements.json').write_text(json.dumps(replacement,indent=2))
        before = cells(original)
        expected = canonical(changed)
        old_other = {n:(k,p) for n,(k,p,_) in before.items() if k != 'EFX_COMB4'}
        new_other = {n:(k,p) for n,(k,p,_) in cells(changed).items() if n in old_other}
        assert old_other == new_other
        # Preserve the vendor XML element order and encoding declaration.
        # Appending design_file after partition_definitions is rejected by
        # the tool's schema even though it is well-formed generic XML.
        xml = (PARENT/'ti60f225_oob.xml').read_text()
        file_count = 0
        def replace_file(match):
            nonlocal file_count
            file_count += 1
            return '<efx:design_file name="native_source.v" version="default" library="default" />' if file_count == 1 else ''
        xml = re.sub(r'<efx:design_file\b[^>]*/>',replace_file,xml)
        assert file_count > 1
        pattern = r'(<efx:param name="pack-luts-to-comb4" value=")[^"]+("[^>]*/>)'
        xml, count = re.subn(pattern,r'\g<1>0\g<2>',xml)
        if not count:
            xml = xml.replace('</efx:synthesis>','<efx:param name="pack-luts-to-comb4" value="0" value_type="e_option" /></efx:synthesis>')
        parsed = ET.fromstring(xml)
        assert len(parsed.findall('.//{'+NS+'}design_file')) == 1
        (root/'ti60f225_oob.xml').write_text(xml,encoding='utf8')
        for name in ('ti60f225_oob.sdc','ti60f225_oob.peri.xml'):
            shutil.copy2(PARENT/name, root/name)
        for path in PARENT.glob('*.bin'):
            shutil.copy2(path, root/path.name)
        (root/'map.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat ti60f225_oob.xml --flow map --timeout 450\nexit /b %errorlevel%\n')
        meta = dict(parent=str(PARENT),parent_map_sha256=sha(source),runner_sha256=sha(__file__),
                    transform_sha256=sha(HERE/'native_logic_split.py'),
                    files={f.name:sha(f) for f in root.iterdir() if f.is_file()},
                    replacements=110,hardware_access=False,physical_pass=False)
        (root/'inputs.json').write_text(json.dumps(meta,indent=2))
        body, seconds = run(['cmd.exe','/d','/c','map.cmd'], root,'map.log',timeout=480)
        assert re.search(r'map\s*:\s*PASS',body)
        net = root/'outflow/ti60f225_oob.map.v'
        mapped = net.read_text()
        actual = canonical(mapped)
        missing = sorted(expected.keys()-actual.keys())
        added = sorted(actual.keys()-expected.keys())
        different = sorted(n for n in expected.keys() & actual.keys() if expected[n] != actual[n])
        audit = dict(expected=len(expected),actual=len(actual),missing=missing,added=added,different=different,
                     examples={n:dict(expected=expected[n],actual=actual[n]) for n in different[:16]})
        (root/'structural_comparison.json').write_text(json.dumps(audit,indent=2))
        assert not missing and not added and not different, 'Fresh map changed more than the approved primitive split'
        assert sha(source)==meta['parent_map_sha256']
        assert (root/'outflow/ti60f225_oob.vdb').stat().st_size>0
        counts=collections.Counter(k[0] for k in actual.values())
        assert counts.get('EFX_COMB4',0)==0
        record=dict(status='FULL_NATIVE_SPLIT_FRESH_MAP_STRUCTURAL_PASS',seconds=seconds,counts=dict(counts),
                    map_sha256=sha(net),vdb_sha256=sha(root/'outflow/ti60f225_oob.vdb'),
                    inputs_sha256=sha(root/'inputs.json'),physical_pass=False,hardware_access=False)
        (root/'result.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc(),hardware_access=False),indent=2))
        raise


if __name__=='__main__':
    main()
