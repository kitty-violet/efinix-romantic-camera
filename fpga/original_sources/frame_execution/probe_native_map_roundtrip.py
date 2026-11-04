"""Small offline feasibility check: exported primitives -> fresh map/VDB.

No source or physical project is overwritten. This is not a board release.
"""
import argparse
import json
import re
import shutil
from pathlib import Path
from run_units import ROOT, HERE, EF, sha, run

TOP = 'native_roundtrip'
RUN = ROOT / 'tmp/frame-execution/native-map-roundtrip1'


def split_logic_comb(text):
    pattern = re.compile(r'^\s*EFX_COMB4\s+(\S+)\s*\((.*?)\) /\*[^\n]*\*/\s*;[^\n]*', re.M | re.S)
    rows = []
    def replace(match):
        name, ports = match.groups()
        con = dict(re.findall(r'\.(\w+)\(([^()]*)\)', ports))
        definitions = re.findall(r'defparam\s+' + re.escape(name) + r'\s*\.([A-Z_]+)\s*=\s*([^;]+);', text)
        params = dict(definitions)
        assert params['MODE'].strip() == '"LOGIC"', (name, params)
        assert not any(con.get(p, '').strip() for p in ('CO', 'FCO')), (name, con)
        m = re.fullmatch(r"16'([hb])([0-9a-fA-F]+)", params['LUTMASK'].strip())
        assert m, params
        mask = int(m[2], 16 if m[1] == 'h' else 2)
        pmask = ((mask >> 8) & 15) * 0x1111
        for value in range(16):
            assert (pmask >> value) & 1 == (mask >> (8 + (value & 3))) & 1
        generated = []
        for pin, lm in (('O', mask), ('P', pmask)):
            if not con.get(pin, '').strip():
                continue
            instance = name if pin == 'O' else ('\\' + name.lstrip('\\') + '_splitP')
            ins = ', '.join('.' + p + '(' + con.get(p, "1'b0") + ')' for p in ('I0', 'I1', 'I2', 'I3'))
            generated.append(f"    EFX_LUT4 #(.LUTMASK(16'h{lm:04x})) {instance} ({ins}, .O({con[pin]}));")
        rows.append(dict(name=name, mask=mask, p_mask=pmask, ports=con))
        return '\n' + '\n'.join(generated)
    output = pattern.sub(replace, text)
    for row in rows:
        for parameter in ('LUTMASK', 'MODE'):
            pattern = r'^\s*defparam\s+' + re.escape(row['name']) + r'\s*\.' + parameter + r'\s*=[^;]+;'
            output, count = re.subn(pattern, '', output, flags=re.M)
            assert count == 1, (row['name'], parameter, count)
    assert not re.search(r'^\s*EFX_COMB4\s+\S+\s*\(', output, re.M)
    return output, rows


def map_stage(name, source):
    root = RUN / name
    root.mkdir()
    (root / (TOP + '.sv')).write_text(source, encoding='utf8')
    xml = (ROOT / 'tmp/frame-execution/meter-map1/frame_execution_meter.xml').read_text().replace('frame_execution_meter', TOP)
    xml = xml.replace('</efx:synthesis>', '<efx:param name="pack-luts-to-comb4" value="0" value_type="e_option" /></efx:synthesis>')
    (root / (TOP + '.xml')).write_text(xml)
    (root / (TOP + '.sdc')).write_text('create_clock -name clk -period 6.666 [get_ports {clk}]\n')
    (root / 'run.cmd').write_text('@echo off\ncall "' + str(EF / 'bin/setup.bat') + '"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat ' + TOP + '.xml --flow map --timeout 90\nexit /b %errorlevel%\n')
    body, seconds = run(['cmd.exe', '/d', '/c', 'run.cmd'], root, 'map.log', timeout=110)
    assert re.search(r'map\s*:\s*PASS', body)
    net = root / 'outflow' / (TOP + '.map.v')
    rpt = root / 'outflow' / (TOP + '.map.rpt')
    counts = {k:int(v) for k,v in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)', rpt.read_text(), re.M)}
    vdb = root / 'outflow' / (TOP + '.vdb')
    assert vdb.stat().st_size > 0
    return dict(root=str(root), seconds=seconds, counts=counts, map_sha256=sha(net), vdb_sha256=sha(vdb)), net.read_text()


def main():
    global RUN
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=RUN)
    args = parser.parse_args()
    RUN = args.root.resolve()
    assert RUN.is_relative_to(ROOT / 'tmp')
    assert not RUN.exists()
    RUN.mkdir()
    (RUN / 'runner.py').write_bytes(Path(__file__).read_bytes())
    source = '''module native_roundtrip(input clk, ce, rst, input [3:0] data, output [1:0] result);
 wire o, p;
 EFX_COMB4 #(.LUTMASK(16'h1144),.MODE("LOGIC")) pair(.I0(data[0]),.I1(data[1]),.I2(data[2]),.I3(data[3]),.O(o),.P(p));
 EFX_FF #(.SR_SYNC(1),.SR_VALUE(0)) q_o(.D(o),.CLK(clk),.CE(ce),.SR(rst),.Q(result[0]));
 EFX_FF #(.SR_SYNC(1),.SR_VALUE(0)) q_p(.D(p),.CLK(clk),.CE(ce),.SR(rst),.Q(result[1]));
 endmodule
'''
    stages = []
    try:
        first, mapped = map_stage('original', source)
        stages.append(first)
        replacement, rows = split_logic_comb(mapped)
        assert len(rows) == 1
        second, remapped = map_stage('split', replacement)
        stages.append(second)
        assert second['counts'].get('COMB4', 0) == 0
        assert second['counts'].get('LUT4') == 2 and second['counts'].get('FF') == 2
        from audit_snapshot_cells import cells
        before, after = cells(mapped), cells(remapped)
        assert {n:v[:2] for n,v in before.items() if v[0]=='EFX_FF'} == {n:v[:2] for n,v in after.items() if v[0]=='EFX_FF'}
        for name, cell in before.items():
            if cell[0] == 'EFX_FF':
                pattern = r'defparam\s+' + re.escape(name) + r'\s*\.([A-Z_]+)\s*=\s*([^;]+);'
                assert dict(re.findall(pattern, mapped)) == dict(re.findall(pattern, remapped))
        record = dict(status='SMALL_NATIVE_NETLIST_REIMPORT_AND_SPLIT_MAP_PASS', stages=stages,
                      replacements=rows, runner_sha256=sha(__file__),
                      full_design_verified=False, physical_pass=False, hardware_access=False)
        (RUN/'result.json').write_text(json.dumps(record,indent=2))
        print(json.dumps(record),flush=True)
    except Exception as exc:
        import traceback
        (RUN/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc(),stages=stages),indent=2))
        raise


if __name__=='__main__':
    main()
