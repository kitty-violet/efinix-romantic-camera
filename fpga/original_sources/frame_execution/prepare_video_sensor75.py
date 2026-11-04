"""Remove only vendor-confirmed, source-confirmed dangling public cells.

The sensor74 functional edit is unchanged. This is dead-code elimination,
not a CNN/data/control redesign, and does not grant physical release.
"""
import collections,json,pickle,re,shutil,sys,faulthandler
from pathlib import Path
from run_units import ROOT,HERE,sha
from native_local_drivers import CELL,REF,graph,lexical_tokens,render_token
from continuous_control_native import connections
from complete_native_logic_split import protected_blocks
from probe_native_open_ports import protected_pins
from audit_native_roundtrip import tokens
from audit_local_reset_implementation import norm

BASE=ROOT/'tmp/frame-execution/video-sensor74-minimal6'
PARENT=ROOT/'tmp/frame-execution/connected-board-video-service-probe60-input-isp-explicit'
MAP=ROOT/'tmp/frame-execution/connected-board-video-sensor74-map1'
OUT=ROOT/'tmp/frame-execution/video-sensor75-pruned1'

def main():
    faulthandler.dump_traceback_later(45,repeat=True)
    print('SENSOR75 source audit start',flush=True)
    assert not OUT.exists()
    proof=json.loads((BASE/'inputs.json').read_text());assert proof['source_sha256']==sha(BASE/'native_source.v')
    tested=ROOT/'tmp/frame-execution/video-sensor74-pin2'
    assert json.loads((tested/'result.json').read_text())['status']=='SENSOR74_MINIMAL_PIN_PASS_NOT_PHYSICAL_RELEASE'
    log=ROOT/'tmp/frame-execution/video-sensor74-place1/outflow/ti60f225_oob.place.out'
    names=set(re.findall(r'logical_block (.*?) #\d+ has no fanout',log.read_text()))
    assert len(names)==91
    source=(BASE/'native_source.v').read_text();g=graph(source);_,loads=connections(g)
    print('source graph parsed',len(g),flush=True)
    h,mg,mp=pickle.loads((MAP/'sensor74-map-parse.pkl').read_bytes());assert h==sha(MAP/'outflow/ti60f225_oob.map.v')
    _,ml=connections(mg)
    outputs={g[n][1]['O' if g[n][0]=='EFX_LUT4' else 'Q'][0] for n in names}
    assert len(outputs)==91
    assert collections.Counter(g[n][0] for n in names)=={'EFX_LUT4':84,'EFX_FF':7}
    for n in names:
        k,p=g[n];assert k in ('EFX_LUT4','EFX_FF')
        t=p['O' if k=='EFX_LUT4' else 'Q'][0]
        assert not loads[t] and not ml[t],('used_primitive_output',n,t)
    protected={t for expr in protected_pins(source).values() for t in tokens(expr)}
    assert not {norm(render_token(t)) for t in outputs}&protected
    removed=[]
    def drop(m):
        if m[2].lstrip('\\') in names:removed.append(m[0]);return ''
        return m[0]
    body=CELL.sub(drop,source);assert len(removed)==91
    print('91 disconnected instances removed in memory',flush=True)
    # Defparams are single-line in this generated native source. Avoid a
    # whole-file backtracking regex over the large embedded vendor text.
    kept=[];removed_parameters=[]
    for line in body.splitlines(keepends=True):
        stripped=line.lstrip()
        if stripped.startswith('defparam '):
            lhs=stripped.removeprefix('defparam ').split('=',1)[0].strip()
            name=lhs.rsplit('.',1)[0].strip().lstrip('\\')
            if name in names:
                assert stripped.rstrip().endswith(';')
                removed_parameters.append(line);continue
        kept.append(line)
    body=''.join(kept)
    assert removed_parameters
    ng=graph(body);assert set(g)-set(ng)==names and not set(ng)-set(g)
    print('remaining graph matched',len(ng),flush=True)
    assert all(g[n]==ng[n] for n in ng)
    assert protected_blocks(source)==protected_blocks(body)
    assert protected_pins(source)==protected_pins(body)
    for pattern in (r'^\s*(?:input|output|inout)\b.*$',r'^\s*assign\b.*$'):
        assert re.findall(pattern,source,re.M)==re.findall(pattern,body,re.M)
    # Also inspect ports of non-primitive source modules, which the primitive
    # graph alone cannot cover. No surviving instance may read a removed net.
    target_names={t[1] for t in outputs}
    print('nonprimitive ports audit',flush=True)
    for m in re.finditer(r'\.[\w\\/\[\].]+\s*\(([^()]*)\)',body):
        expr=m[1]
        if not any(n in expr for n in target_names):continue
        assert not set(lexical_tokens(expr))&outputs,('other_instance_load',expr)
    print('assignment references audit',flush=True)
    for m in re.finditer(r'^\s*assign\s+([^;]+);',body,re.M):
        expr=m[1].split('/*',1)[0]
        if not any(n in expr for n in target_names):continue
        # Assignments can contain bitwise operators, unlike instance ports.
        # Enumerate every identifier/range without treating operators as nets.
        refs={t for m in REF.finditer(expr.split('=',1)[-1]) for t in lexical_tokens(m[0])}
        assert not refs&outputs,('assignment_load',expr)
        assert not {t[1] for t in refs if t[0]=='net' and t[2] is None}&{t[1] for t in outputs if t[2] is not None},('unsliced_bus_load',expr)
    print('all deleted outputs have no consumer',flush=True)
    OUT.mkdir()
    (OUT/'native_source.v').write_text(body,encoding='utf8')
    for name in ('search.json','expected.hex','mapped_camera.v','camera_checked_config_reference.sv','tb.sv','efx_add.v','efx_ff.v','efx_lut4.v','efx_ram10.v'):
        shutil.copy2(BASE/name,OUT/name)
    short_cells=set(re.findall(r'^\s*EFX_\w+\s+\\(\S+)\s+\(', (OUT/'mapped_camera.v').read_text(),re.M))
    assert len(short_cells)==689 and not short_cells&names
    r=dict(status='SENSOR75_DANGLING_CELLS_SOURCE_PROVED_NOT_SYNTHESIZED',
        source_sha256=sha(OUT/'native_source.v'),script_sha256=sha(__file__),original_synthesis_input_sha256=sha(PARENT/'native_source.v'),
        sensor74_source=str(BASE),sensor74_source_sha256=sha(BASE/'native_source.v'),
        removed_cells=sorted(names),removed_kinds={'EFX_LUT4':84,'EFX_FF':7},no_surviving_loads=True,
        unchanged_remaining_source_graph=True,all_unlisted_source_byte_equal=True,
        protected_source_and_ports_unchanged=True,primary_ports_assignments_unchanged=True,
        existing_camera_689_cells_unchanged=True,prototype_test=str(tested),prototype_test_sha256=sha(tested/'result.json'),
        new_luts=0,new_flip_flops=0,new_clocks=0,changed_ram_address_pins=0,
        gain_register=52,exposure_half_lines=1120,
        evidence={str(f):sha(f) for f in (Path(__file__),BASE/'inputs.json',BASE/'native_source.v',MAP/'sensor74-map-audit.json',log,tested/'inputs.json',tested/'result.json')},
        files={str(f):sha(f) for f in OUT.iterdir() if f.is_file()},hardware_access=False,board_release=False)
    (OUT/'inputs.json').write_text(json.dumps(r,indent=2));print(r['status'],r['removed_kinds'],flush=True)
    faulthandler.cancel_dump_traceback_later()

if __name__=='__main__':main()
