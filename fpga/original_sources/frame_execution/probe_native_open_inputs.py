"""Check all exported unconnected input buses, including warning grouping.

The frontend reports one undriven name per source group; EFX-0201 prints only
ten. Expand each reported name to its complete uniform-direction port bus.
Drive those otherwise unused input bits high in an isolated mapping experiment.
"""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha,run
from complete_native_logic_split import resource_counts
from audit_native_roundtrip import verify,tokens
from probe_native_open_ports import PARENT,protected_pins,compare_boundary

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    out=a.root.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists();out.mkdir()
    try:
        source=PARENT/'native_source.v';original=source.read_text()
        warning=PARENT/'outflow/ti60f225_oob.warn.log';wt=warning.read_text()
        reported=set(re.findall(r"net '(Open_\d+)' does not have a driver",wt))
        assert len(reported)==175
        ports=protected_pins(original);port_tokens={k:tokens(v) for k,v in ports.items()}
        groups={k:sorted({n for n in v if re.fullmatch(r'Open_\d+',n)}) for k,v in port_tokens.items() if set(v)&reported}
        names=sorted({n for v in groups.values() for n in v},key=lambda x:int(x[5:]))
        assert reported<=set(names)
        all_occurrences=re.findall(r'\bOpen_\d+\b',original)
        from collections import Counter
        counts=Counter(all_occurrences);assert all(counts[n]==1 for n in names)
        declaration='\n'.join("    wire "+n+" = 1'b1;" for n in names)+'\n'
        insertion=original.index('    protected_cell protected_inst (')
        (out/'native_source.v').write_text(original[:insertion]+declaration+original[insertion:],encoding='utf8')
        for n in ('ti60f225_oob.xml','ti60f225_oob.sdc','ti60f225_oob.peri.xml','map.cmd'):
            shutil.copy2(PARENT/n,out/n)
        for f in PARENT.glob('*.bin'):shutil.copy2(f,out/f.name)
        shutil.copy2(__file__,out/'runner.py')
        meta=dict(parent=str(PARENT),source_sha256=sha(source),map_sha256=sha(PARENT/'outflow/ti60f225_oob.map.v'),
            warning_sha256=sha(warning),reported_names=sorted(reported),input_port_groups=groups,
            tied_names=names,files={f.name:sha(f) for f in out.iterdir() if f.is_file()},hardware_access=False)
        (out/'inputs.json').write_text(json.dumps(meta,indent=2))
        body,seconds=run(['cmd.exe','/d','/c','map.cmd'],out,'map.log',timeout=480)
        assert re.search(r'map\s*:\s*PASS',body)
        mapped=out/'outflow/ti60f225_oob.map.v';oldmap=PARENT/'outflow/ti60f225_oob.map.v'
        audit=verify(oldmap.read_text(),mapped.read_text());assert audit['added_protected']==0
        boundary=compare_boundary(oldmap.read_text(),mapped.read_text(),audit['renamed_nets'])
        (out/'public_connectivity_audit.json').write_text(json.dumps(audit,indent=2))
        assert resource_counts(out/'outflow/ti60f225_oob.map.rpt')==resource_counts(PARENT/'outflow/ti60f225_oob.map.rpt')
        warnings=(out/'outflow/ti60f225_oob.warn.log').read_text()
        remaining=re.findall(r"\[EFX-0201 WARNING\].*",warnings)
        assert not remaining and '[EFX-0654' not in warnings,remaining
        result=dict(status='ALL_EXPORTED_OPEN_INPUT_CONSTANT_SENSITIVITY_PASS',inputs_sha256=sha(out/'inputs.json'),
            seconds=seconds,input_ports=len(groups),tied_bits=len(names),frontend_warning_groups=len(reported),
            public_cells=audit['common'],resource_counts_unchanged=True,protected_boundary=boundary,
            public_audit_sha256=sha(out/'public_connectivity_audit.json'),map_sha256=sha(mapped),
            evidence={str(p):sha(p) for p in (source,oldmap,warning,mapped,out/'outflow/ti60f225_oob.map.rpt',
                out/'outflow/ti60f225_oob.warn.log',Path(__file__),HERE/'probe_native_open_ports.py')},
            hardware_access=False,ddr_calibration_proven=False)
        (out/'result.json').write_text(json.dumps(result,indent=2));print(result['status'],len(names),seconds,flush=True)
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise

if __name__=='__main__':main()
