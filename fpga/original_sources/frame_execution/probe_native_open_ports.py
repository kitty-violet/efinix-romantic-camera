"""Map-only sensitivity check of ten unconnected exported protected ports.

This is a disposable experiment, never an implementation or download source.
Tie the exact ten warning nets high and require the official mapper to discard
all of those connections again, with every public cell and resource unchanged.
"""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha,run
from complete_native_logic_split import resource_counts
from audit_native_roundtrip import verify,tokens
from audit_local_reset_implementation import norm

PARENT=ROOT/'tmp/frame-execution/full-native-split-map4'

def protected_pins(text):
    start=text.index('protected_cell protected_inst (')
    end=text.index(';',start)
    return {norm(n):norm(v) for n,v in re.findall(r'\.(\\\S+|\w+)\s*\(([^()]*)\)',text[start:end])}

def compare_boundary(before,after,renamed):
    a,b=protected_pins(before),protected_pins(after)
    assert a.keys()==b.keys(),('protected boundary ports',len(a),len(b))
    bits=0;open_bits=0
    for name,old in a.items():
        x,y=tokens(old),tokens(b[name]);assert len(x)==len(y),name
        for v,w in zip(x,y):
            bits+=1
            if re.fullmatch(r'Open_\d+',v) and re.fullmatch(r'Open_\d+',w):
                open_bits+=1;continue
            assert renamed.get(v,v)==w,('protected boundary changed',name,v,w)
    return dict(ports=len(a),bits=bits,disconnected_bits=open_bits,other_connections_equal=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();out=a.root.resolve()
    assert out.is_relative_to(ROOT/'tmp') and not out.exists();out.mkdir()
    try:
        source=PARENT/'native_source.v';original=source.read_text()
        warning=PARENT/'outflow/ti60f225_oob.warn.log'
        names=re.findall(r"\[EFX-0201 WARNING\] Re-wiring to GND non-driven net/port '([^']+)'",warning.read_text())
        assert names==['Open_'+str(i) for i in range(8,18)],names
        ports=protected_pins(original)
        samples=tokens(ports['u_ddr3_top/samples']);assert len(samples)==40
        locations={n:39-samples.index(n) for n in names}
        assert sorted(locations.values())==[12,13,14,15,32,33,36,37,38,39]
        # These identifiers occur only at that protected boundary. No public
        # driver/consumer is silently overridden by the experiment.
        assert all(len(re.findall(r'\b'+n+r'\b',original))==1 for n in names)
        declaration='\n'.join("    wire "+n+" = 1'b1;" for n in names)+'\n'
        insertion=original.index('    protected_cell protected_inst (')
        changed=original[:insertion]+declaration+original[insertion:]
        (out/'native_source.v').write_text(changed,encoding='utf8')
        for name in ('ti60f225_oob.xml','ti60f225_oob.sdc','ti60f225_oob.peri.xml','map.cmd'):
            shutil.copy2(PARENT/name,out/name)
        for f in PARENT.glob('*.bin'):shutil.copy2(f,out/f.name)
        shutil.copy2(__file__,out/'runner.py')
        meta=dict(parent=str(PARENT),source_sha256=sha(source),map_sha256=sha(PARENT/'outflow/ti60f225_oob.map.v'),
                  warning_sha256=sha(warning),warning_sample_bits=locations,
                  files={f.name:sha(f) for f in out.iterdir() if f.is_file()},hardware_access=False)
        (out/'inputs.json').write_text(json.dumps(meta,indent=2))
        body,seconds=run(['cmd.exe','/d','/c','map.cmd'],out,'map.log',timeout=480)
        assert re.search(r'map\s*:\s*PASS',body)
        mapped=out/'outflow/ti60f225_oob.map.v';oldmap=PARENT/'outflow/ti60f225_oob.map.v'
        audit=verify(oldmap.read_text(),mapped.read_text())
        assert audit['added_protected']==0
        boundary=compare_boundary(oldmap.read_text(),mapped.read_text(),audit['renamed_nets'])
        (out/'public_connectivity_audit.json').write_text(json.dumps(audit,indent=2))
        counts=resource_counts(out/'outflow/ti60f225_oob.map.rpt')
        assert counts==resource_counts(PARENT/'outflow/ti60f225_oob.map.rpt')
        output_samples=tokens(protected_pins(mapped.read_text())['u_ddr3_top/samples'])
        assert len(output_samples)==40
        # A used input tied high would retain the constant or change mapped
        # logic. Re-exporting each as Open demonstrates its disconnection.
        for bit in locations.values():assert re.fullmatch(r'Open_\d+',output_samples[39-bit]),(bit,output_samples[39-bit])
        warnings=(out/'outflow/ti60f225_oob.warn.log').read_text()
        assert '[EFX-0201 WARNING]' not in warnings and '[EFX-0654' not in warnings
        result=dict(status='EXPORTED_OPEN_PORT_CONSTANT_SENSITIVITY_PASS',inputs_sha256=sha(out/'inputs.json'),
                    seconds=seconds,warning_sample_bits=locations,public_cells=audit['common'],
                    resource_counts_unchanged=True,all_tested_protected_ports_disconnected=True,
                    protected_boundary=boundary,
                    public_audit_sha256=sha(out/'public_connectivity_audit.json'),map_sha256=sha(mapped),
                    evidence={str(p):sha(p) for p in (source,oldmap,warning,mapped,out/'outflow/ti60f225_oob.map.rpt',out/'outflow/ti60f225_oob.warn.log',Path(__file__))},
                    hardware_access=False,ddr_calibration_proven=False)
        (out/'result.json').write_text(json.dumps(result,indent=2));print(result['status'],seconds,flush=True)
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise

if __name__=='__main__':main()
