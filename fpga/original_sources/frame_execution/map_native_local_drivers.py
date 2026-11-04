"""Fresh official map for the proved local-driver candidate, never download."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha,run
from complete_native_logic_split import resource_counts,protected_blocks
from audit_native_roundtrip import verify
from probe_native_open_ports import compare_boundary
from native_local_drivers import graph,lexical_tokens,render_token,CELL

def flattened_public_ports(text):
    def cell(m):
        def port(p):
            values=[render_token(t) for t in lexical_tokens(p[2])]
            body='' if not values else (values[0] if len(values)==1 else '{'+', '.join(values)+'}')
            return '.'+p[1]+'('+body+')'
        pins=re.sub(r'\.(\w+)\(([^()]*)\)',port,m[3])
        return m[0][:m.start(3)-m.start()]+pins+m[0][m.end(3)-m.start():]
    return CELL.sub(cell,text)

def lexical_roundtrip(before,after):
    a,b=graph(before),graph(after);forward={};reverse={}
    assert a.keys()==b.keys()
    def bind(x,y,where):
        if x[0]=='literal' or y[0]=='literal':assert x==y,('literal',where,x,y);return
        assert x not in forward or forward[x]==y,('forward',where,x,y,forward.get(x))
        assert y not in reverse or reverse[y]==x,('reverse',where,x,y,reverse.get(y))
        forward[x]=y;reverse[y]=x
    for n,(kind,pins) in a.items():
        assert b[n][0]==kind
        for p in pins.keys() & b[n][1].keys():
            x,y=pins[p],b[n][1][p];assert len(x)==len(y),(n,p)
            for v,w in zip(x,y):bind(v,w,n+'.'+p)
    # Primary assignment destinations remain anchored; internal aliases may rename.
    def assigns(t):return dict(re.findall(r'^\s*assign\s+(.*?)\s*=\s*(.*?)\s*/\*',t,re.M))
    aa,bb=assigns(before),assigns(after);assert aa.keys()==bb.keys()
    for n in aa:
        for t in lexical_tokens(n):bind(t,t,'primary output')
        x,y=lexical_tokens(aa[n]),lexical_tokens(bb[n]);assert len(x)==len(y)
        for v,w in zip(x,y):bind(v,w,'primary assignment')
    return dict(status='LEXICAL_PUBLIC_NET_BIJECTION_PASS',cells=len(a),nets=len(forward),
        escaped_scalar_and_bus_bit_distinguished=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--audit-only',action='store_true')
    a=p.parse_args();src=a.source.resolve();root=a.root.resolve()
    assert root.is_relative_to(ROOT/'tmp') and (root.exists() if a.audit_only else not root.exists())
    proof=json.loads((src/'result.json').read_text());assert proof['status']=='REVERSIBLE_LOCAL_DRIVER_GRAPH_PASS'
    assert proof['source_sha256']==sha(src/'native_source.v') and proof['replications_sha256']==sha(src/'replications.json')
    for f,h in proof['inputs'].items():assert sha(f)==h,f
    parent=ROOT/'tmp/frame-execution/full-native-split-map4'
    if not a.audit_only:root.mkdir()
    try:
        if a.audit_only:
            assert not (root/'result.json').exists()
            meta=json.loads((root/'inputs.json').read_text())
            assert meta['transform_result_sha256']==sha(src/'result.json')
            for f,h in meta['files'].items():assert sha(root/f)==h,f
            body=(root/'map.log').read_text();seconds=None
        else:
            for f in ('native_source.v','replications.json'):shutil.copy2(src/f,root/f)
            for f in ('ti60f225_oob.xml','ti60f225_oob.sdc','ti60f225_oob.peri.xml','map.cmd'):shutil.copy2(parent/f,root/f)
            for f in parent.glob('*.bin'):shutil.copy2(f,root/f.name)
            shutil.copy2(__file__,root/'runner.py')
            meta=dict(parent=str(parent),transform_result=str(src/'result.json'),transform_result_sha256=sha(src/'result.json'),
                parent_map_sha256=sha(parent/'outflow/ti60f225_oob.map.v'),files={f.name:sha(f) for f in root.iterdir() if f.is_file()},hardware_access=False)
            (root/'inputs.json').write_text(json.dumps(meta,indent=2))
            body,seconds=run(['cmd.exe','/d','/c','map.cmd'],root,'map.log',timeout=480)
        assert re.search(r'map\s*:\s*PASS',body)
        net=root/'outflow/ti60f225_oob.map.v';source=(root/'native_source.v').read_text();mapped=net.read_text()
        graph=verify(flattened_public_ports(source),flattened_public_ports(mapped));assert graph['added_protected']==0
        lexical=lexical_roundtrip(source,mapped)
        boundary=compare_boundary(source,mapped,graph['renamed_nets'])
        (root/'public_connectivity_audit.json').write_text(json.dumps(graph,indent=2))
        old=resource_counts(parent/'outflow/ti60f225_oob.map.rpt');new=resource_counts(root/'outflow/ti60f225_oob.map.rpt')
        expected=dict(old)
        for k,v in proof['added_kinds'].items():expected[k]=expected.get(k,0)+v
        assert {k:v for k,v in expected.items() if v}=={k:v for k,v in new.items() if v},(expected,new)
        assert protected_blocks((parent/'outflow/ti60f225_oob.map.v').read_text())==protected_blocks(source)
        assert '[EFX-0654' not in (root/'outflow/ti60f225_oob.warn.log').read_text()
        for f,h in meta['files'].items():assert sha(root/f)==h,f
        result=dict(status='FULL_NATIVE_LOCAL_DRIVERS_MAP_GRAPH_PASS',seconds=seconds,counts=new,added_kinds=proof['added_kinds'],
            map_sha256=sha(net),vdb_sha256=sha(root/'outflow/ti60f225_oob.vdb'),inputs_sha256=sha(root/'inputs.json'),
            public_audit_sha256=sha(root/'public_connectivity_audit.json'),protected_boundary=boundary,
            lexical_audit=lexical,
            audit_only_without_resynthesis=a.audit_only,final_auditor_sha256=sha(__file__),
            hardware_access=False,physical_pass=False,board_release=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as e:
        (root/('audit_failure.json' if a.audit_only else 'failure.json')).write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise

if __name__=='__main__':main()
