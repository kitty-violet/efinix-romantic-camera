"""Read-only check of public primitive connectivity across native reimport.

Net renaming is proved consistent and bijective, never stripped by suffix.
Protected content is not decoded; original encrypted bytes are bound separately.
"""
import argparse
import collections
import json
import re
from pathlib import Path
from audit_snapshot_cells import cells
from probe_full_native_split_map import params, lut_function
from run_units import sha
from audit_local_reset_implementation import norm


def tokens(value):
    out=[]
    for token in re.findall(r"[^,{}\s]+", value):
        m=re.fullmatch(r'(.*)\[(\d+):(\d+)\]',token)
        if m:
            left,right=int(m[2]),int(m[3]);step=-1 if left>right else 1
            out.extend(m[1]+'['+str(i)+']' for i in range(left,right+step,step))
        else:out.append(token)
    return out


def verify(source, mapped):
    before, after = cells(source), cells(mapped)
    bp, ap = params(source), params(mapped)
    missing = sorted(before.keys() - after.keys())
    added = sorted(after.keys() - before.keys())
    assert not missing, missing[:12]
    assert all(n.startswith('protected_inst/') for n in added), added[:12]
    forward, reverse, conflicts = {}, {}, []
    def bind(x, y, where):
        if "'" in x or "'" in y:
            if x != y: conflicts.append((where, x, y, 'literal'))
            return
        if x in forward and forward[x] != y:
            conflicts.append((where,x,y,'forward',forward[x])); return
        if y in reverse and reverse[y] != x:
            conflicts.append((where,x,y,'reverse',reverse[y])); return
        forward[x], reverse[y] = y, x
    # Bind every common cell's corresponding port, allowing only the mapper
    # to omit LUT inputs that its truth table proves irrelevant.
    for name, (kind, pins, _) in before.items():
        other, outs, _ = after[name]
        assert kind == other, (name,kind,other)
        if kind!='EFX_LUT4':
            assert bp.get(name,{}) == ap.get(name,{}), ('parameter',name)
        if kind != 'EFX_LUT4':
            assert pins.keys() == outs.keys(), ('ports',name)
        for port in pins.keys() & outs.keys():
            x,y=tokens(pins[port]),tokens(outs[port])
            assert len(x)==len(y), ('width',name,port)
            for a,b in zip(x,y): bind(a,b,name+'.'+port)
    # Primary names are externally observable, so renaming cannot exchange
    # ports even if the internal graph happened to be symmetric.
    def ports(text):
        return {norm(n): (d, (rng or '').replace(' ','')) for d,rng,n in
                re.findall(r'^\s*(input|output|inout)\s*(\[[\d: ]+\])?\s*(\S+)\s*/\*',text,re.M)}
    assert ports(source)==ports(mapped), 'top ports changed'
    for name,(_,rng) in ports(source).items():
        for n in tokens(name+rng):bind(n,n,'top port')
    def assigns(text):
        return {norm(a):norm(b) for a,b in re.findall(r'^\s*assign\s+(.*?)\s*=\s*(.*?)\s*/\*',text,re.M)}
    ba,aa=assigns(source),assigns(mapped)
    assert ba.keys()==aa.keys(), 'top assignment destinations changed'
    for name,value in ba.items():
        x,y=tokens(value),tokens(aa[name]);assert len(x)==len(y),name
        for v in tokens(name):bind(v,v,'assignment destination')
        for v,w in zip(x,y):bind(v,w,'assignment source '+name)
    assert not conflicts, ('net_bijection',len(conflicts),conflicts[:8])
    def rename(value):
        return re.sub(r"[^,{}\s]+",lambda m:forward.get(m[0],m[0]),value)
    differences=[]
    for name,(kind,pins,_) in before.items():
        outs=after[name][1]
        changed={p:rename(v) for p,v in pins.items()}
        if kind=='EFX_LUT4':
            mask=bp[name]['LUTMASK'][1]
            if lut_function(changed,mask)!=lut_function(outs,ap[name]['LUTMASK'][1]):differences.append(name)
        elif changed!=outs:differences.append(name)
    assert not differences, ('primitive_equivalence',differences[:10])
    # Existing primitive parameters include all RAM initialization bits,
    # FF polarity/reset/clock controls, SRLs and every DSP mode/register.
    return dict(status='PUBLIC_PRIMITIVE_RENAME_EQUIVALENCE_PASS',common=len(before),
                common_kinds=dict(collections.Counter(v[0] for v in before.values())),
                added_protected=len(added),added_kinds=dict(collections.Counter(after[n][0] for n in added)),
                renamed_nets={x:y for x,y in forward.items() if x!=y},
                sequential_parameters_equal=True,lut_functions_equal=True,top_ports_and_assignments_equal=True,hardware_access=False,
                protected_internal_equivalence='Not asserted by this public-cell checker')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root
    src=root/'native_source.v';dst=root/'outflow/ti60f225_oob.map.v'
    try:
        result=verify(src.read_text(),dst.read_text())
        result.update(source_sha256=sha(src),map_sha256=sha(dst),checker_sha256=sha(__file__))
        (root/'public_connectivity_audit.json').write_text(json.dumps(result,indent=2))
        print(result['status'],result['common'],'renamed',len(result['renamed_nets']))
    except Exception as exc:
        (root/'public_connectivity_failure.json').write_text(json.dumps(dict(error=repr(exc)),indent=2))
        raise


if __name__=='__main__':main()
