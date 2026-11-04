"""Prove mapper folding only on the exact retired-field serializer LUTs.

Normalize the source representation after complete truth-table comparison,
then run the original strict whole-graph bijection auditor on that result.
No tool output or actual synthesis input is edited.
"""
import hashlib,re
from native_local_drivers import CELL,graph,render_token
from probe_full_native_split_map import params
from camera_joint_native import OUT

def normalize(source,mapped,allowed):
    a,b=graph(source),graph(mapped);ap,bp=params(source),params(mapped)
    assert a.keys()==b.keys()
    allowed=set(allowed);assert allowed<=a.keys()
    forward={};reverse={}
    def bind(x,y,where):
        if x[0]=='literal' or y[0]=='literal':assert x==y,('literal',where,x,y);return
        assert x not in forward or forward[x]==y,('forward',where,x,y)
        assert y not in reverse or reverse[y]==x,('reverse',where,x,y)
        forward[x]=y;reverse[y]=x
    for n,(k,pins) in a.items():
        assert k==b[n][0]
        for pin in pins.keys() & b[n][1].keys():
            if n in allowed and pin not in OUT[k]:continue
            assert len(pins[pin])==len(b[n][1][pin])
            for x,y in zip(pins[pin],b[n][1][pin]):bind(x,y,(n,pin))
    normalized={};rows=[];patterns=0
    def bit(t,values):
        if t[0]=='literal':assert t[1] in ("1'b0","1'b1");return int(t[1][-1])
        return values[t]
    for n in sorted(allowed):
        k,pins=a[n];other=b[n][1];assert k=='EFX_LUT4'
        assert any(t[0]=='literal' for p,ts in pins.items() if p!='O' for t in ts)
        renamed={p:[forward[t] if t[0]=='net' else t for t in ts] for p,ts in pins.items()}
        assert renamed['O']==other['O']
        variables=sorted({t for ps in (renamed,other) for p,ts in ps.items() if p!='O' for t in ts if t[0]=='net'},key=str)
        assert len(variables)<=4
        for v in range(1<<len(variables)):
            values={t:(v>>i)&1 for i,t in enumerate(variables)}
            def val(ps,mask):
                return mask>>sum(bit(ps.get('I'+str(i),[('literal',"1'b0",None)])[0],values)<<i for i in range(4))&1
            assert val(renamed,ap[n]['LUTMASK'][1])==val(other,bp[n]['LUTMASK'][1]),('truth',n,v)
            patterns+=1
        norm={p:[reverse[t] if t[0]=='net' else t for t in ts] for p,ts in other.items()}
        normalized[n]=norm
        rows.append(dict(name=n,old_mask=ap[n]['LUTMASK'],mapped_mask=bp[n]['LUTMASK'],truth_patterns=1<<len(variables)))
    def change(m):
        n=m[2].lstrip('\\')
        if n not in normalized:return m[0]
        body=', '.join('.'+p+'('+(', '.join(render_token(t) for t in ts) if len(ts)==1 else '{'+', '.join(render_token(t) for t in ts)+'}')+')' for p,ts in normalized[n].items())
        return m[0][:m.start(3)-m.start()]+body+m[0][m.end(3)-m.start():]
    result=CELL.sub(change,source)
    def parameter(m):
        n=m[1].lstrip('\\')
        if n not in normalized:return m[0]
        return m[0][:m.start(2)-m.start()]+str(bp[n]['LUTMASK'][0])+"'d"+str(bp[n]['LUTMASK'][1])+m[0][m.end(2)-m.start():]
    result=re.sub(r'^\s*defparam\s+(\S+?)\s*\.LUTMASK\s*=\s*([^;]+);',parameter,result,flags=re.M)
    return result,dict(status='EXACT_RETIRED_SERIALIZER_LUT_CONSTANT_FOLDING_TRUTH_PASS',
        cells=len(rows),binary_patterns=patterns,rows=rows,normalized_reference_sha256=hashlib.sha256(result.encode()).hexdigest(),
        actual_design_and_tool_output_modified=False,remaining_graph_requires_original_strict_audit=True)
