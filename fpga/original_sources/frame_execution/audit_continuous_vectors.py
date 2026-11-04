"""Vector-aware graph including compact declarations and multiple wire names."""
import inspect,re
from native_local_drivers import graph

def expand_graph(text):
    ranges={}
    for a,b,names in re.findall(r'^\s*wire\s+\[(\d+):(\d+)\]\s*([^;]+);',text,re.M):
        names=re.sub(r'/\*.*?\*/','',names,flags=re.S)
        for name in names.split(','):
            name=name.strip().lstrip('\\');assert name and not any(c.isspace() for c in name),name
            ranges[name]=(int(a),int(b))
    g=graph(text)
    for kind,pins in g.values():
        for p,ts in pins.items():
            values=[]
            for t in ts:
                if t[0]=='net' and t[2] is None and t[1] in ranges:
                    a,b=ranges[t[1]];step=-1 if a>b else 1
                    values.extend(('net',t[1],i) for i in range(a,b+step,step))
                else:values.append(t)
            pins[p]=values
    return g

if __name__=='__main__':
    import audit_continuous_cut as original
    scope=dict(original.__dict__);scope.update(expand_graph=expand_graph,__file__=__file__)
    code=inspect.getsource(original.main).replace("DEST/'expanded_cut_audit.json'","DEST/'expanded_cut_audit2.json'")
    exec(compile(code,__file__+':strict-vector-cut','exec'),scope);scope['main']()
