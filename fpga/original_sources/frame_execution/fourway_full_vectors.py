"""Continuous-frame inputs: tensor oracle is assertion-only, never an RTL source."""
from __future__ import annotations
import argparse,json,sys,hashlib,shutil
from pathlib import Path
import numpy as np
from fourway_frame_program import build,ROOT,HERE
sys.path.insert(0,str(HERE.parent/'pw_array_feed'))
from feed_reference import generate,parameters,quantize
from model_contract import compile_contract,load_contract
from coefficient_layout import build_layout
from residual_q17_align import encode_alignment,fixtures
from contract_reference import residual_value
import hardware_schedule_model as hw

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rq(x,m):
    p=x.astype(np.int64)*m
    return np.clip(np.sign(p)*((np.abs(p)+65536)//131072),-128,127).astype(np.int8)
def make(root,optimized=False,pipeline=False,compact_lifetime=False):
    generate(root,'dense')
    plan=compile_contract(load_contract());params=parameters('dense')
    align=[encode_alignment(f) for f in fixtures()]
    y,x,c=np.indices((720,1280,3),dtype=np.int32)
    tensors={'input':(((x*x+3*x*c+7*x+11*y+y*y+5*c)%37)-18).astype(np.int8)}
    tensors['input'].tofile(root/'input.bin')
    offsets={};size=0;metadata=[];checks=0
    with (root/'golden.bin').open('wb') as fout:
        for node in plan['nodes']:
            name=node['id'];a=tensors[node['inputs'][0]];kind=node['kind'];p=params.get(name)
            if kind=='nearest_x2':z=a.repeat(2,0).repeat(2,1)
            elif kind=='residual_add':
                record=align[int(name[3])-1];b=tensors[node['inputs'][1]]
                # Independent exact rational half-away branch alignment.
                f=record['parameters'];ratio=np.array(f['main_scale'] if record['main_scaled'] else f['skip_scale'])
                common=f['common_scale'];num=int(ratio[0])*common[1];den=int(ratio[1])*common[0]
                small=(a if record['main_scaled'] else b).astype(np.int64)
                scaled=np.sign(small)*((2*np.abs(small)*num+den)//(2*den))
                z=rq(scaled+(b if record['main_scaled'] else a),record['post_multiplier'])
                for index in (0,17,1111,a.size-1):
                    assert int(z.flat[index])==residual_value(int(a.flat[index]),int(b.flat[index]),f);checks+=1
            else:
                weights=np.array(p.weights,dtype=np.int64)
                if kind=='pw1x1':acc=a.astype(np.int64)@weights.T
                else:
                    h,w,co=node['output_shape_hwc'];s=node['stride'];acc=np.zeros((h,w,co),dtype=np.int64)
                    padded=np.pad(a,((1,1),(1,1),(0,0)),mode='edge').astype(np.int64)
                    for ky in range(3):
                        for kx in range(3):
                            patch=padded[ky:ky+s*h:s,kx:kx+s*w:s,:]
                            if kind=='dw3x3':acc+=patch*weights[:,ky*3+kx]
                            else:
                                for ic in range(3):acc+=patch[:,:,ic,None]*weights[:,(ky*3+kx)*3+ic]
                biased=np.clip(acc+np.array(p.bias),-(1<<31),(1<<31)-1)
                z=rq(biased,np.array(p.multiplier))
                if p.relu:z=np.maximum(z,0)
                # Scalar independent contract checks, including every channel.
                for yy,xx in ((0,0),(1,7),(z.shape[0]-1,z.shape[1]-1)):
                    for oc in range(z.shape[2]):
                        dot=0
                        if kind=='pw1x1':dot=sum(int(a[yy,xx,ic])*p.weights[oc][ic] for ic in range(a.shape[2]))
                        else:
                            for ky in range(3):
                                for kx in range(3):
                                    sy=max(0,min(a.shape[0]-1,yy*node['stride']+ky-1));sx=max(0,min(a.shape[1]-1,xx*node['stride']+kx-1))
                                    if kind=='dw3x3':dot+=int(a[sy,sx,oc])*p.weights[oc][ky*3+kx]
                                    else:dot+=sum(int(a[sy,sx,ic])*p.weights[oc][(ky*3+kx)*3+ic] for ic in range(3))
                        assert quantize(dot,p.bias[oc],p.multiplier[oc],p.relu)[0]==int(z[yy,xx,oc]);checks+=1
            assert list(z.shape)==node['output_shape_hwc']
            tensors[name]=z
            padded=np.pad(z,((0,0),(0,0),(0,(-z.shape[2])%8)))
            offsets[name]=size;raw=padded.tobytes();fout.write(raw);size+=len(raw)
            metadata.append(dict(id=name,offset=offsets[name],shape=list(z.shape),padded_channels=padded.shape[2],sha256=hashlib.sha256(z.tobytes()).hexdigest(),unique_values=int(np.unique(z).size)))
            print('GOLD',name,z.shape,flush=True)
    program=build(adaptive=True,optimized=optimized,pipeline=pipeline,compact_lifetime=compact_lifetime);ops={o['name']:o for o in build_layout()['operators']};nodes={n['id']:n for n in plan['nodes']}
    packed=[];annotations=[];total=0;expected_words=0
    pack=lambda rows,key,bits:sum(d[key]<<(bits*i) for i,d in enumerate(rows))
    for ti,t in enumerate(program['tiles']):
        sourcebase,destbase={'encoder':(hw.RGB_INPUT_A_BASE,hw.FEATURE0_BASE),'residual':(hw.FEATURE0_BASE,hw.FEATURE1_BASE),'decoder':(hw.FEATURE1_BASE,hw.RGB_OUTPUT_C_BASE)}[t['group']]
        for command in t['commands']:
            f=[0]*32
            if command['kind']!='compute':
                row=command['row'];store=command['kind']=='store';f[0]=int(store);f[1]=nodes[command['final']]['index'];f[2]=row['width'];f[3]=row['groups'];f[4]=row['plane'];f[5]=row['base']
                f[6]=int(command['channels']==3);f[7]=(destbase if store else sourcebase)+(row['y']*command['global_width']+row['origin'])*command['channels']
                f[8]=int(store and t['group']=='decoder');f[9]=row['y'];f[10]=row['origin']
            else:
                q=command;node=nodes[q['node']];op=ops[q['parameter_node']];fl=(int(q['operator'] in ('pw1x1','pw_fused'))|int(q['nn'])<<1|int(q['operator']=='conv3x3')<<2|int(q['stride2'])<<3|int(bool(q['skip']))<<4)
                rec=align[int(q['node'][3])-1] if q['skip'] else dict(main_scaled=False,multiplier=0,post_multiplier=0)
                fl|=int(rec['main_scaled'])<<5;ib=pack(q['inputs'],'base',10);ob=pack(q['outputs'],'base',10);sb=pack(q['skip'],'base',10)
                assert len({d['origin'] for d in q['inputs']})==1
                go=q['output_groups'];globalx=q['output_x']//(2 if q['stride2'] else 1);globaly=q['outputs'][0]['y']
                f[:29]=[2,node['index'],q['pixels'],q['rows'],q['input_groups'],go,fl,q['output_x'],q['source_width'],q['inputs'][0]['origin'],
                    pack(q['inputs'],'plane',2),ib&0xffffffff,ib>>32,pack(q['outputs'],'plane',2),ob&0xffffffff,ob>>32,
                    op['weights']['offset'],op['bias']['offset'],op['multiplier']['offset'],op['relu']['offset'],
                    pack(q['skip'],'plane',2),sb&0xffffffff,sb>>32,q['skip'][0]['delta'] if q['skip'] else 0,rec['multiplier'],rec['post_multiplier'],
                    offsets[q['node']]+(globaly*node['output_shape_hwc'][1]+globalx)*go*8,node['output_shape_hwc'][1]*go*8,q['mask']]
                expected_words+=q['pixels']*q['rows']*go
            f[30]=ti;f[31]=total
            assert all(0<=v<2**32 for v in f);packed.append(sum(v<<(32*i) for i,v in enumerate(f)));total+=1
            annotations.append(dict(command=total-1,tile=ti,group=t['group'],**command))
    last={}
    for i,q in enumerate(annotations):
        if q['kind']=='compute':
            names=[q['node']]
            if q['skip']:names.append(q['parameter_node'])
            if q['nn']:names.append(nodes[q['node']]['inputs'][0])
            for name in names:last[nodes[name]['index']]=i
    assert set(last)==set(range(29))
    for nid,i in last.items():packed[i]|=1<<(29*32+nid)
    (root/'program.hex').write_text(''.join(f'{v:0256x}\n' for v in packed))
    (root/'program.json').write_text(json.dumps(annotations,separators=(',',':')))
    manifest=dict(status='CONTINUOUS_FRAME_VECTOR_READY_NOT_RTL_PASS',commands=total,golden_bytes=size,expected_compute_words=expected_words,
        input_bytes=720*1280*3,final_pixels=921600,scalar_oracle_checks=checks,nodes=metadata,residual=align,
        finite_program={k:v for k,v in program.items() if k!='tiles'},
        golden_assertions_only=True,intermediate_tensor_injection=False,trained_weights=False,
        evidence={str(p):digest(p) for p in (Path(__file__),HERE/'fourway_frame_program.py',HERE/'compile_frame.py',ROOT/'network_ab/model_b_contract.json')})
    snapshot=root/'generation-source';snapshot.mkdir();manifest['generation_sources']=[]
    for source,value in manifest['evidence'].items():
        dst=snapshot/Path(source).name;shutil.copy2(source,dst);assert digest(dst)==value
        manifest['generation_sources'].append(dict(path=source,snapshot=dst.relative_to(root).as_posix(),sha256=value))
    manifest['files']={p.relative_to(root).as_posix():digest(p) for p in root.rglob('*') if p.is_file()}
    (root/'frame_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:manifest[k] for k in ('commands','golden_bytes','expected_compute_words','scalar_oracle_checks')}),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--optimized',action='store_true');ap.add_argument('--pipeline',action='store_true');ap.add_argument('--compact-lifetime',action='store_true');a=ap.parse_args();make(a.root,a.optimized,a.pipeline,a.compact_lifetime)

