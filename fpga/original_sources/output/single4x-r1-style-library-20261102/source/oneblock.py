"""Independent S4R1 experiment; all S4R2 artifacts are read-only teachers."""
from pathlib import Path
import sys,copy,json,struct,hashlib,zlib,shutil,os
import numpy as np
ROOT=Path(r'E:\fpga');BASE=ROOT/'output/single4x-r2-style-library-20261102';D=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'source'))
import library as r2
from library import torch,F,HalfAway,save_json,sha,tx,setup
sys.path.insert(0,str(ROOT/'output/single4x-r1-assessment-20261102/source'))
from assess_one_block import reduced,variant

class Model(r2.StyleModel):
    def __init__(self):
        super().__init__();self.rows=reduced(self.rows,2);self.residual={'res1_add':self.residual['res1_add']}
        for n in list(self.w):
            if n.startswith('res2_'):
                del self.w[n];del self.b[n];delattr(self,n+'_m')
        self.qat=True

def guard():
    D.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(D).free<7*2**30:raise RuntimeError('E free below 7GiB')

def decode(folder,version=1):
    if version==2:return r2.decode(folder)
    folder=Path(folder);g=json.loads((folder/'graph.json').read_text('utf-8'));layout=json.loads((folder/'layout.json').read_text('utf-8'));raw=(folder/'coefficients-padded.bin').read_bytes()
    canon=json.dumps(g,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    assert raw[:8]==b'S4R1PKG1' and raw[32:64]==hashlib.sha256(canon).digest()
    assert struct.unpack_from('<6I',raw,8)==(1,1280,720,11,1,3)
    assert len(raw)==8192 and not any(raw[layout['effective_bytes']:]) and hashlib.sha256(raw).hexdigest()==layout['sha256']
    ps={}
    for row in layout['operators']:
        ps[row['node']]={n:np.frombuffer(raw[f['offset']:f['offset']+f['length']],dtype=f['dtype']).reshape(f['shape']).copy() for n,f in row['fields'].items()}
    a,b,c=struct.unpack_from('<3i',raw,layout['residual_offset']);assert dict(main_scaled=bool(a),multiplier=b,post_multiplier=c)==g['residual']['res1_add']
    return g,ps

class Engine(r2.PackedEngine):
    def __init__(self,folder,version=1):
        self.description,data=decode(folder,version);self.rows=self.description['nodes'];self.residual=self.description['residual'];self.ops={}
        for n,p in data.items():self.ops[n]=(torch.tensor(p['weight'].astype(np.float32),device='cuda'),torch.tensor(p['bias'],device='cuda',dtype=torch.float64)[None,:,None,None],torch.tensor(p['multiplier'],device='cuda',dtype=torch.float64)[None,:,None,None])

def teacher_folder(key):return BASE/'revisions/film-pixel-r2/models'/key if key in ['OLD_FILM','PIXEL_R39'] else BASE/'models'/key

def inventory():
    entries=json.loads((BASE/'revisions/film-pixel-r2/inventory.json').read_text('utf-8'))['styles'];rows=[]
    for e in entries:
        if e['id']=='BW_LINE':continue
        f=teacher_folder(e['id']);row=copy.deepcopy(e)
        row.update(original_B1_source=e['source'],original_B1_sha256=e['source_sha256'],source=str(f/'coefficients-padded.bin'),source_sha256=sha(f/'coefficients-padded.bin'),source_folder=str(f),source_format='S4R2PKG1',new_topology='S4R1_8_32_8_V1')
        if e['id']=='R8-06':row['current_display_name']='热成像'
        if e['id']=='R8-07':row['current_display_name']='漫画'
        rows.append(row)
    assert len(rows)==16;save_json(D/'inventory.json',dict(styles=rows,count=16,all_two_residual_sources_read_only=True));return rows

@torch.no_grad()
def from_ops(ops,desc):
    m=Model().cuda();m.residual={'res1_add':copy.deepcopy(desc['residual']['res1_add'])}
    for n in m.w:
        w,b,mm=ops[n];s=mm.flatten().float()/131072
        m.w[n].copy_(w*s[:,None,None,None]);m.b[n].copy_(b.flatten().float()*s);getattr(m,n+'_m').copy_(mm.flatten())
    return m

@torch.no_grad()
def initialize(entry):
    old=Engine(entry['source_folder'],2);ops=dict(old.ops);method='delete_second_block'
    if entry['id']=='B03':ops=variant(old,2,True).ops;method='move_channel_shift_before_existing_synthesis_then_delete_second'
    if entry['id'].startswith('R8-'):
        wm,bm,mm=ops['res1_b_pw'];wt,bt,mt=ops['res2_a_pw'];wp,bp,mp=ops['res2_b_pw']
        merge=wm[:,:,0,0].double()*mm.flatten().double()[:,None]/131072
        biasmerge=bm.flatten()*mm.flatten()/131072
        assert torch.equal(merge,merge.round()) and not biasmerge.any()
        joined=wt[:,:,0,0].double()@merge
        assert torch.equal(joined,joined.round()) and joined.abs().max()<=127
        ops['res1_b_pw']=(joined.float()[:,:,None,None],bt.clone(),mt.clone())
        # The decoder DW supplies the deferred threshold ReLU, then its PW
        # performs the original palette readout. Intermediate sums are exact
        # integers, so no rounding step is approximated in this composition.
        ops['up1_low_pw']=(wp[:8].clone(),bp[:,:8].clone(),mp[:,:8].clone())
        method='integer_exact_compose_synthesis_and_threshold_move_palette_to_decoder'
    m=from_ops(ops,old.description)
    if entry['id']=='OLD_FILM':m=film_initialize(m);method='single_block_finite_laplacian_gate_and_preserved_horizontal_spread'
    return m,old,method

@torch.no_grad()
def film_initialize(m):
    # Use one signed Laplacian response instead of two sequential min stages,
    # leaving the two residual DWs available to preserve the long-bar support.
    old_head=(m.w['rgb'].clone(),m.b['rgb'].clone(),m.rgb_m.clone())
    lap=torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],device='cuda').float()*4
    g=torch.tensor([[1,2,1],[2,4,2],[1,2,1]],device='cuda').float()/16
    m.w['down_dw'][4,0]=lap;m.w['down_dw'][5,0]=-lap;m.w['down_dw'][6,0]=g;m.b['down_dw'][6]=-80
    def row(n,c,terms,b=0):
        m.w[n][c].zero_();m.b[n][c]=b
        for j,v in terms.items():m.w[n][c,j,0,0]=v
    m.w['down_pw'].zero_();m.b['down_pw'].zero_()
    for c in range(3):row('down_pw',c,{c:1})
    row('down_pw',3,{3:1},-45);row('down_pw',4,{3:1},-90);row('down_pw',5,{3:1});row('down_pw',6,{6:1});row('down_pw',30,{3:-1},45)
    for first,c in [(20,4),(23,5)]:
        for i in range(3):
            terms={c:120,3:-12,6:12};bias=-120*(3.5+i*.5)
            row('down_pw',first+i,terms,bias);row('down_pw',first-12+i,terms,bias)
    for n,w in m.w.items():
        if n.startswith('res1_'):
            w.zero_();m.b[n].zero_()
            if w.shape[-1]==3:w[:,0,1,1]=1
            else:
                for c in range(32):w[c,c,0,0]=1
    for c in range(20,26):m.w['res1_a_dw'][c].zero_();m.w['res1_a_dw'][c,0,1,:]=.25
    row('res1_a_pw',26,{20:4,21:-8,22:4});row('res1_a_pw',27,{23:4,24:-8,25:4});row('res1_a_pw',28,{8:.5,9:-1,10:.5,11:.5,12:-1,13:.5})
    for c in [26,27]:m.w['res1_b_dw'][c].zero_();m.w['res1_b_dw'][c,0,1,:]=1
    m.w['res1_b_dw'][28].zero_();m.w['res1_b_dw'][28,0,0,1]=1
    m.calibrate();m.w['rgb'].copy_(old_head[0]);m.b['rgb'].copy_(old_head[1]);m.rgb_m.copy_(old_head[2])
    return m

def export(m,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    g=r2.spec(m);g['schema']='S4R1_8_32_8_V1';data=r2.arrays(m)
    digest=hashlib.sha256(json.dumps(g,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')).digest()
    blob=bytearray(64);blob[:8]=b'S4R1PKG1';struct.pack_into('<6I',blob,8,1,1280,720,11,1,3);blob[32:64]=digest;layout=[]
    for row in m.rows:
        n=row['id']
        if n not in data:continue
        fields={}
        for kind,dtype in [('weight','i1'),('bias','<i4'),('multiplier','<i4')]:
            while len(blob)%8:blob.append(0)
            a=data[n][kind];b=a.astype(dtype).tobytes();fields[kind]=dict(offset=len(blob),length=len(b),shape=list(a.shape),dtype=dtype);blob.extend(b)
        layout.append(dict(node=n,fields=fields))
    while len(blob)%8:blob.append(0)
    offset=len(blob);p=m.residual['res1_add'];blob.extend(struct.pack('<3i',int(p['main_scaled']),p['multiplier'],p['post_multiplier']));effective=len(blob);raw=bytes(blob)+bytes(8192-effective)
    assert effective<=8192 and len(data)==11
    (folder/'coefficients.bin').write_bytes(bytes(blob));(folder/'coefficients-padded.bin').write_bytes(raw)
    (folder/'load_words.hex').write_text(''.join(f'{int.from_bytes(raw[i:i+8],"little"):016x}\n' for i in range(0,len(raw),8)),encoding='ascii')
    save_json(folder/'graph.json',g);save_json(folder/'layout.json',dict(schema='S4R1PKG1',header_bytes=64,graph_sha256=digest.hex(),operators=layout,residual_offset=offset,effective_bytes=effective,padded_bytes=8192,crc32=f'{zlib.crc32(raw):08x}',sha256=hashlib.sha256(raw).hexdigest()))
    np.savez(folder/'integer-coefficients.npz',**{n+'_'+k:a for n,p in data.items() for k,a in p.items()});return raw

def independent_oracle(folder):
    from network_ab.contract_reference import convolution
    desc,ps=decode(folder);engine=Engine(folder);rng=np.random.default_rng(771);total=0;vectors=[]
    for a in [rng.integers(0,256,(8,12,3),dtype=np.uint8),np.full((8,8,3),255,np.uint8)]:
        values={'input':(a.astype(int)-128).tolist()};actual=engine(tx(a[None]),True)
        for r in desc['nodes']:
            n=r['id'];x=values[r['inputs'][0]]
            if n in ps:
                p=ps[n];w=p['weight'].transpose(0,2,3,1).reshape(r['channels'],-1).astype(int).tolist();y,_=convolution(x,r,w,p['bias'].astype(int).tolist(),p['multiplier'].astype(int).tolist())
            elif r['kind']=='nearest':
                f=r['factor'];y=[[list(x[i//f][j//f]) for j in range(len(x[0])*f)] for i in range(len(x)*f)]
            else:
                p=desc['residual'][n];z=np.array(values[r['inputs'][1]],np.int64);xx=np.array(x,np.int64);small,other=(xx,z) if p['main_scaled'] else (z,xx)
                rnd=lambda v:np.sign(v)*((np.abs(v)+65536)//131072)
                y=rnd((rnd(small*p['multiplier']).clip(-128,127)+other)*p['post_multiplier']).clip(-128,127).astype(int).tolist()
            values[n]=y;expected=torch.tensor(y,device='cuda').permute(2,0,1)[None].float();assert torch.equal(expected,actual[n]),n;total+=expected.numel()
        vectors.append(dict(input_u8=a.tolist(),output_i8=values['rgb']))
    save_json(Path(folder)/'reference-vectors.json',dict(vectors=vectors,all_node_scalars=total));return dict(all_nodes_exact=True,scalar_comparisons=total)

def setup_portable():
    src=(BASE/'source/verify_package.py').read_text('utf-8').replace('S4R2','S4R1').replace('(1,1280,720,15,2,3)','(1,1280,720,11,1,3)').replace('len(raw)==16384','len(raw)==8192').replace('len(lines)==2048','len(lines)==1024').replace('weight_count==6432 and bias_count==355 and len(parameters)==15 and len(graph[\'nodes\'])==18','weight_count==3808 and bias_count==227 and len(parameters)==11 and len(graph[\'nodes\'])==13').replace("['res1_add','res2_add']","['res1_add']").replace('parameters=6787','parameters=4035')
    (D/'source/verify_package.py').write_text(src,encoding='utf-8')
    inf=(BASE/'source/infer_package.py').read_text('utf-8').replace('S4R2','S4R1');(D/'source/infer_package.py').write_text(inf,encoding='utf-8')
