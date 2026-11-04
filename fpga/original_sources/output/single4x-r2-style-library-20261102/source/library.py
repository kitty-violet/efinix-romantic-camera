"""Isolated S4R2 style migration. Frozen B1 integers are teachers only."""
from pathlib import Path
import os,sys,json,hashlib,copy,struct,time,shutil
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(r'E:\fpga');D=Path(__file__).resolve().parents[1]
RECON=ROOT/'output/single-down-up-reconstruction-20261102'
sys.path.insert(0,str(RECON/'source'))
import reconstruct as rec
from reconstruct import torch,nn,F,HalfAway,setup,save_json,sha,Model,tx,rgb,metrics,costs
from core import B1,IntegerEngine,TrainingLeakyClamp
from network_ab.model_contract import compile_contract,load_contract
import numpy as np
from PIL import Image
OLD=ROOT/'output/b1-color-cycle-negative-20261027'
ARCHIVE=OLD/'retained-12-styles-20261029'
RESIDUAL={f'res{i}_add':dict(main_scaled=False,multiplier=1,post_multiplier=131071) for i in (1,2)}

def guard():
    free=shutil.disk_usage(D).free
    if free<7*2**30:raise RuntimeError(f'E disk free below 7GiB: {free}')

def inventory():
    entries=[]
    named=json.loads((OLD/'named-style-handoff-20261030/style-catalog.json').read_text('utf-8'))['named_styles']
    for key,name,oldid in [('A1','油画','A1'),('INK','水墨风','B1-水墨风-20261027'),('OLD_FILM','清朝老片风格','B1-清朝老片风格-20261027'),('PIXEL_R39','像素动画风格','B1-像素动画风格-R39-20261027')]:
        p=next(x for x in named if x['id']==oldid)
        entries.append(dict(id=key,name=name,formal_name=name,source=p['files']['padded_bin']['path'],source_sha256=p['coefficient_sha256'],selection='previously_finalized',family='legacy'))
    for x in json.loads((ARCHIVE/'archive-index.json').read_text('utf-8'))['styles']:
        entries.append(dict(id=x['id'],name=x['formal_name'] or x['old_display_alias'],formal_name=x['formal_name'],source=str(ARCHIVE/x['directory']/'coefficients-padded.bin'),source_sha256=x['coefficient_sha256'],selection='retained_backup',family=x['family']))
    current={'R8-06':('热成像',0),'B03':('阿凡达',1),'R8-07':('漫画',2),'PIXEL_R39':('像素动画',3),'W-T256':('阴曹地府',4)}
    historical_board_labels={'R8-06':'漫画','R8-07':'热成像'}
    deployed=ROOT/'output/deployment/legacy222-ble-five-cn-20261101-ble5/data'
    for e in entries:
        assert sha(e['source'])==e['source_sha256'],e['id']
        if e['id'] in current:
            name,slot=current[e['id']];p=deployed/f'slot{slot}/coefficients-padded.bin'
            assert sha(p)==e['source_sha256']
            e.update(current_display_name=name,current_deployed_slot=slot,current_deployed_source=str(p))
            if e['id'] in historical_board_labels:
                e.update(historical_board_label=historical_board_labels[e['id']],name_correction='2026-11-02 user: red R8-06 is 热成像; other R8-07 is 漫画; no hardware label write')
        e['new_topology']='S4R2_8_32_8_V1'
    backup=ROOT/'model-training-workspace/runs/style-library/black-white-line-v1-20261004/contour_fp32.pt'
    entries.append(dict(id='BW_LINE',name='黑白线条（教师参考实验）',formal_name=None,source=str(backup),source_sha256=sha(backup),selection='teacher_reference_only_not_an_accepted_B1_style',family='legacy_fp32',new_topology='S4R2_8_32_8_V1',user_clarification='Old archived FP32 research/teacher output; neither old fixed B1 nor new S4R2 had achieved it. Try last and report separately.'))
    assert len(entries)==17 and len({e['id'] for e in entries})==17
    save_json(D/'inventory.json',dict(styles=entries,count=17,accepted_or_retained_B1_styles=16,teacher_reference_experiments=1,source_weights_read_only=True,excluded=['R4-01','R4-09','R8-04','P4-structure','other_rejected_combinations'],historical_V14='not a retained style; no new starry training',new_hardware_implemented=False))
    return entries

def read_inventory():return json.loads((D/'inventory.json').read_text('utf-8'))['styles']

class StyleModel(Model):
    def __init__(self):
        super().__init__(True);self.residual=copy.deepcopy(RESIDUAL)
    def forward(self,x,all_nodes=False):
        clip=lambda z,lo,hi:TrainingLeakyClamp.apply(z,lo,hi,float(getattr(self,'leak',0.)))
        values={'input':x*128}
        if self.qat:values['input']=HalfAway.apply(values['input']).clamp(-128,127)
        for r in self.rows:
            name=r['id'];a=values[r['inputs'][0]]
            if name in self.w:
                if r['kernel']==3:a=F.pad(a,(1,1,1,1),mode='replicate')
                if self.qat:
                    m=getattr(self,name+'_m');s=(m/131072).float()
                    wi=HalfAway.apply(self.w[name]/s[:,None,None,None]).clamp(-127,127)
                    bi=HalfAway.apply(self.b[name]/s).double()[None,:,None,None]
                    dot=F.conv2d(a,wi,stride=r['stride'],groups=r['groups'])
                    y=clip(HalfAway.apply((dot.double()+bi).clamp(-(2**31),2**31-1)*m[None,:,None,None]/131072).float(),-128,127)
                else:y=clip(F.conv2d(a,self.w[name],self.b[name],stride=r['stride'],groups=r['groups']),-128,127)
                if r['activation']=='relu':y=clip(y,0,float('inf'))
            elif r['kind']=='nearest':y=a.repeat_interleave(r['factor'],-2).repeat_interleave(r['factor'],-1)
            else:
                z=values[r['inputs'][1]];p=self.residual[name];small,other=(a,z) if p['main_scaled'] else (z,a)
                if self.qat:
                    aligned=HalfAway.apply(small.double()*p['multiplier']/131072).clamp(-128,127)
                    y=HalfAway.apply((aligned+other.double())*p['post_multiplier']/131072).clamp(-128,127).float()
                else:y=((small*p['multiplier']/131072).clamp(-128,127)+other).mul(p['post_multiplier']/131072).clamp(-128,127)
            values[name]=y
        return values if all_nodes else values['rgb']/128

def teacher_parameters(source):
    teacher=B1(source=Path(source)).cuda();teacher.eval()
    return teacher

@torch.no_grad()
def initialize(entry):
    if entry['id']=='BW_LINE':
        m=StyleModel().cuda()
        with torch.no_grad():
            for name,w in m.w.items():
                if name.startswith('res') and name.endswith('_b_pw'):
                    w.zero_()
                    for c in range(32):w[c,c,0,0]=1
            m.w['rgb'].zero_();m.b['rgb'].fill_(100)
        m.calibrate();return m
    old=teacher_parameters(entry['source']);m=StyleModel().cuda()
    mapping={'stem':'stem','down_dw':'down1__dw','down_pw':'down2__pw','up1_low_dw':'up1_low_dw','up1_high':'up2_high_dw','rgb':'rgb_head'}
    for i in (1,2):
        for a,b in [('a','conv1'),('b','conv2')]:
            for typ in ['dw','pw']:mapping[f'res{i}_{a}_{typ}']=f'res{i}_{b}__{typ}'
    for new,oldkey in mapping.items():
        w=old.weight[oldkey];b=old.bias[oldkey]
        if new=='down_pw':w=w[:,:8]
        assert w.shape==m.w[new].shape,(new,w.shape,m.w[new].shape)
        m.w[new].copy_(w);m.b[new].copy_(b)
    if entry['id'] in ['A1','PIXEL_R39']:
        # Keep nonlinear encoder projections as distinct layers. Multiplying
        # their matrices would erase ReLU thresholds and collapse the style.
        m.w['down_pw'].zero_();m.b['down_pw'].zero_()
        m.w['down_pw'][:16].copy_(old.weight['down1__pw']);m.b['down_pw'][:16].copy_(old.bias['down1__pw'])
        m.w['res1_a_pw'].zero_();m.b['res1_a_pw'].copy_(old.bias['down2__pw'])
        m.w['res1_a_pw'][:,:16].copy_(old.weight['down2__pw'])
        for name in ['res1_b_pw','res2_a_pw']:
            m.w[name].zero_();m.b[name].zero_()
            for c in range(32):m.w[name][c,c,0,0]=1
        if entry['id']=='PIXEL_R39':
            m.w['res2_a_pw'].copy_(old.weight['res3_conv1__pw']);m.b['res2_a_pw'].copy_(old.bias['res3_conv1__pw'])
            m.w['res2_b_pw'].copy_(old.weight['res3_conv2__pw']);m.b['res2_b_pw'].copy_(old.bias['res3_conv2__pw'])
        else:
            m.w['res2_b_pw'].zero_();m.b['res2_b_pw'].zero_()
            m.w['res2_b_pw'][:16].copy_(old.weight['up1_low_pw']);m.b['res2_b_pw'][:16].copy_(old.bias['up1_low_pw'])
    # The two low-resolution projections are combined for initialization;
    # intervening nonlinear/spatial effects are learned from integer targets.
    w1=old.weight['up1_low_pw'][:,:,0,0];w2=old.weight['up2_low_pw'][:,:,0,0]
    m.w['up1_low_pw'].copy_((w2@w1)[:,:,None,None]);m.b['up1_low_pw'].copy_(w2@old.bias['up1_low_pw']+old.bias['up2_low_pw'])
    if entry['id']=='A1':
        m.w['up1_low_pw'].zero_();m.w['up1_low_pw'][:,:16].copy_(old.weight['up2_low_pw']);m.b['up1_low_pw'].copy_(old.bias['up2_low_pw'])
    for i in (1,2):m.residual[f'res{i}_add']=copy.deepcopy(old.package['residual_encoding'][f'res{i}_add'])
    m.calibrate();del old
    return m

class Teacher:
    def __init__(self,entry):
        self.entry=entry;self.edge=None
        if entry['id']=='BW_LINE':
            import importlib.util
            folder=Path(entry['source']).parent;sys.path.insert(0,str(folder))
            module_spec=importlib.util.spec_from_file_location('archived_bw_context',folder/'model.py')
            module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
            self.edge=module.ContextContourNet(difference=False).cuda().eval()
            self.edge.load_state_dict(torch.load(entry['source'],weights_only=True,map_location='cpu')['model'])
        else:self.engine=IntegerEngine(Path(entry['source']).read_bytes())
    @torch.inference_mode()
    def __call__(self,x,all_nodes=False):
        if self.edge is not None:
            y=(255*(1-self.edge(x).sigmoid())).floor().clamp(0,255).expand(-1,3,-1,-1)/128-1
            return {'rgb_head':y*128} if all_nodes else y
        if all_nodes:return self.engine(x*128,True)[0]
        return self.engine(x*128)/128

def arrays(model):
    result={}
    for n,w in model.w.items():
        m=getattr(model,n+'_m');s=(m/131072).float()
        result[n]=dict(weight=HalfAway.apply(w/s[:,None,None,None]).clamp(-127,127).to(torch.int8).cpu().numpy(),bias=HalfAway.apply(model.b[n]/s).int().cpu().numpy(),multiplier=m.int().cpu().numpy())
    return result

def spec(model):
    return dict(schema='S4R2_8_32_8_V1',input_shape_hwc=[720,1280,3],nodes=model.rows,residual=model.residual,numeric=dict(input='RGB uint8 minus 128',weights='signed INT8 OIHW',activations='signed INT8 scale 1/128',bias='signed INT32',multiplier='signed18 Q17 stored LE32',rounding='nearest half away from zero',padding='replicate',output='clamp signed INT8 plus 128'),old_fixed_B1_binary_compatible=False,hardware_implemented=False)

def export(model,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);description=spec(model);data=arrays(model)
    graph_bytes=json.dumps(description,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8');digest=hashlib.sha256(graph_bytes).digest()
    blob=bytearray(64);blob[:8]=b'S4R2PKG1';struct.pack_into('<6I',blob,8,1,1280,720,15,2,3);blob[32:64]=digest
    layout=[]
    for r in model.rows:
        n=r['id']
        if n not in data:continue
        fields={}
        for kind,dtype in [('weight','i1'),('bias','<i4'),('multiplier','<i4')]:
            while len(blob)%8:blob.append(0)
            a=data[n][kind];raw=a.astype(dtype).tobytes();fields[kind]=dict(offset=len(blob),length=len(raw),shape=list(a.shape),dtype=dtype);blob.extend(raw)
        layout.append(dict(node=n,fields=fields))
    while len(blob)%8:blob.append(0)
    residual_offset=len(blob)
    for name in ['res1_add','res2_add']:
        p=model.residual[name];blob.extend(struct.pack('<3i',int(p['main_scaled']),p['multiplier'],p['post_multiplier']))
    effective=len(blob);assert effective<=16384
    raw=bytes(blob)+bytes(16384-effective)
    (folder/'coefficients.bin').write_bytes(bytes(blob));(folder/'coefficients-padded.bin').write_bytes(raw)
    (folder/'load_words.hex').write_text(''.join(f'{int.from_bytes(raw[i:i+8],"little"):016x}\n' for i in range(0,len(raw),8)),encoding='ascii')
    save_json(folder/'graph.json',description)
    save_json(folder/'layout.json',dict(schema='S4R2PKG1',header_bytes=64,graph_sha256=digest.hex(),operators=layout,residual_offset=residual_offset,effective_bytes=effective,padded_bytes=len(raw),crc32=f'{__import__("zlib").crc32(raw):08x}',sha256=hashlib.sha256(raw).hexdigest()))
    np.savez(folder/'integer-coefficients.npz',**{n+'_'+k:a for n,p in data.items() for k,a in p.items()})
    return raw

def decode(folder):
    folder=Path(folder);desc=json.loads((folder/'graph.json').read_text('utf-8'));layout=json.loads((folder/'layout.json').read_text('utf-8'));raw=(folder/'coefficients-padded.bin').read_bytes()
    canonical=json.dumps(desc,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    assert raw[:8]==b'S4R2PKG1' and raw[32:64]==hashlib.sha256(canonical).digest()
    assert len(raw)==16384 and not any(raw[layout['effective_bytes']:]) and hashlib.sha256(raw).hexdigest()==layout['sha256']
    ps={}
    for row in layout['operators']:
        fields={}
        for name,f in row['fields'].items():fields[name]=np.frombuffer(raw[f['offset']:f['offset']+f['length']],dtype=f['dtype']).reshape(f['shape']).copy()
        ps[row['node']]=fields
    for i,name in enumerate(['res1_add','res2_add']):
        a,b,c=struct.unpack_from('<3i',raw,layout['residual_offset']+12*i)
        assert dict(main_scaled=bool(a),multiplier=b,post_multiplier=c)==desc['residual'][name]
    return desc,ps

class PackedEngine:
    def __init__(self,folder):
        self.description,data=decode(folder);self.rows=self.description['nodes'];self.residual=self.description['residual'];self.ops={}
        for n,p in data.items():self.ops[n]=(torch.tensor(p['weight'].astype(np.float32),device='cuda'),torch.tensor(p['bias'],device='cuda',dtype=torch.float64)[None,:,None,None],torch.tensor(p['multiplier'],device='cuda',dtype=torch.float64)[None,:,None,None])
    @torch.inference_mode()
    def __call__(self,x,all_nodes=False):
        v={'input':x*128};uses={}
        for r in self.rows:
            for k in r['inputs']:uses[k]=uses.get(k,0)+1
        for r in self.rows:
            n=r['id'];a=v[r['inputs'][0]]
            if n in self.ops:
                w,b,m=self.ops[n]
                if r['kernel']==3:a=F.pad(a,(1,1,1,1),mode='replicate')
                dot=F.conv2d(a,w,stride=r['stride'],groups=r['groups'])
                y=HalfAway.apply((dot.double()+b).clamp(-(2**31),2**31-1)*m/131072).clamp(-128,127).float()
                if r['activation']=='relu':y=y.clamp_min(0)
            elif r['kind']=='nearest':y=a.repeat_interleave(r['factor'],-2).repeat_interleave(r['factor'],-1)
            else:
                z=v[r['inputs'][1]];p=self.residual[n];small,other=(a,z) if p['main_scaled'] else (z,a)
                aligned=HalfAway.apply(small.double()*p['multiplier']/131072).clamp(-128,127)
                y=HalfAway.apply((aligned+other.double())*p['post_multiplier']/131072).clamp(-128,127).float()
            v[n]=y
            if not all_nodes:
                for k in r['inputs']:
                    uses[k]-=1
                    if not uses[k]:del v[k]
        return v if all_nodes else v['rgb']/128

def independent_oracle(folder):
    from network_ab.contract_reference import convolution
    desc,parameters=decode(folder);engine=PackedEngine(folder);rng=np.random.default_rng(771);count=0;gold=[]
    for a in [rng.integers(0,256,(8,12,3),dtype=np.uint8),np.full((8,8,3),255,np.uint8)]:
        values={'input':(a.astype(int)-128).tolist()};actual=engine(tx(a[None]),True)
        for r in desc['nodes']:
            n=r['id'];x=values[r['inputs'][0]]
            if n in parameters:
                p=parameters[n];w=p['weight'].transpose(0,2,3,1).reshape(r['channels'],-1).astype(int).tolist()
                y,_=convolution(x,r,w,p['bias'].astype(int).tolist(),p['multiplier'].astype(int).tolist())
            elif r['kind']=='nearest':
                f=r['factor'];y=[[list(x[i//f][j//f]) for j in range(len(x[0])*f)] for i in range(len(x)*f)]
            else:
                p=desc['residual'][n];z=np.asarray(values[r['inputs'][1]],np.int64);x=np.asarray(x,np.int64);small,other=(x,z) if p['main_scaled'] else (z,x)
                def rnd(v):return np.sign(v)*((np.abs(v)+65536)//131072)
                aligned=np.clip(rnd(small*p['multiplier']),-128,127);y=np.clip(rnd((aligned+other)*p['post_multiplier']),-128,127).tolist()
            values[n]=y;expected=np.asarray(y);got=actual[n][0].permute(1,2,0).cpu().numpy().astype(int)
            assert np.array_equal(expected,got),n;count+=expected.size
        gold.append(dict(input_u8=a.tolist(),output_i8=values['rgb']))
    save_json(Path(folder)/'reference-vectors.json',dict(vectors=gold))
    return dict(all_nodes_exact=True,scalar_comparisons=count)

if __name__=='__main__':
    setup(20261102);guard();entries=inventory();rows=[]
    for e in entries:
        m=initialize(e);tmp=D/'initial'/e['id'];raw=export(m,tmp);m.qat=True
        a=np.asarray(Image.open(ROOT/'output/b1-color-cycle-negative-20261027/rule-preview/outdoor_source.png').convert('RGB'))[240:400,320:480]
        x=tx(a[None]);teacher=Teacher(e);target=teacher(x)
        with torch.no_grad():out=m(x);assert torch.equal(PackedEngine(tmp)(x),out)
        row=dict(id=e['id'],initial_teacher_mae=float((out-target).abs().mean()*128),third_residual_pw_diagonal_only=None);rows.append(row)
        print('INITIAL',e['id'],round(row['initial_teacher_mae'],3),flush=True)
        del m,teacher
    save_json(D/'initialization-probe.json',dict(styles=rows))
