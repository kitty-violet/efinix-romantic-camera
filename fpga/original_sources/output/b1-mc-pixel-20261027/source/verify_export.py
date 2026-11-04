"""Export and independently verify all B1 integer nodes and container fields."""
import argparse,struct,zlib
from fractions import Fraction
from core import *
from network_ab.contract_reference import execute_integer


def oracle_records(parameters,residual,plan):
    """Recover consistent rational scale records from decoded residual Q17 fields."""
    scales={'input':Fraction(1,128)};params={};res={}
    ratio=lambda x:[x.numerator,x.denominator]
    for r in plan['nodes']:
        name=r['id'];si=scales[r['inputs'][0]]
        if r['kind']=='nearest_x2':scales[name]=si;continue
        if r['kind']=='residual_add':
            p=residual[name];skip=scales[r['inputs'][1]]
            common=skip if p['main_scaled'] else si
            so=common*Fraction(131072,p['post_multiplier'])
            res[name]=dict(main_scale=ratio(si),skip_scale=ratio(skip),common_scale=ratio(common),output_scale=ratio(so))
        else:
            p=parameters[name];so=si
            if name=='rgb_head':so=Fraction(1,128)
            for add in plan['nodes']:
                if add['kind']=='residual_add' and add['inputs'][0]==name:
                    rr=residual[add['id']];skip=scales[add['inputs'][1]]
                    so=skip*Fraction(rr['multiplier'],131072) if rr['main_scaled'] else skip*Fraction(131072,rr['multiplier'])
            params[name]=dict(weights=[list(v) for v in p.weights],bias=list(p.bias),multipliers=list(p.multiplier),input_scale=ratio(si),output_scale=ratio(so),weight_scales=[ratio(Fraction(m,131072)*so/si) for m in p.multiplier])
        scales[name]=so
    return params,res


def main(args):
    setup();disk_guard()
    dest=OUT/'exports'/args.name;dest.mkdir(parents=True,exist_ok=True)
    model=load_model(args.checkpoint);model.quantized=True;raw=model.export()
    params,residual=decode(raw);runtime=IntegerEngine(raw)
    (dest/'coefficients.bin').write_bytes(raw[:17792]);(dest/'coefficients-padded.bin').write_bytes(raw)
    (dest/'load_words.hex').write_text(''.join(f'{x:016x}\n' for x in struct.unpack('<2560Q',raw)),encoding='ascii')
    assert b''.join(int(s,16).to_bytes(8,'little') for s in (dest/'load_words.hex').read_text('ascii').splitlines())==raw
    save_json(dest/'residual_encoding.json',residual)
    contract=load_contract();plan=compile_contract(contract);op,rp=oracle_records(params,residual,plan)
    rng=np.random.default_rng(241024);tests=[]
    arrays=[rng.integers(-128,128,(8,12,3),dtype=np.int16),np.full((8,8,3),-128,np.int16),np.full((8,8,3),127,np.int16),np.zeros((12,16,3),np.int16)]
    sample=read(OUT/'targets/outdoor_source.png')
    arrays.extend([sample[300:312,480:496].astype(np.int16)-128,np.broadcast_to(np.arange(16,dtype=np.int16)[None,:,None]*17-128,(12,16,3)).copy()])
    scalars=0
    with torch.no_grad():
        for i,a in enumerate(arrays):
            x=torch.tensor(a.transpose(2,0,1)[None].astype(np.float32),device='cuda')
            refs=execute_integer(contract,a.tolist(),op,rp)['tensors']
            theirs,_=runtime(x,True);ours=model(x/128,True);errors={}
            for name,reference in refs.items():
                arr=np.array(reference,np.int64);got=theirs[name][0].permute(1,2,0).cpu().numpy().astype(np.int64)
                diff=int(np.abs(arr-got).max());quant_error=float((ours[name]-theirs[name]).abs().max())
                assert diff==0 and quant_error==0,(i,name,diff,quant_error)
                errors[name]=dict(stdlib_max_error=diff,qat_export_max_error=quant_error,scalar_count=arr.size);scalars+=arr.size
            tests.append(dict(id=i,shape=list(a.shape),nodes=errors))
            print('ORACLE_PASS',i,flush=True)
        x=tensor(sample)[None];nodes,sat=runtime(x*128,True);qat=model(x,True)
        full={k:dict(max_error=float((v-nodes[k]).abs().max()),shape=list(v.shape),saturation_fraction=sat.get(k)) for k,v in qat.items()}
        assert all(r['max_error']==0 for r in full.values())
        fp=model.quantized;model.quantized=False;y=model(x);model.quantized=fp
        intout=nodes['rgb_head']/128
        quant_diff=dict(mae_rgb255=float((y-intout).abs().mean()*128),max_rgb255=float((y-intout).abs().max()*128))
    contracts=dest/'contract';contracts.mkdir(exist_ok=True)
    accepted=WORK/'accepted/B1-A-20261008/contract'
    for p in accepted.iterdir():
        if p.is_file():shutil.copy2(p,contracts/p.name)
    shutil.copy2(args.checkpoint,dest/'selected.pt')
    fields={name:dict(main_scaled=p['main_scaled'],multiplier=p['multiplier'],post_multiplier=p['post_multiplier']) for name,p in residual.items()}
    manifest=dict(schema='pure-fixed-B1-offline-candidate-v1',status='SOFTWARE_INTEGER_VERIFIED_NOT_BOARD_ACCEPTED',selected_checkpoint_sha256=sha(dest/'selected.pt'),source_checkpoint=str(args.checkpoint),source_step=torch.load(args.checkpoint,map_location='cpu',weights_only=True)['step'],contract_sha256=plan['contract_sha256'],convolutions=24,nodes=29,internal_resolution=[1280,720],effective_bytes=17792,transport_bytes=len(raw),zero_padding_bytes=2688,zero_padding_valid=not any(raw[17792:]),crc32_padded=f'{zlib.crc32(raw):08x}',crc32_effective=f'{zlib.crc32(raw[:17792]):08x}',sha256_padded=sha(dest/'coefficients-padded.bin'),sha256_effective=sha(dest/'coefficients.bin'),residual_program_fields=fields,coefficient_codec_roundtrip=encode(params,residual)==raw[:17792],small_oracle_tests=tests,small_oracle_scalar_comparisons=scalars,full720_qat_vs_decoded_integer_nodes=full,fp32_vs_integer_diagnostic=quant_diff,production_program_modified=False,rtl_validation=False,board_validation=False,inference_extra_operators=False,disk=disk_guard())
    manifest.update(style_accepted=False,temporal_accepted=False,task_completed=False)
    save_json(dest/'verification.json',manifest)
    print('EXPORT_VERIFIED',args.name,'scalars',scalars,'CRC',manifest['crc32_padded'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--name',required=True);main(p.parse_args())
