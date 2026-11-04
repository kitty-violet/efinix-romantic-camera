"""Check the frozen accepted package and six full-resolution known answers."""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .b1_trained_image import decode, runtime_from_image
from .probe_b1_cartoon_int8 import IntegerB1
from .train_b1_learned_colour import ColourB1
from .runtime import contract_guard, file_hash, write_json
from .b1_package_oracle import oracle_records
from network_ab.contract_reference import execute_integer


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--report',type=Path,required=True)
    a=ap.parse_args();root=a.root;torch.set_num_threads(2);contract,plan=contract_guard()
    acceptance=json.loads((root/'acceptance.json').read_text('utf-8'))
    assert plan['contract_sha256']==acceptance['canonical_contract_sha256']
    path=root/'coefficients-padded.bin';raw=path.read_bytes();assert file_hash(path)==acceptance['package_sha256']
    params,residual=decode(raw);assert len(raw)==20480 and raw[:17792]==(root/'coefficients.bin').read_bytes()
    assert b''.join(int(s,16).to_bytes(8,'little') for s in (root/'load_words.hex').read_text('ascii').splitlines())==raw
    state=torch.load(root/'selected-A.pt',map_location='cpu',weights_only=True)
    model=ColourB1(revive=False,source=path);model.load_state_dict(state['model']);model.package['residual_encoding']=state['residual_encoding'];model.quantized=True
    assert model.export()==raw;model.to(a.device)
    actual=IntegerB1(runtime_from_image(raw),device=a.device);rows=[]
    for item in json.loads((root/'validation/data.json').read_text('utf-8')):
        src=np.array(Image.open(root/item['source']).convert('RGB'));ref=np.array(Image.open(root/item['output']).convert('RGB'))
        assert src.shape==(720,1280,3)
        x=torch.tensor(src.transpose(2,0,1)[None].copy(),device=a.device,dtype=torch.float32)-128
        nodes,_=actual(x,True);trained=model(x/128,all_nodes=True)
        node_error=max(float((nodes[k]-v).abs().max()) for k,v in trained.items());assert node_error==0
        result=(nodes['rgb_head'][0].permute(1,2,0).cpu().numpy()+128).astype(np.uint8)
        err=int(np.abs(result.astype(np.int16)-ref.astype(np.int16)).max());assert err==0
        rows.append(dict(id=item['id'],node_error=node_error,image_error=err));print('ACCEPTED_A_CHECK',item['id'],err,flush=True)
    op,rp=oracle_records(params,residual);small=np.random.default_rng(81).integers(-128,128,(12,12,3),dtype=np.int16)
    reference=execute_integer(contract,small.tolist(),op,rp)
    got,_=actual(torch.tensor(small.transpose(2,0,1)[None].copy(),device=a.device,dtype=torch.float32),True)
    oracle_error=max(float(np.abs(v[0].permute(1,2,0).cpu().numpy()-np.array(reference['tensors'][k])).max()) for k,v in got.items());assert oracle_error==0
    for bad in (raw[:-1],raw[:1300]+bytes([raw[1300]^1])+raw[1301:],raw[:-1]+b'\x01'):
        try:decode(bad)
        except (ValueError,AssertionError):pass
        else:raise AssertionError('bad image accepted')
    write_json(a.report,dict(status='PASS',package_sha256=file_hash(path),device=a.device,records=rows,checkpoint_reexport_identical=True,oracle_error=oracle_error,invalid_images_rejected=True,actual_board=False))


if __name__=='__main__':main()
