"""Strict checkpoint / binary correspondence and independent integer vectors."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'network'))
import torch
from model import load_checkpoint,HalfAway
from verify_package import load,oracle
torch.set_num_threads(2)
def main():
 results=[]
 for i in range(5):
  folder=ROOT/'network/models'/f'slot{i}'
  graph,parameters=load(folder);model=load_checkpoint(folder)
  assert sum(p.numel() for p in model.parameters())==4035
  assert model.rows==graph['nodes'] and model.residual==graph['residual']
  checks=0
  for name,p in parameters.items():
   m=getattr(model,name+'_m');scale=(m/131072).float()
   w=HalfAway.apply(model.w[name]/scale[:,None,None,None]).clamp(-127,127).to(torch.int8).flatten().tolist()
   b=HalfAway.apply(model.b[name]/scale).to(torch.int32).tolist()
   assert w==list(p['weight']) and b==list(p['bias']) and m.to(torch.int32).tolist()==list(p['multiplier']),(i,name)
   checks+=len(w)+len(b)+len(p['multiplier'])
  outputs=0
  for case in json.loads((folder/'reference-vectors.json').read_text())['vectors']:
   expected=oracle(graph,parameters,case['input_u8']);assert expected==case['output_i8']
   x=torch.tensor(case['input_u8']).permute(2,0,1)[None].float()/128-1
   with torch.no_grad():y=model(x)*128
   got=y[0].permute(1,2,0).to(torch.int32).tolist()
   assert got==expected,(i,'checkpoint/reference mismatch');outputs+=y.numel()
  results.append({'slot':i,'parameters':4035,'coefficient_checks':checks,'exact_output_scalars':outputs,'strict_checkpoint_load':True})
 print(json.dumps({'status':'ALL_FIVE_CHECKPOINTS_AND_INTEGER_NETWORKS_MATCH','torch':torch.__version__,'styles':results},indent=2))
if __name__=='__main__':main()
