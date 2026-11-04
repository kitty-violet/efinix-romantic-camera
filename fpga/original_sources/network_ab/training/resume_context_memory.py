"""Same bound FP32 update, returning idle CUDA cache after native-size steps."""
import argparse,gc,json,time
from pathlib import Path
import torch
from .train_context_regions import Training,ROOT,RUN
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train']);p.add_argument('--resume',type=Path,required=True);p.add_argument('--stop',type=int,default=2000);a=p.parse_args()
    out=ROOT/'runs'/RUN/'context-fp32-memory';state=torch.load(a.resume,map_location='cpu',weights_only=True)
    assert state['mode']=='fp32' and state['scales'] is None and state['step']>0
    assert all(file_hash(k)==v for k,v in state['binding']['code'].items())
    if a.action=='prepare':
        out.mkdir(exist_ok=False);source=a.resume.parent
        for name in ['config.json','data.json','graph.json','preflight.json']:(out/name).write_bytes((source/name).read_bytes())
        binding=json.loads((source/'binding.json').read_text());assert binding==state['binding'];binding['code'][str(Path(__file__))]=file_hash(__file__)
        write_json(out/'binding.json',binding);write_json(out/'migration.json',dict(source=str(a.resume),sha256=file_hash(a.resume),step=state['step'],
            numerical_update_unchanged=True,reason='Return unused cache after full720 updates to reduce WDDM paging',wrapper_sha256=file_hash(__file__)))
        print('MEMORY_RESUME_PREPARED',state['step'],flush=True);return
    configure_workspace();contract_guard();set_reproducible(2026100432);tr=Training(out)
    expected=dict(state['binding']);expected['code']=dict(expected['code']);expected['code'][str(Path(__file__))]=file_hash(__file__)
    assert expected==tr.binding
    # Explicit, recorded code-only migration. We do not reset model, Adam,
    # sampling position, LR step, or CPU/CUDA random state.
    state['binding']=tr.binding;tr.restore(state);del state;gc.collect();torch.cuda.empty_cache()
    write_json(out/f'launch_{tr.step:06d}.json',dict(checkpoint=str(a.resume),checkpoint_sha256=file_hash(a.resume),step=tr.step,binding=tr.binding))
    try:
        with (out/'fp32_metrics.jsonl').open('a',encoding='utf-8',buffering=1) as f:
            while tr.step<min(a.stop,tr.cfg['fp32_steps']):
                torch.cuda.reset_peak_memory_stats();row=tr.update();row.update(allocated=torch.cuda.memory_allocated(),reserved_before=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated())
                if row['native']:gc.collect();torch.cuda.empty_cache()
                row['reserved_after']=torch.cuda.memory_reserved();f.write(json.dumps(row)+'\n')
                if tr.step%50==0:print('CONTEXT_MEMORY',tr.step,round(row['seconds'],3),row['reserved_before'],row['reserved_after'],flush=True)
                if tr.step%500==0 or tr.step==a.stop:
                    torch.save(tr.state(),out/f'fp32_checkpoint_{tr.step:06d}.pt');tr.evaluate();gc.collect();torch.cuda.empty_cache()
                if (out/'stop_after_update').exists():
                    torch.save(tr.state(),out/f'fp32_checkpoint_{tr.step:06d}.pt');print('CONTEXT_SAVED_STOP',tr.step,flush=True);return
    except KeyboardInterrupt:
        torch.save(tr.state(),out/f'interrupted_checkpoint_{tr.step:06d}.pt');raise
    print('CONTEXT_MEMORY_DONE',tr.step,flush=True)


if __name__=='__main__':main()
