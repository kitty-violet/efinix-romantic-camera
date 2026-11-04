"""Fit only the two internal chroma biases, using training images, after QAT."""
import copy
import torch
from . import train_b1_19_levels as n
from . import train_b1_learned_colour as c
from .runtime import configure_workspace,write_json,file_hash


@torch.inference_mode()
def measure(tr,rows):
    result=[]
    for item in rows:
        nodes=tr.net(item['source'][None],all_nodes=True)
        actual=nodes['up2_high_dw'][:,6:]*2;target=tr.targets[item['row']['id']][:,6:]
        diff=actual-target
        result.append(dict(id=item['row']['id'],bias=diff.mean((0,2,3)).tolist(),mae=float(diff.abs().mean())))
    return result


@torch.inference_mode()
def main():
    configure_workspace();data,selected,cache=c.load_data();out=n.OUT/'chroma-calibration';out.mkdir(exist_ok=False)
    parent=n.OUT/'qat/checkpoint_001000.pt';tr=n.Trainer(cache,selected,1000,True);tr.restore(parent)
    training=[r for r in cache if r['row']['split']=='train'];validation=[r for r in cache if r['row']['split']!='train']
    before=measure(tr,training);oldval=measure(tr,validation)
    correction=torch.tensor([r['bias'] for r in before],device='cuda').median(0).values
    assert correction.abs().max()<6,correction
    state=copy.deepcopy(tr.net.state_dict());tr.net.bias['up2_low_pw'][6:]-=correction
    for k,v in tr.net.state_dict().items():
        if k!='bias.up2_low_pw':assert torch.equal(v,state[k])
        else:assert torch.equal(v[:6],state[k][:6])
    after=measure(tr,training);newval=measure(tr,validation)
    accepted=sum(r['mae'] for r in after)<sum(r['mae'] for r in before) and all(b['mae']<=a['mae']+.1 for a,b in zip(oldval,newval))
    write_json(out/'calibration.json',dict(parent=str(parent),parent_sha256=file_hash(parent),
        fitted_on_train_only=True,train_images=len(training),bias_subtracted=correction.tolist(),
        unchanged_except='up2_low_pw bias channels6 and7',before_train=before,after_train=after,
        before_reviewed_validation=oldval,after_reviewed_validation=newval,
        numeric_chroma_improved=accepted,visual_accepted=False,training_steps_added=0,
        note='This is fitted internal coefficient calibration, not an external output filter; validation is previously reviewed, not blind.'))
    tr.save(out/'checkpoint_calibrated.pt');n.review(tr,data,selected,out/'package')
    print('CHROMA_CALIBRATED',correction.tolist(),'numeric_improved',accepted,
          'validation_mae_before',[r['mae'] for r in oldval],'after',[r['mae'] for r in newval],flush=True)


if __name__=='__main__':main()
