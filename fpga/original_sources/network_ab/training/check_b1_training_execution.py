"""Exercise checkpoint execution state, dependency capture and loss resolution."""
import copy
import json
from argparse import Namespace
from pathlib import Path
import torch
from . import train_b1_minimum_area as m
from .research_b1_reconstruction import Experiment
from .train_b1_aligned_decoder_regions import DecoderAlignedTraining, regression
from .train_b1_late_region_supervision import executed_sources
from .engine import set_reproducible
from .runtime import configure_workspace, write_json, contract_guard


def main():
    configure_workspace();set_reproducible(20261006291);contract_guard();torch.set_num_threads(2)
    dest=m.OUT/'complete-chain-audit';dest.mkdir(exist_ok=True)
    checkpoint=m.OUT/'native-boundary-field-probe/span5/candidate.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    # The CURRENT initializer happens to equal this checkpoint. Explicitly
    # make a supported alternative residual configuration as a negative
    # control; do not falsely attribute a mismatch to the current candidate.
    fixture=copy.deepcopy(state)
    fixture['residual_encoding']['res1_add']['multiplier']=65536
    fixture_path=dest/'alternative-residual-fixture.pt';torch.save(fixture,fixture_path)
    original_checkpoint=checkpoint;checkpoint=fixture_path;state=fixture
    args=Namespace(warm=str(checkpoint),seed=20261006291,init='identity',aux=0,perceptual=0,qat=True,
        scale_lr=0,lr=.00015,steps=20,shift=0,latent_refine=0,loss='mse',crop=320,batch=2,hard=False,rotate=False,colour=False)
    rows=m.load_data()['training'];tr=Experiment(args,rows)
    assert tr.net.package['residual_encoding']==state['residual_encoding']
    x=torch.randint(-128,128,(1,3,32,48),device='cuda').float()/128
    with torch.no_grad():
        expected=tr.net(x,all_nodes=True);blob=tr.net.export()
        fresh=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin').cuda()
        fresh.load_state_dict(state['model']);fresh.quantized=True
        stale=fresh(x,all_nodes=True)
        stale_error=max(float((stale[k]-v).abs().max()) for k,v in expected.items())
        fresh.package['residual_encoding']=copy.deepcopy(state['residual_encoding'])
        corrected=fresh(x,all_nodes=True)
        assert all(torch.equal(v,corrected[k]) for k,v in expected.items()) and fresh.export()==blob
    # Real GPU forward/backward and a save/restore after changing the metadata
    # in the fresh instance. Tensor-only replay must be observably insufficient.
    native=tr.update(True);crop=tr.update(False)
    path=dest/'residual-state-regression.pt';tr.save(path)
    restored=torch.load(path,map_location='cpu',weights_only=True)
    assert restored['residual_encoding']==state['residual_encoding']
    expected_row=tr.update(False);expected_state=copy.deepcopy(tr.net.state_dict())
    new=Experiment(args,rows)
    new.net.package['residual_encoding']=copy.deepcopy(fresh.__class__(revive=False,source=m.t.r.OUT/'initial_identity.bin').package['residual_encoding'])
    new.restore(path);assert new.net.package['residual_encoding']==state['residual_encoding']
    actual_row=new.update(False)
    error=max(float((v-expected_state[k]).abs().max()) for k,v in new.net.state_dict().items())
    print('EXECUTION_RESUME_DIAGNOSTIC',dict(error=error,samples_equal=actual_row['samples']==expected_row['samples'],
        stale_error=stale_error,residual=state['residual_encoding']),flush=True)
    assert error<1e-5 and actual_row['samples']==expected_row['samples'] and stale_error>0
    imported,resolution=executed_sources(DecoderAlignedTraining)
    names={p.name for p in imported}
    assert {'train_b1_quantum_regions.py','train_b1_binary_regions.py','coherent_shared_model.py'}<=names
    assert resolution['methods']['additional_loss']['name']=='NativeContrastTraining.additional_loss'
    assert resolution['methods']['appearance_loss']['name']=='CountRegionTraining.appearance_loss'
    del tr,new,fresh,expected_state;torch.cuda.empty_cache()
    regression(checkpoint)
    report=dict(status='PASS',tensor_only_stale_node_error=stale_error,reloaded_all_node_error=0,
        fresh_resume_error=error,bytes_equal_before_update=True,native=native,crop=crop,
        imported_source_count=len(imported),execution=resolution,
        scope='Alternative-metadata regression fixture, not a claimed error in the current candidate; current initializer metadata matches candidate',
        original_checkpoint=str(original_checkpoint))
    write_json(dest/'code_execution_checks.json',report)
    print('TRAINING_EXECUTION_CHECKS_PASS',stale_error,error,len(imported),flush=True)


if __name__=='__main__':main()
