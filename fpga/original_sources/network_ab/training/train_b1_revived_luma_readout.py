"""Continue the same14Y/chroma objective after exact-RGB path revival."""
from pathlib import Path
import torch
from .train_b1_trainable_luma_readout import TrainableLumaReadoutTraining,driver,evaluate,preview


class RevivedLumaReadoutTraining(TrainableLumaReadoutTraining):
    def __init__(self,args,data):
        warm=torch.load(args.warm,map_location='cpu',weights_only=True)
        assert warm.get('revived_reserved_luma_features')==[30,31]
        super().__init__(args,data)

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        warm=torch.load(self.args.warm,map_location='cpu',weights_only=True)
        for key in ('revived_reserved_luma_features','duplicated_source_channel','source_channel_chosen_on_train_ids',
                    'revival_readout_weights','initial_rgb_exactly_preserved'):
            state[key]=warm[key]
        torch.save(state,path)

    def preflight_extras(self):
        result=super().preflight_extras();live={}
        for key in ('res3_conv1__pw','res3_conv2__dw','res3_conv2__pw','up1_low_dw'):
            w=self.net.weight[key];moment=self.opt.state[w]['exp_avg'][30:]
            value=moment.flatten(1).norm(dim=1)
            assert torch.isfinite(value).all() and (value>0).all(),key
            live[key]=value.tolist()
        result['revived_paths_real_objective_optimizer_moments']=live
        return result


if __name__=='__main__':
    torch.set_num_threads(2)
    driver.main(training_factory=RevivedLumaReadoutTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
