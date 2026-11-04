"""Freeze verified coefficient-only weak-edge merging for twelve colour styles."""
import shutil,json,time
import library as l
from library import D,torch,save_json,sha
from refine import load_packed
from train import prepare_inputs,targets,cut

def main():
    l.setup(20261102);inputs,nat=prepare_inputs();V=inputs[1]
    entries=[e for e in l.read_inventory() if e['family'] not in ['legacy','legacy_fp32']]
    assert len(entries)==12;rows=[]
    for e in entries:
        folder=D/'models'/e['id'];candidate=D/'clipped-merge-candidates/12'/e['id'];analysis=json.loads((candidate/'analysis.json').read_text('utf-8'))
        assert analysis['uniform_rgb64_max_channel_error_vs_old']==0,e['id']
        assert all(x['metrics']['clipped']['tiny_cells_fraction']<=x['metrics']['base']['tiny_cells_fraction']+.0002 for x in analysis['records'])
        report=json.loads((folder/'training.json').read_text('utf-8'))
        if report.get('local_clipped_merge',{}).get('threshold')==12:
            print('CLIPPED_ADOPTION_ALREADY_DONE',e['id'],flush=True);continue
        backup=folder/'before-clipped-merge';backup.mkdir(exist_ok=True)
        for name in ['training.json','selected.pt','coefficients.bin','coefficients-padded.bin','load_words.hex','graph.json','layout.json','reference-vectors.json','integer-coefficients.npz']:shutil.copy2(folder/name,backup/name)
        old_sha=sha(folder/'coefficients-padded.bin');m=load_packed(candidate);m.eval();l.export(m,folder)
        assert sha(folder/'coefficients-padded.bin')==sha(candidate/'coefficients-padded.bin')
        oracle=l.independent_oracle(folder);teacher=l.Teacher(e);W,_=targets(teacher,V,False)
        with torch.inference_mode():
            outputs=torch.cat([cut(m(x)) for x in V.split(4)]);error=(outputs-cut(W)).abs().mean((1,2,3))*128
            natural=float(error[:nat[1]].mean());palette=float(error[nat[1]:].mean())
            engine=l.PackedEngine(folder)
            for x in V.split(4):assert torch.equal(m(x),engine(x))
        torch.save(dict(model=m.state_dict(),residual=m.residual,qat=True,style_id=e['id'],teacher_sha256=e['source_sha256'],post_training_coefficient_construction=True,constructed_weak_edge_merging=True,additional_gradient_steps=0),folder/'selected.pt')
        report['pre_clipped_validation_score']=report['selected_score']
        report.update(local_clipped_merge=dict(threshold=12,latent_kernel=[[0,.125,0],[.125,.5,.125],[0,.125,0]],formula='mean + ReLU(center-mean-T) - ReLU(mean-center-T)',within_existing_operators=True,post_training_coefficient_construction=True,additional_gradient_steps=0,uniform_rgb64_max_channel_error_vs_original_teacher=0,previous_coefficients_sha256=old_sha,diagnostic_midframe_records=analysis['records']),selected_score=natural+.25*palette,selected_natural_mae=natural,selected_palette_mae=palette,selected_score_target='old frozen teacher on original validation inputs after deliberate weak-edge merging; not the previous teacher-simplified objective',coefficient_sha256=sha(folder/'coefficients-padded.bin'),independent_oracle=oracle,authoritative_checkpoint='selected.pt',no_inference_filter=True)
        save_json(folder/'training.json',report);rows.append(dict(id=e['id'],validation_natural_mae=natural,uniform_rgb64_max_error=0,coefficient_sha256=report['coefficient_sha256']))
        print('FINAL_CLIPPED_MODEL_FROZEN',e['id'],'MAE',round(natural,3),flush=True)
        del m,engine,teacher,W;torch.cuda.empty_cache()
    save_json(D/'clipped-merge-freeze.json',dict(status='PASS_ALL_12_PALETTES_AND_INTEGER_RULES',models=rows))
if __name__=='__main__':main()
