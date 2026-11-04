"""Account for the retained dense convolution work without touching RTL."""
import argparse
import json
from .importance_pruned_teacher import RUN, parent, layer_groups, WIDTHS
from .runtime import configure_workspace, write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',required=True);args=ap.parse_args()
    root=configure_workspace();run=root/'runs'/RUN
    plan=json.loads((run/args.candidate/'plan.json').read_text())['plan']
    spatial=json.loads((run/'spatial.json').read_text());base=parent(root);rows=[]
    for name,(ig,og) in layer_groups().items():
        m=base.get_submodule(name)
        ci=len(plan[ig]) if ig else m.in_channels;co=len(plan[og]) if og else m.out_channels
        weights=ci*co*m.kernel_size[0]*m.kernel_size[1]
        rows.append(dict(name=name,input_group=ig,output_group=og,in_channels=ci,out_channels=co,
                         kernel=list(m.kernel_size),output_positions=spatial[name],
                         parameters=weights+(co if m.bias is not None else 0),
                         conv_macs=weights*spatial[name]))
    total=sum(r['conv_macs'] for r in rows);params=sum(r['parameters'] for r in rows)
    rows.sort(key=lambda r:r['conv_macs'],reverse=True)
    for r in rows:r['fraction']=r['conv_macs']/total
    # The independently executed GPU hook profile will be the delivery authority.
    write_json(run/args.candidate/'cost_breakdown.json',dict(candidate=args.candidate,conv_macs=total,
        parameters=params,widths={g:len(plan[g]) for g in WIDTHS},layers=rows,
        scope='Dense mathematical convolution MAC only. Excludes normalization, softmax, resizing, memory traffic, padding of hardware lanes and timing.'))
    print(json.dumps(dict(parameters=params,conv_macs=total,top5=rows[:5]),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
