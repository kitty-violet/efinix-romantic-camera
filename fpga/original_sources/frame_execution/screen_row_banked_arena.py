"""Screen a six-row-bank alternative using the complete graph's lifetimes.

Six 2048x64 dual-port banks have the same 98,304 byte capacity as the
existing twelve 1024x64 banks. Port A supplies one column per distinct row;
port B must arbitrate compute writes, skip reads and DMA. RGB pairs/stride
pairs need two words of a row and are explicitly charged as two reads.
No cycle result, RTL fit or relaxed memory-port assumption is produced here.
"""
import argparse
import collections
import json
from pathlib import Path
from fourway_frame_program import graph, row_tasks, hw
from model_contract import compile_contract, load_contract
from run_units import ROOT, sha


def place(records, depth):
    neighbors = {key: set() for key in records}
    active = [set() for _ in range(6)]
    for key, rec in sorted(records.items(), key=lambda kv: kv[1]['born']):
        bank = rec['bank']
        active[bank] = {k for k in active[bank] if records[k]['last'] >= rec['born']}
        neighbors[key].update(active[bank])
        for k in active[bank]:
            neighbors[k].add(key)
        active[bank].add(key)
    best = None
    for order in (lambda k: (-records[k]['length'], records[k]['born'], k),
                  lambda k: (-(records[k]['last']-records[k]['born']), -records[k]['length'], k),
                  lambda k: (records[k]['born'], k)):
        positions = {}
        high = [0]*6
        for key in sorted(records, key=order):
            rec = records[key]
            at = 0
            for lo, hi in sorted((positions[k], positions[k]+records[k]['length'])
                                 for k in neighbors[key] if k in positions):
                if lo-at >= rec['length']:
                    break
                at = max(at, hi)
            positions[key] = at
            high[rec['bank']] = max(high[rec['bank']], at+rec['length'])
        if best is None or max(high) < max(best[0]):
            best = high, positions
    high, positions = best
    # Independently recheck every co-live pair, not just the allocator's scan.
    for key, peers in neighbors.items():
        for peer in peers:
            a, b = positions[key], positions[peer]
            assert a+records[key]['length'] <= b or b+records[peer]['length'] <= a
    return high, max(high) <= depth


def one(plan, group, tile, width, offset_step, packed_rgb):
    local, ranges, final = graph(plan, group, tile, -1 if group['name']=='residual' else 0)
    tasks = row_tasks(local, final, 4, group['name']!='residual',
                      2 if group['name']=='residual' else 4, True)
    last = {tuple(key): task['id'] for task in tasks for key in task['dependencies']}
    if group['name']!='residual':
        commits = [t for t in tasks if t['node']=='commit']
        for old, new in zip(commits, commits[1:]):
            for key in old['dependencies']:
                last[tuple(key)] = new['id']
    offset = {'source': 0}
    for name, node in local.items():
        if name!='source':
            offset[name] = (offset[node['parents'][0]]+
                            (offset_step if node['kind']!='nearest_x2' else 0))%6
    records = {}
    for task in tasks:
        for name, y in task['outputs']:
            channels = local[name]['output_shape_hwc'][2]
            row_width = ranges[name][1]-ranges[name][0]+1
            length = (row_width+1)//2 if packed_rgb and channels==3 else row_width*((channels+7)//8)
            records[(name,y)] = dict(bank=(y+offset[name])%6, born=task['id'],
                                     last=last[(name,y)], length=length)
    high, fits = place(records, 2048)
    # A logical operand may repeat a clamped/NN row. Those are aliases, not
    # additional reads. Different physical rows in one bank are real conflicts.
    read_conflicts = output_conflicts = pair_read_commands = 0
    compute_commands = source_words = commit_words = 0
    def physical(name, y):
        n = local[name]
        y = max(0, min(n['output_shape_hwc'][0]-1, y))
        return physical(n['parents'][0], y//2) if n['kind']=='nearest_x2' else (name,y)
    for task in tasks:
        name = task['node']
        if name in ('source','commit'):
            word_count = sum(records[tuple(k)]['length'] for k in
                             (task['outputs'] if name=='source' else task['dependencies']))
            if name=='source': source_words += word_count
            else: commit_words += word_count
            continue
        n = local[name]
        stride = n.get('stride',1)
        max_rows = 2 if stride==2 else 4
        max_width = 160 if stride==2 else 320
        for y in range(task['lo'],task['hi']+1,max_rows):
            nr = min(max_rows,task['hi']-y+1)
            parent = n['parents'][0]
            yy = range(y*stride-1, y*stride-1+(nr-1)*stride+3) if n['kind'] in ('dw3x3','conv3x3') else range(y,y+nr)
            source = set(physical(parent,z) for z in yy)
            dest = {(name,z) for z in range(y,y+nr)}
            r = collections.Counter(records[k]['bank'] for k in source)
            w = collections.Counter(records[k]['bank'] for k in dest)
            count = (ranges[name][1]-ranges[name][0]+max_width)//max_width
            compute_commands += count
            read_conflicts += int(max(r.values())>1)*count
            output_conflicts += int(max(w.values())>1)*count
            if stride==2 or n['output_shape_hwc'][2]==3:
                pair_read_commands += count
    return dict(group=group['name'], tile=tile['tile_index'], width=width,
                offset_step=offset_step, packed_rgb=packed_rgb,
                rows=len(records), high_water=high, capacity_pass=fits,
                main_read_row_conflicts=read_conflicts, output_row_conflicts=output_conflicts,
                pair_input_commands_requiring_second_word=pair_read_commands,
                compute_geometry_commands=compute_commands,
                source_words=source_words,commit_words=commit_words)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    plan=compile_contract(load_contract());cases=[]
    for group_name,widths in [('encoder',(256,160)),('residual',(80,64)),('decoder',(256,160))]:
        for width in widths:
            group=next(g for g in hw.build_tile_manifest(width)['groups'] if g['name']==group_name)
            if group_name=='residual':
                group=dict(group,tiles=[dict(tile_index=i,commit_interval=dict(x0=x,x1=min(x+width,320)-1))
                                       for i,x in enumerate(range(0,320,width))])
            for offset in (1,2,3):
                for packed in (False,True):
                    rows=[one(plan,group,t,width,offset,packed) for t in group['tiles']]
                    cases.append(dict(group=group_name,width=width,offset_step=offset,packed_rgb=packed,
                                      capacity_pass=all(r['capacity_pass'] for r in rows),
                                      high_water=[max(r['high_water'][b] for r in rows) for b in range(6)],
                                      main_read_row_conflicts=sum(r['main_read_row_conflicts'] for r in rows),
                                      output_row_conflicts=sum(r['output_row_conflicts'] for r in rows),
                                      pair_input_commands=sum(r['pair_input_commands_requiring_second_word'] for r in rows),
                                      tiles=rows))
                    print({k:v for k,v in cases[-1].items() if k!='tiles'},flush=True)
    result=dict(status='SIX_BANK_LIFETIME_SCREEN_NOT_RTL_OR_TIMING',cases=cases,
                bytes=6*2048*8,primitive_ram10=6*2*7,
                requires_new_descriptor_abi=True,requires_new_physical_dependency_reorder=True,
                issues_not_assumed_free=['auxiliary port arbitration for DMA/skip/write',
                  'second-word pair input including RGB alignment',
                  'clocked RAM throughput and retained responses',
                  'cancel/drain recovery and final-frame publication'],
                contract_sha256=sha(ROOT/'network_ab/model_b_contract.json'),script_sha256=sha(__file__),
                reference_program_sha256=sha(ROOT/'tmp/frame-execution/joint-reorder-vectors1/program.json'),
                full_frame_cycles=None,whole_board_xlr=None)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
