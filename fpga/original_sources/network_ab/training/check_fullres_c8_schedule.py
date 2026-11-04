"""Static use of existing finite allocator for a research-only alternative graph.

No shared producer is edited, no program is submitted to hardware, and passing
allocation is not a numerical/throughput/physical-implementation pass.
"""
import json
from pathlib import Path
import shutil
import sys
from collections import Counter
from .runtime import configure_workspace, file_hash, write_json

ROOT=Path('E:/fpga')
HW=ROOT/'hw/efinity/prototypes/frame_execution'
sys.path.insert(0,str(HW))
from fourway_frame_program import compile_tile
from resource_route_contract import check_fields


def make_graph():
    specs=[('stem','conv3x3',3,8)]+[(f'{k}{i}',kind,8,8) for i in range(4)
        for k,kind in [('dw','dw3x3'),('pw','pw1x1')]]+[('rgb','pw1x1',8,3)]
    nodes=[];previous='input'
    for name,kind,cin,cout in specs:
        nodes.append(dict(id=name,kind=kind,inputs=[previous],input_shape_hwc=[720,1280,cin],
                          output_shape_hwc=[720,1280,cout],input_channels=cin,stride=1))
        previous=name
    return dict(nodes=nodes)


def tile(plan,index,start,width):
    end=min(start+width,1280)-1;a,b=start,end;ranges={}
    for node in reversed(plan['nodes']):
        ranges[node['id']]=dict(x0=a,x1=b)
        if node['kind'] in ('dw3x3','conv3x3'):a,b=max(0,a-1),min(1279,b+1)
    # RGB DMA address must be 4-byte aligned, keep the extra halo in its row.
    a=(a//4)*4
    return dict(tile_index=index,commit_interval=dict(x0=start,x1=end),source_interval=dict(x0=a,x1=b),
                steps=[dict(operation=n['id'],output_interval=ranges[n['id']]) for n in plan['nodes']])


def check_command(q):
    f=[0]*32
    if q['kind']!='compute':
        f[0]=int(q['kind']=='store');f[6]=int(q['channels']==3)
        f[7]=(q['row']['y']*q['global_width']+q['row']['origin'])*q['channels']
    else:
        f[0]=2;f[3]=q['rows'];f[6]=int(q['operator']=='pw1x1')|int(q['operator']=='conv3x3')<<2
        assert 0<q['pixels']<=320 and q['input_groups']==q['output_groups']==1
        assert q['source_width']==1280 and not q['nn'] and not q['stride2'] and not q['skip']
        bases=sum(d['base']<<(10*i) for i,d in enumerate(q['inputs']))
        f[10]=sum(d['plane']<<(2*i) for i,d in enumerate(q['inputs']))
        f[11]=bases&0xffffffff;f[12]=bases>>32;f[28]=q['mask']
    return check_fields(f)


def main():
    root=configure_workspace();out=root/'runs/sampling-hardware-audit-20261003/fullres-c8-probe'
    assert (out/'result.json').exists() and not (out/'allocation.json').exists()
    sources=[HW/'fourway_frame_program.py',HW/'compile_frame.py',HW/'resource_route_contract.py',Path(__file__)]
    hashes={str(p):file_hash(p) for p in sources}
    dst=out/'allocator-source';dst.mkdir()
    for p in sources:shutil.copy2(p,dst/p.name)
    plan=make_graph();rejected=[];parts=None
    for width in (320,256,160,128,80,64):
        try:
            candidate=[]
            for i,x in enumerate(range(0,1280,width)):
                t=tile(plan,i,x,width)
                result=compile_tile(plan,dict(name='research_fullres'),t,0,4,aligned=True,
                                    plane_shift=True,prefetch_rows=4,retain_commit=True,compact_lifetime=True)
                for q in result['commands']:check_command(q)
                candidate.append(result)
            parts=candidate;break
        except (MemoryError,AssertionError) as exc:
            rejected.append(dict(width=width,exception=type(exc).__name__,reason=str(exc)))
    if parts is None:
        write_json(out/'allocation.json',dict(status='REJECTED_BY_EXISTING_ALLOCATOR',rejected=rejected,source_hashes=hashes))
        print('ALLOCATION_REJECTED',json.dumps(rejected),flush=True);return
    commands=Counter();macs=0;dma_bytes=0;row_maps=set()
    byid={n['id']:n for n in plan['nodes']}
    for p in parts:
        for q in p['commands']:
            commands[q['kind']]+=1
            row_maps.add(check_command(q))
            if q['kind']=='compute':
                n=byid[q['node']];terms=27 if n['kind']=='conv3x3' else 9 if n['kind']=='dw3x3' else 8
                macs+=q['pixels']*q['rows']*n['output_shape_hwc'][2]*terms
            else:dma_bytes+=q['row']['width']*q['channels']
    write_json(out/'research_schedule.json',dict(architecture='fullres-c8-ds4',plan=plan,tiles=parts,
                                                production=False,rtl_executed=False))
    result=dict(status='STATIC_ALLOCATOR_AND_ROUTE_CHECKS_PASS_NOT_RTL_PASS',output_tile_width=width,
                tiles=len(parts),rejected=rejected,commands=dict(commands),
                bank_high_words=[max(p['finite_high_water'][i] for p in parts) for i in range(3)],
                bank_depth_words=1024,physical_banks=12,word_bits=64,
                peak_live_padded_bytes=max(p['peak_live_padded_bytes'] for p in parts),
                scheduled_macs_with_halo=macs,scheduled_rgb_dma_payload_bytes=dma_bytes,
                scheduled_command_bytes=sum(commands.values())*128,
                route_maps=len(row_maps-{None}),
                source_hashes=hashes,schedule_sha256=file_hash(out/'research_schedule.json'),
                limitations=[
                    'Same allocator and route contract, on a new research graph. Production graph remains B1.',
                    'No trained coefficient addresses, integer golden, or RTL command execution yet.',
                    'DMA payload excludes command/coefficients, burst alignment, RAW, display, and gesture.',
                    'No proof of actual overlap, bank service cycles, DDR throughput, Fmax, or resource fit.'])
    assert all(file_hash(p)==h for p,h in hashes.items())
    write_json(out/'allocation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
