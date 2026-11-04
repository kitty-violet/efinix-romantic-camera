"""Place seven public Gray source FFs near the prior legal sync-head sites.

Physical assignments only: preserve RTL, clock relationships, and sync chains.
Protected synchronizer internals are not assignable mapped primitives; pinning
them crashed Efinity 2026.1.132. Head placement must remain tool-controlled.
"""
import json,re
from pathlib import Path
from run_units import ROOT, sha


def generate(root, reference, sources_only=False):
    assert sources_only,'Do not pin protected synchronizer internals: Efinity placement crash reproduced'
    reference=Path(reference).resolve()
    assert reference.is_relative_to(ROOT/'tmp')
    meta=json.loads((reference/'physical_inputs.json').read_text())
    terminal=json.loads((reference/'result.json').read_text())
    assert any(s['flow']=='pnr' and s['exit_code']==0 for s in terminal['stages'])
    assert sha(root/'outflow/ti60f225_oob.map.v')==meta['base_map_sha256']
    assert sha(reference/'outflow/ti60f225_oob.map.v')==meta['base_map_sha256']
    for row in meta['inputs']:
        assert sha(row['snapshot'])==row['sha256']
    placement=reference/'outflow/ti60f225_oob.out.placeloc'
    cells={}
    for line in placement.read_text().splitlines():
        words=line.split()
        if len(words)==4 and all(x.isdigit() for x in words[1:]):
            assert words[0] not in cells
            cells[words[0]]=tuple(map(int,words[1:]))
    ff_sites={xyz for name,xyz in cells.items() if name.endswith('~FF')}
    prefix=('u_ddr3_top/controller/u_efx_ddr3_soft_controller/genblk1.inst_ddr3_axi/'
            'inst_efx_ddr3/top_mc/controller_main/bram_u2/u_efx_fifo_top/xefx_fifo_ctl/async_clk.')
    names=[(prefix+f'waddr_cntr_gry_r[{i}]~FF',
            prefix+f'wr2rd_addr_sync/async_reg[0][{i}]~FF') for i in range(7)]
    net=(root/'outflow/ti60f225_oob.map.v').read_text()
    public_ff=set(re.findall(r'^\s*EFX_FF\s+\\(\S+)\s',net,re.M))
    assert all(source in public_ff for source,_ in names),'Only explicitly mapped FF primitives are assignable'
    assert all(head not in public_ff for _,head in names),'Review changed protected-model boundary'
    occupied={cells[head] for _,head in names}
    assert len(occupied)==7 and all(cells[n][2]==2 for pair in names for n in pair)
    rows=[]
    for bit,(source,head) in enumerate(names):
        x,y,sub=cells[head]
        candidates=[(x,y-1,sub),(x,y+1,sub),(x-1,y,sub),(x+1,y,sub)]
        candidates=[pos for pos in candidates if pos in ff_sites and pos not in occupied]
        assert candidates,('No proven adjacent FF site',bit,cells[head])
        target=candidates[0];occupied.add(target)
        old=cells[source]
        assert abs(old[0]-x)+abs(old[1]-y)>1,('Already adjacent',source)
        rows.append(dict(bit=bit,source=source,head=head,old_source=old,
                         source_location=target,head_location=cells[head]))
    assignment=root/'fixed_dsp.placeloc'
    assert assignment.exists(),'Use only with the established fixed DSP layout'
    original=assignment.read_text()
    assert not any(name in original for pair in names for name in pair)
    requested=(('source','source_location'),) if sources_only else (('source','source_location'),('head','head_location'))
    extra=''.join(f"{row[key]} {' '.join(map(str,row[loc]))}\n"
                  for row in rows for key,loc in requested)
    assignment.write_text(original+extra)
    result=dict(scope=f'{7 if sources_only else 14} FF location assignments only; no RTL, SDC, frequency, or pin edits',
        sources_only=sources_only,actual_head_locations_must_be_verified=True,
        reference=str(reference),reference_map_sha256=meta['base_map_sha256'],
        reference_manifest_sha256=sha(reference/'physical_inputs.json'),
        reference_placement_sha256=sha(placement),assignments_sha256=sha(assignment),
        generator_sha256=sha(__file__),pairs=rows,strict_sta_required=True,board_release=False)
    (root/'related_gray_placement.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'related_gray_generator.py').write_bytes(Path(__file__).read_bytes())
    return result
