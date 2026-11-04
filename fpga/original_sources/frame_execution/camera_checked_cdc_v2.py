"""Camera held-data proof plus the three surviving legacy probe crossings.

Only the unobserved payload cone was removed. Its reset pair and synchronized
done bit still feed the existing observer and must remain explicitly audited.
"""
import inspect
from audit_snapshot_cells import cells
from probe_full_native_split_map import params
from run_units import sha
import camera_checked_cdc as prior


def retained_probe(board):
    net=board/'outflow/ti60f225_oob.map.v';text=net.read_text();cc=cells(text);ps=params(text)
    resets=['cell_probe/ddr_cells/release_pipe[0]~FF','cell_probe/ddr_probe[55]~FF']
    previous="1'b1"
    for name in resets:
        k,p,a=cc[name]
        assert k=='EFX_FF' and p['CLK']=='sdram_clk' and p['D']==previous
        assert p['CE']=="1'b1" and p['SR']=='w_arstn' and 'async_reg=' in a
        assert all(ps[name][x]==(1,v) for x,v in dict(SR_SYNC=0,SR_POLARITY=0,SR_VALUE=0,D_POLARITY=1,CE_POLARITY=1).items())
        previous=p['Q']
    names=[f'cell_probe/done_sync[{i}]~FF' for i in range(3)]
    for i,name in enumerate(names):
        k,p,a=cc[name]
        assert k=='EFX_FF' and p['CLK']=='process_clk_150' and 'async_reg=' in a
        assert p['CE']=="1'b1" and p['SR']=='w_arstn'
        assert ps[name]['SR_SYNC']==(1,1) and ps[name]['SR_VALUE']==(1,0)
        if i:assert p['D']==cc[names[i-1]][1]['Q'] and ps[name]['D_POLARITY']==(1,1)
    # The done source is still generated in DDR and reset by the local pair.
    source='cell_probe/ddr_probe[48]~FF';k,p,a=cc[source]
    assert k=='EFX_FF' and p['CLK']=='sdram_clk' and p['SR']==previous
    return dict(status='RETAINED_PROBE_RESET_AND_DONE_SYNC_PASS',reset_endpoints=[n+'|SR' for n in resets],
                done_endpoint=names[0]+'|D',done_start=source+'|CLK',three_done_stages=names,map_sha256=sha(net))


def inventory(board):
    structure=retained_probe(board)
    source=inspect.getsource(prior.inventory)
    needle='scope=dict(old.__dict__);exec('
    assert source.count(needle)==1
    injection='''before="if re.fullmatch(r'clock_probe/meta\\\\[[0-5]\\\\]~FF\\\\|D', end):"
 after="""if end == retained['done_endpoint']:
                assert launch=='sdram_clk' and capture=='process_clk_150' and start==retained['done_start']
                kind='retained_probe_done_first_stage'
            elif end in retained['reset_endpoints']:
                assert launch=='process_clk_150' and capture=='sdram_clk' and start=='startup_reset/released~FF|CLK'
                kind='retained_probe_reset_pair'
            el"""+before
 assert source.count(before)==1;source=source.replace(before,after)
 source=source.replace("'clock_probe_first_stage')", "'clock_probe_first_stage','retained_probe_done_first_stage','retained_probe_reset_pair')")
 scope=dict(old.__dict__,retained=retained);exec('''
    source=source.replace(needle,injection)
    env=dict(prior.__dict__,retained=structure)
    exec(compile(source,__file__+':inventory','exec'),env)
    result=env['inventory'](board)
    assert result['classes']['retained_probe_done_first_stage']==1
    assert result['classes']['retained_probe_reset_pair']==2
    result.update(retained_probe_structure=structure,observer_auditor_sha256=sha(__file__),camera_auditor_sha256=sha(prior.__file__))
    return result


map_structure=prior.map_structure
