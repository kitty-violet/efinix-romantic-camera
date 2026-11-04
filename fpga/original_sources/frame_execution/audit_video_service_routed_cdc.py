"""Finite current-map service CDC endpoints and conservative held-bus settling.

Both mailbox directions wait through two actual receiving-clock FFs before
using the held word/selector on the third edge. The required bound is below
two destination periods, charging clock insertion difference, uncertainty,
setup and rounding against that budget. Async inter-PLL slack is NOT a
synchronous timing acceptance result. Reset release has its own gate.
"""
import collections, json, re
from run_units import sha


def inventory(root, structure_path):
    structure=json.loads(structure_path.read_text())
    assert structure['status']=='VIDEO_SERVICE_NEW_CDC_INVENTORIED_NOT_RELEASE'
    assert structure['map_sha256']==sha(root/'outflow/ti60f225_oob.map.v')
    assert not structure['unclassified'] and len(structure['two_stage_chains'])==10
    cross=structure['cross_domain_cones']; expected={}
    # Closed supported mailbox geometries. Establish width from the actual
    # mapped endpoints, not a user-selected assumption or an empty report.
    held_ends={r['end'] for r in cross if r['start'].startswith('service_pixel_snapshot/held[') and r['end'].startswith('service_pixel_cache[')}
    ram=structure.get('dualclock_sample_ram',[])
    if ram:
        assert len(ram)==17 and not held_ends
        width=128
    else:
        width=len(held_ends);assert width in (32,128),('unsupported held width',width)
    selector_bits=4 if width==32 else 2
    selectors={f'service_pixel_snapshot/part[{i}]~FF' for i in range(selector_bits)}
    seen_selectors=set()
    for row in cross+structure.get('ram_input_cross_domain_cones',[]):
        start,end=row['start'],row['end']
        kind=None
        if start.startswith('service_pixel_snapshot/held[') and end.startswith('service_pixel_cache['):
            kind='held_pixel_data';limit=2*6.666
        elif start in selectors and end.startswith('service_pixel_snapshot/held['):
            kind='held_part_selector';limit=2*13.468
            seen_selectors.add(start)
        elif ram and start in selectors and re.fullmatch(r'service_pixel_snapshot/storage/banks\[[0-6]\]\.sample\|WDATA\[\d+\]',end):
            bank,bit=map(int,re.findall(r'\[(\d+)\]',end));assert bank*20+bit<128 and bit<20
            assert row['source_clocks']==[[['net','process_clk_150',None]]] and row['destination_clock']==[['net','hdmi_tx_slow_clk',None]]
            kind='held_part_selector_ram';limit=2*13.468;seen_selectors.add(start)
        elif end in {x['first']+'|D' for x in structure['two_stage_chains']}:
            kind='two_ff_first_stage';limit=None
        assert kind,(start,end)
        rec=expected.setdefault(end,dict(kind=kind,starts=set(),limit_ns=limit))
        assert rec['kind']==kind and rec['limit_ns']==limit
        rec['starts'].add(start+'|CLK')
    counts=collections.Counter(v['kind'] for v in expected.values())
    assert counts==(dict(held_part_selector_ram=115,two_ff_first_stage=5) if ram else dict(held_pixel_data=width,held_part_selector=width,two_ff_first_stage=5)),counts
    assert seen_selectors==selectors,('missing selector sources',selectors-seen_selectors)
    # Every direct external data input must be either the button first stage,
    # an existing reset/PLL qualifier, or one of the four lock first stages.
    reset='u_ddr3_top/controller/u_efx_ddr3_soft_controller/genblk1.inst_ddr3_axi/inst_efx_ddr3/top_mc/clk_rstn'
    allowed={reset,'DDR3_PLL_RSTN','i_pll_locked','pll_locked','process_pll_locked','user_pll_locked','service_key_n'}
    assert {r['source'] for r in structure['external_inputs']}==allowed
    for r in structure['external_inputs']:
        if r['source']=='service_key_n':assert r['end']=='service_button/key_meta~FF|D'
    for r in structure.get('ram_external_inputs',[]):
        assert r['source'] in allowed-{'service_key_n'}
        assert re.fullmatch(r'service_serializer/storage/banks\[\d+\]\.sample\|WDATA\[\d+\]',r['end'])
        assert r['clock']==[['net','process_clk_150',None]]
    measured={};reports={}
    for stem,launch,capture in [('service_pixel_to_system','hdmi_tx_slow_clk','process_clk_150'),
                               ('service_system_to_pixel','process_clk_150','hdmi_tx_slow_clk')]:
        path=root/'outflow'/('ti60f225_oob.'+stem+'.rpt');body=path.read_text()
        assert '-npaths 10000 -nworst 1' in body
        blocks=body.split('++++ Path ')[1:];assert 0<len(blocks)<10000
        reports[str(path)]=sha(path)
        for block in blocks:
            def field(label):
                m=re.search(r'^'+re.escape(label)+r'\s*:\s*(.+)$',block,re.M);assert m,label
                return m[1].strip()
            end=field('Path End')
            if end not in expected:continue  # retained endpoints have their existing audit
            start=field('Path Start');entry=expected[end]
            assert start in entry['starts'],(start,end,entry)
            assert field('Launch Clock').split()[0]==launch and field('Capture Clock').split()[0]==capture
            assert end not in measured,('duplicate endpoint',end)
            data=float(field('Data Path Delay').split()[0])
            charged=None
            if entry['limit_ns'] is not None:
                lc=float(field('Launch Clock Path Delay'));cc=float(field('+ Capture Clock Path Delay'))
                uncertainty=abs(float(field('- Clock Uncertainty')));setup=abs(float(field('- Setup')))
                charged=data+abs(lc-cc)+uncertainty+setup+.010
                assert charged<entry['limit_ns'],('held data settling',start,end,charged,entry['limit_ns'])
            measured[end]=dict(start=start,end=end,kind=entry['kind'],data_delay_ns=data,
                               charged_ns=charged,limit_ns=entry['limit_ns'])
    assert set(measured)==set(expected),('missing endpoints',sorted(set(expected)-set(measured)))
    return dict(status='VIDEO_SERVICE_ROUTED_CDC_SETTLING_PASS_NOT_RELEASE',map_sha256=structure['map_sha256'],
                structure_sha256=sha(structure_path),checker_sha256=sha(__file__),
                endpoint_counts=dict(counts),mailbox_bits=width,selector_bits=selector_bits,paths=list(measured.values()),reports=reports,
                analogue_metastability_proven=False,reset_release_separate=True,board_release=False)
