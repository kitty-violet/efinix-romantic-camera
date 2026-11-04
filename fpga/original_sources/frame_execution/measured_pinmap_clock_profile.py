"""Constrain the two real PLL domains; keep all related DDR clocks together.

This does not certify CDC or reset circuitry and does not permit programming.
The evidence inventory is produced without exceptions, from routed STA paths.
"""
import collections,json,re,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,sha

DDR={'tx_cal_clk_90edge','sdram_clk','rx_cal_clk','ddr_core_clk200'}
GROUP='set_clock_groups -asynchronous -group {process_clk_150} -group {tx_cal_clk_90edge sdram_clk rx_cal_clk ddr_core_clk200}'

def inventory(board):
    board=board.resolve();assert board.is_relative_to(ROOT/'tmp')
    xml=board/'ti60f225_oob.peri.xml';tree=ET.parse(xml)
    plls={e.attrib['name']:e for e in tree.iter() if e.tag.endswith('}pll')}
    assert plls['DDR3_PLL'].get('pll_def')=='PLL_TR0'
    assert plls['pll_inst4'].get('pll_def')=='PLL_BR0'
    outputs=lambda p:{e.get('name') for e in p if e.tag.endswith('}comp_output_clock')}
    assert DDR<=outputs(plls['DDR3_PLL']) and 'process_clk_150' in outputs(plls['pll_inst4'])
    groups=collections.defaultdict(list);reports=[];rows=[]
    for stem,expected in [('user_to_ddr',44),('ddr_to_user',40)]:
        report=board/f'outflow/ti60f225_oob.{stem}_crossings.rpt'
        body=report.read_text();assert '-npaths 10000 -nworst 1' in body
        paths=body.split('++++ Path ')[1:];assert len(paths)==expected<10000
        for path in paths:
            get=lambda field:re.search(r'^'+field+r'\s*:\s*(.+)$',path,re.M).group(1).strip()
            start=get('Path Start');end=get('Path End');launch=get('Launch Clock').split()[0];capture=get('Capture Clock').split()[0]
            assert (launch=='process_clk_150' and capture in DDR) or (capture=='process_clk_150' and launch in DDR)
            assert start.startswith('u_ddr3_top/controller/') and end.startswith('u_ddr3_top/controller/')
            if end.endswith('~FF|SR'):
                kind='vendor_async_reset_pin_review_required'
            elif re.search(r'(Gray_buffercc/buffers_0\[\d+\]|addr_sync/async_reg\[0\]\[\d+\])~FF\|D$',end):
                kind='vendor_fifo_gray_first_stage'
                key=re.sub(r'\[\d+\](?=~FF)', '[*]',end)
                groups[key].append((start,end))
            elif re.search(r'/r_read_back_en_pipe_sync_P1\[\d+\]~FF\|D$',end):
                kind='vendor_read_control_first_stage'
            else:raise AssertionError(('Unclassified cross-domain endpoint',end))
            rows.append(dict(start=start,end=end,launch=launch,capture=capture,kind=kind))
        reports.append(dict(path=str(report),sha256=sha(report),endpoints=len(paths)))
    counts=dict(collections.Counter(r['kind'] for r in rows))
    assert counts=={'vendor_async_reset_pin_review_required':11,'vendor_fifo_gray_first_stage':70,'vendor_read_control_first_stage':3}
    assert len(groups)==10 and all(len(v)==7 for v in groups.values())
    assert len({r['end'] for r in rows})==84
    return dict(status='CROSS_PLL_ENDPOINT_INVENTORY_NOT_CDC_RELEASE',board=str(board),
        peri_sha256=sha(xml),sdc_sha256=sha(board/'ti60f225_oob.sdc'),
        ddr_source_sha256=sha(board/'ddr516_reference.sv'),wrapper_sha256=sha(board/'ddr516_v4_controller.sv'),
        timing_sha256=sha(board/'outflow/ti60f225_oob.timing.rpt'),reports=reports,
        endpoints=rows,classes=counts,gray_buses=[dict(name=k,pins=v) for k,v in sorted(groups.items())],
        clock_source={'DDR3_PLL':'PLL_TR0','process_clk_150':'pll_inst4/PLL_BR0'},
        reset_safety_proven=False,cdc_release=False,board_release=False)

def apply_profile(root,proof):
    data=json.loads(proof.read_text());assert data['status']=='CROSS_PLL_ENDPOINT_INVENTORY_NOT_CDC_RELEASE'
    skew_file=proof.parent/'gray_skew_audit.json'
    skew=json.loads(skew_file.read_text())
    assert skew['status']=='ROUTED_GRAY_SKEW_PASS_NOT_RESET_OR_CDC_RELEASE'
    assert skew['inventory_sha256']==sha(proof) and skew['buses']==10 and skew['endpoints']==70
    assert skew['max_skew_ns']<=skew['limit_ns']==2.5
    for r in data['reports']:assert sha(r['path'])==r['sha256']
    for name,key in [('ti60f225_oob.peri.xml','peri_sha256'),('ti60f225_oob.sdc','sdc_sha256'),
                     ('ddr516_reference.sv','ddr_source_sha256'),('ddr516_v4_controller.sv','wrapper_sha256')]:
        if name=='ti60f225_oob.peri.xml':
            from board_measured_pinmap import pin_audit, BASE as parent_board
            assert sha(parent_board/name)==data[key],('Parent clock profile differs',name)
            pin_audit(root/name)  # Exact five GPIO-only differences; full PLL trees unchanged.
        else:
            assert sha(root/name)==data[key],('Clock profile input differs',name)
    sdc=root/'ti60f225_oob.sdc';before=sdc.read_text();assert GROUP not in before
    extra='\n# DDR PLL_TR0 and user PLL_BR0 have no guaranteed relative phase.\n'
    extra+='# Retain related DDR200/DDR400 phase requirements and every same-domain check.\n'+GROUP+'\n'
    extra+='# Explicit Gray first-stage bus skew: stricter than the slow 150MHz period.\n'
    for bus in data['gray_buses']:
        starts=' '.join(sorted({a for a,b in bus['pins']}));ends=' '.join(sorted({b for a,b in bus['pins']}))
        extra+=f'set_bus_skew -from [get_pins {{{starts}}}] -to [get_pins {{{ends}}}] 2.500\n'
    sdc.write_text(before+extra,encoding='utf-8')
    assert sdc.read_text()==before+extra
    return dict(profile='independent_user_ddr_plls',proof=str(proof),proof_sha256=sha(proof),
        before_sha256=data['sdc_sha256'],after_sha256=sha(sdc),related_ddr_clocks_preserved=sorted(DDR),
        added_clock_groups=1,gray_bus_constraints=10,gray_evidence=str(skew_file),gray_evidence_sha256=sha(skew_file),
        note='Rerun explicit Gray analysis on each new routed candidate in an uncut diagnostic view; grouped report_bus_skew gives zero endpoints in this tool.',
        gpio_delta_audit='board_measured_pinmap.pin_audit',
        current_periphery_sha256=sha(root/'ti60f225_oob.peri.xml'),
        cdc_release=False,reset_safety_proven=False)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);a=p.parse_args()
    output=a.board/'clock_crossing_inventory.json';assert not output.exists()
    data=inventory(a.board);output.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(status=data['status'],classes=data['classes'],gray_buses=len(data['gray_buses']))))
