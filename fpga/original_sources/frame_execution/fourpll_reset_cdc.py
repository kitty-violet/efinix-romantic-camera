"""Inventory additional observation-only crossings without waiving data paths."""
import collections
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT, sha
from ddr_user_clock_profile import DDR


def inventory(board):
    board = board.resolve()
    assert board.is_relative_to(ROOT / 'tmp')
    peri = board / 'ti60f225_oob.peri.xml'
    plls = {e.attrib['name']: e for e in ET.parse(peri).iter() if e.tag.endswith('}pll')}
    assert plls['DDR3_PLL'].get('pll_def') == 'PLL_TR0'
    assert plls['pll_inst4'].get('pll_def') == 'PLL_BR0'
    physical = Path(json.loads((board / 'inputs.json').read_text())['base'])
    pm = json.loads((physical / 'physical_inputs.json').read_text())
    top = next(Path(r['snapshot']) for r in pm['inputs'] if Path(r['snapshot']).name == 'ti60f225_oob_top.v')
    assert sha(top) == next(r['sha256'] for r in pm['inputs'] if Path(r['snapshot']) == top)
    body = top.read_text()
    assert 'bringup_ddr_meta<=bringup_ddr_raw;' in body
    assert 'bringup_ddr_sync<=bringup_ddr_meta;' in body
    assert body.count('bringup_ddr_sync') == 3  # declaration, sequential D/Q, APB input
    assert '.bringup_ddr_status(bringup_ddr_sync)' in body
    groups = collections.defaultdict(list)
    rows, reports = [], []
    for stem in ('user_to_ddr', 'ddr_to_user'):
        report = board / f'outflow/ti60f225_oob.{stem}_crossings.rpt'
        text = report.read_text()
        assert '-npaths 10000 -nworst 1' in text
        paths = text.split('++++ Path ')[1:]
        assert 0 < len(paths) < 10000
        for path in paths:
            def get(field):
                return re.search(r'^' + field + r'\s*:\s*(.+)$', path, re.M).group(1).strip()
            start, end = get('Path Start'), get('Path End')
            launch, capture = get('Launch Clock').split()[0], get('Capture Clock').split()[0]
            assert (launch == 'process_clk_150' and capture in DDR) or (capture == 'process_clk_150' and launch in DDR)
            if re.fullmatch(r'bringup_ddr_meta\[\d+\]~FF\|D', end):
                assert capture == 'process_clk_150'
                kind = 'read_only_diagnostic_first_stage'
            else:
                assert end.startswith('u_ddr3_top/controller/')
                if start.startswith('startup_reset/released~FF|') and end.endswith('~FF|SR'):
                    rows.append(dict(start=start,end=end,launch=launch,capture=capture,
                                     kind='new_global_async_reset_pin_review_required'))
                    continue
                assert start.startswith('u_ddr3_top/controller/'), ('Unknown CDC source', start, end)
                if end.endswith('~FF|SR'):
                    kind = 'vendor_async_reset_pin_review_required'
                elif re.search(r'(Gray_buffercc/buffers_0\[\d+\]|addr_sync/async_reg\[0\]\[\d+\])~FF\|D$', end):
                    kind = 'vendor_fifo_gray_first_stage'
                    groups[re.sub(r'\[\d+\](?=~FF)', '[*]', end)].append((start, end))
                elif re.search(r'/r_read_back_en_pipe_sync_P1\[\d+\]~FF\|D$', end):
                    kind = 'vendor_read_control_first_stage'
                else:
                    raise AssertionError(('Unclassified cross-domain endpoint', end))
            rows.append(dict(start=start, end=end, launch=launch, capture=capture, kind=kind))
        reports.append(dict(path=str(report), sha256=sha(report), endpoints=len(paths)))
    counts = dict(collections.Counter(r['kind'] for r in rows))
    vendor = {k: v for k, v in counts.items() if k not in ('read_only_diagnostic_first_stage','new_global_async_reset_pin_review_required')}
    assert vendor == dict(vendor_async_reset_pin_review_required=11, vendor_fifo_gray_first_stage=70, vendor_read_control_first_stage=3)
    assert 1 <= counts.get('read_only_diagnostic_first_stage', 0) <= 32
    assert len(groups) == 10 and all(len(v) == 7 for v in groups.values())
    assert len({r['end'] for r in rows}) == len(rows)
    return dict(status='CROSS_PLL_ENDPOINT_INVENTORY_WITH_READ_ONLY_DIAGNOSTICS', board=str(board),
                peri_sha256=sha(peri), sdc_sha256=sha(board/'ti60f225_oob.sdc'),
                ddr_source_sha256=sha(board/'ddr516_reference.sv'), wrapper_sha256=sha(board/'ddr516_v4_controller.sv'),
                timing_sha256=sha(board/'outflow/ti60f225_oob.timing.rpt'), reports=reports,
                endpoints=rows, classes=counts, gray_buses=[dict(name=k, pins=v) for k, v in sorted(groups.items())],
                diagnostic_top_sha256=sha(top),
                diagnostic_use='Non-atomic repeated read observations only; no control feedback; handshake samples are not counters',
                new_global_reset_paths=[r for r in rows if r['kind']=='new_global_async_reset_pin_review_required'],
                reset_safety_proven=False, cdc_release=False, board_release=False)
