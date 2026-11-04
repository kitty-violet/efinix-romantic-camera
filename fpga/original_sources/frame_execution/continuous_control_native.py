"""Replace only the measured boot-control and admission cones of current map.

All shared address/data decode stays; a native-boundary equivalence test is
required before official mapping. This script does not release or download.
"""
import collections,json,re
from pathlib import Path
from run_units import ROOT,HERE,sha
from native_local_drivers import CELL,graph,render_token,raw_cells
from camera_joint_native import OUT as OUTPUT
from probe_full_native_split_map import params
from complete_native_logic_split import protected_blocks
from control_command_repair import once

PARENT=ROOT/'tmp/frame-execution/connected-board-cnn-sparse-control1'
VARIANT=HERE/'variants/continuous_run'
DEST=ROOT/'tmp/frame-execution/continuous-control-native2'

def net(name,bit=None):return ('net',name,bit)

def connections(g):
    drivers={};loads=collections.defaultdict(set)
    for n,(k,p) in g.items():
        for port,ts in p.items():
            for t in ts:
                if t[0]!='net':continue
                if port in OUTPUT.get(k,[]):
                    assert t not in drivers,(t,n,drivers.get(t));drivers[t]=n
                else:loads[t].add((n,port))
    return drivers,loads

def source_expression(name):
    if name.startswith('cpu/sequencer/'):
        n=name.removeprefix('cpu/sequencer/').removesuffix('~FF')
        assert re.fullmatch(r'(low_word|high_word|sample|crc|base_count|index|bank|row|tag|frames|nibble|diag_polls|run_ticks|bus_ticks|waiting|operation_state|operation_valid)(\[\d+\])?',n),name
        return n
    n=name.removeprefix('cpu/').removesuffix('~FF')
    for a,b in [('boot_observer_state','state'),('boot_fault','fault'),('boot_failed','failed'),('boot_terminal','terminal'),('permit','run_allowed')]:
        if n==a or n.startswith(a+'['):return n.replace(a,b,1)
    raise AssertionError(name)

def export_rows(g,removed,loads,owned):
    rows=[]
    for n in sorted(owned):
        k,p=g[n]
        if k=='EFX_RAM10':
            assert n=='cpu/sequencer/operations' and len(p['RDATA'])==16
            pairs=[(t,f'operation[{15-i}]') for i,t in enumerate(p['RDATA']) if i>=2]
        else:pairs=[(p['Q'][0],source_expression(n))]
        for t,expr in pairs:
            if any(m not in removed for m,_ in loads[t]):rows.append(dict(cell=n,token=t,expression=expr))
    assert len({tuple(x['token']) for x in rows})==len(rows)
    return rows

def exposed_source(text,rows,name):
    text=once(text,'module cnn_boot_sequencer #(',f'module {name} #(')
    text=once(text,' output wire [5:0] observer_state\n',
         f' output wire [5:0] observer_state,\n output wire [{len(rows)-1}:0] native_state,\n output wire [9:0] frame_budget\n')
    expr=', '.join(x['expression'] for x in reversed(rows))
    text=once(text,'endmodule',
              ' assign native_state={'+expr+'};\n assign frame_budget=frames;\nendmodule')
    return text

def main():
    assert not DEST.exists()
    src=PARENT/'outflow/ti60f225_oob.map.v'
    assert sha(src)=='07808e28eab136793a81cb6c911958fbd6fd8a00b835fff53f4476b6371dd12a'
    s=src.read_text();g=graph(s);ps=params(s);drivers,loads=connections(g)
    survey=ROOT/'tmp/board-test/preview-control-boundary1.json'
    inv=json.loads(survey.read_text());assert inv['source_sha256']==sha(src)
    owned={n for n in inv['removed_cells'] if g[n][0] in ('EFX_FF','EFX_RAM10')}
    removed=set(inv['removed_cells'])
    # The existing mapped response predicate is a preserved controller input,
    # not part of the replacement's sequential logic.
    reply_cell='LUT__161020'
    assert ps[reply_cell]['LUTMASK']==(16,0x1000)
    assert g[reply_cell][1]=={'I0':[net('cpu/last_host')],'I1':[net('apb_write')],
        'I2':[net('ceg_net4871')],'I3':[net('cpu/ctrl_state_match',9)],'O':[net('n78042_2')]}
    removed.discard(reply_cell)
    # Propagate retention upstream, so no preserved cell loses its driver.
    while True:
        rejected={n for n in removed-owned for pin,ts in g[n][1].items() if pin in OUTPUT[g[n][0]]
            for t in ts if any(m not in removed for m,_ in loads[t])}
        if not rejected:break
        removed-=rejected
    valid=drivers[net('cpu/boot_valid')];assert g[valid][0]=='EFX_LUT4'
    removed.add(valid)
    rows=export_rows(g,removed,loads,owned)
    # Replace the old 5-bit admission including the combinational start gate.
    admission={n for n,(k,p) in g.items() if k=='EFX_FF' and n.startswith('cnn_admitted[')}
    assert len(admission)==5,sorted(admission)
    start=net('video_processing_pipeline/compute_start')
    ar=set(admission)|{drivers[start]};todo=list(ar)
    while todo:
        n=todo.pop();k,p=g[n]
        for pin,ts in p.items():
            if pin in OUTPUT[k] or pin in ('CLK','SR'):continue
            for t in ts:
                d=drivers.get(t)
                # Only the private admission gate is eligible. Existing
                # compute request, permit, reset and frame-budget stay cuts.
                if t in (net('cnn_start_request'),net('fixed_boot_run_allowed'),net('w_arstn')):continue
                if d and g[d][0] in ('EFX_LUT4','EFX_ADD') and d not in ar and not d in removed:
                    if t[1].startswith('cpu/') or d.startswith('cpu/'):continue
                    ar.add(d);todo.append(d)
    while True:
        reject={n for n in ar-admission-{drivers[start]} for pin,ts in g[n][1].items() if pin in OUTPUT[g[n][0]]
                for t in ts if any(m not in ar for m,_ in loads[t])}
        if not reject:break
        ar-=reject
    assert not ar & removed
    removed|=ar
    outputs={tuple(x['token']) for x in rows}|{net('cpu/boot_valid'),start}|{net('cnn_admitted',i) for i in range(5)}
    for n in removed:
        for pin,ts in g[n][1].items():
            if pin not in OUTPUT[g[n][0]]:continue
            for t in ts:
                assert t in outputs or all(m in removed for m,_ in loads[t]),(n,pin,t,loads[t])
    body=CELL.sub(lambda m:'' if m[2].lstrip('\\') in removed else m[0],s)
    body=re.sub(r'^\s*defparam\s+(\S+?)\s*\.\w+\s*=[^;]*;',
                lambda m:'' if m[1].lstrip('\\') in removed else m[0],body,flags=re.M)
    extra='''
    wire [9:0] preview_frame_budget,preview_admitted;
    cnn_preview_boot preview_boot(
      .clk(process_clk_150),.reset(!ceg_net4871),
      .cmd_valid(\\cpu/boot_valid ),.cmd_write(),.cmd_address(),.cmd_data(),
      .cmd_ready(\\cpu/boot_ready ),.rsp_valid(n78042_2),
      .rsp_error(\\cpu/reply_error | \\cpu/fatal_bus_error ),.rsp_data(\\cpu/reply_data ),
      .run_allowed(),.terminal(),.failed(),.fault(),.observed_crc(),.observer_state(),
      .native_state({EXPORTS}),.frame_budget(preview_frame_budget));
    cnn_frame_admission preview_admission(
      .clk(process_clk_150),.reset(!w_arstn),.permit(fixed_boot_run_allowed),
      .requested_frames(preview_frame_budget),.start_request(cnn_start_request),
      .start_accept(\\video_processing_pipeline/compute_start ),.admitted(preview_admitted));
    // The fixed-width diagnostic field saturates at 31. Full committed frame
    // count remains a 32-bit APB register; no modulo-32 count is presented.
    assign cnn_admitted=(|preview_admitted[9:5]) ? 5'd31:preview_admitted[4:0];
'''.replace('EXPORTS',', '.join(render_token(tuple(x['token'])) for x in reversed(rows)))
    anchor=body.index('    protected_cell protected_inst (');body=body[:anchor]+extra+body[anchor:]
    ng=graph(body);nps=params(body)
    assert set(ng)==set(g)-removed
    for n in ng:assert ng[n]==g[n] and nps.get(n)==ps.get(n),n
    assert protected_blocks(body)==protected_blocks(s)
    current=(PARENT/'assembly/EfxSapphireSoc.v').read_text()
    old=re.findall(r'^module cnn_boot_sequencer\b.*?^endmodule',current,re.M|re.S)[0]+'\n'
    new=(VARIANT/'cnn_boot_sequencer.sv').read_text()
    DEST.mkdir()
    (DEST/'native_source.v').write_text(body)
    (DEST/'cnn_preview_boot.sv').write_text(exposed_source(new,rows,'cnn_preview_boot'))
    (DEST/'original_exposed.sv').write_text(exposed_source(old,rows,'original_exposed'))
    (DEST/'boundary.json').write_text(json.dumps(dict(rows=rows,removed=sorted(removed),admission_removed=sorted(ar)),indent=2))
    result=dict(status='PREVIEW_NATIVE_CANDIDATE_BOUNDARY_EXPORTED_NOT_VERIFIED',
        parent_map_sha256=sha(src),removed_kinds=dict(collections.Counter(g[n][0] for n in removed)),
        unchanged_primitives=len(ng),exported_bits=len(rows),protected_byte_identical=True,
        changed_modules=['boot sequencer','frame admission'],diagnostic_admitted='saturates at 31; APB commits remain full32',
        source_sha256=sha(DEST/'native_source.v'),boundary_sha256=sha(DEST/'boundary.json'),
        inputs={str(p):sha(p) for p in [src,Path(__file__),survey,VARIANT/'cnn_boot_sequencer.sv',VARIANT/'cnn_frame_admission.sv']},
        hardware_access=False,functional_integration_pass=False,physical_pass=False,board_release=False)
    (DEST/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='inputs'}),flush=True)

if __name__=='__main__':main()
