"""Bind common-board resources, actual video timing and local gates to frozen RTL.

This audit never converts a map/behavioral-model result into physical approval.
Resource rows are nonoverlapping hierarchy partitions; inclusive parents and
their children must never be added together.
"""
import argparse, json, re, xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT, sha

KEYS = ('FF', 'SRL8', 'ADD', 'LUT4', 'COMB4', 'RAM', 'DSP_LOGICAL')

def manifest(root):
    data = json.loads((root/'inputs.json').read_text())
    result = {}
    for row in data['inputs']:
        p = Path(row['snapshot'])
        assert p.is_file() and sha(p) == row['sha256'], p
        assert p.name not in result or result[p.name] == row['sha256'], p.name
        result[p.name] = row['sha256']
    return data, result

def hierarchy(report):
    body = report.split('Module Resource Usage Distribution Estimates (begin)')[1]
    body = body.split('Module Resource Usage Distribution Estimates (end)')[0]
    rows, stack = [], []
    for line in body.splitlines():
        values = re.findall(r'(\d+)\((\d+)\)', line)
        if len(values) != 7:
            continue
        label = line[:re.search(r'\d+\(', line).start()].rstrip()
        depth = len(label)-len(label.lstrip())
        instance = label.strip().lstrip('+').split(':')[0]
        while stack and stack[-1][0] >= depth:
            stack.pop()
        path = '/'.join([x[1] for x in stack[1:]]+[instance])
        rows.append(dict(path=path, depth=depth,
            inclusive=dict(zip(KEYS, [int(x) for x,y in values])),
            exclusive=dict(zip(KEYS, [int(y) for x,y in values]))))
        stack.append((depth, instance))
    assert rows
    for key in KEYS:
        assert sum(row['exclusive'][key] for row in rows) == rows[0]['inclusive'][key], key
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--map',type=Path,required=True)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--assembly-only',action='store_true')
    a=p.parse_args()
    board, video, out = a.map.resolve(), a.video.resolve(), a.output.resolve()
    assert out.is_relative_to(ROOT) and not out.exists()
    br=json.loads((board/'result.json').read_text())
    bm, bs=manifest(board)
    assert sha(board/'inputs.json') == br['inputs_sha256']
    assert sha(board/'outflow/ti60f225_oob.map.v') == br['map_sha256']
    assert sha(board/'outflow/ti60f225_oob.map.rpt') == br['report_sha256']
    vm, vs=manifest(video)
    assert sha(vm['binary']) == vm['binary_sha256']
    bench_only={'efx_dsp24.v','efx_dsp48.v','efx_srl8.v','efx_ram10.v','efx_dpram10.v',
                'frame_execution_meter.sv','tb_fourway_full.sv'}
    shared=[]
    for name, digest in vs.items():
        if name in bench_only:
            continue
        assert name in bs, name
        assert bs[name] == digest, name
        shared.append(name)
    for forbidden in ('cnn_real_axi_pressure_harness.sv',
        'cnn_linebuffer_pressure.sv','operator_core_pnr_top.sv',
        'scaler_pnr_harness.sv','rgb_scaler_display_pipeline.sv'):
        assert forbidden not in bs, forbidden
    metrics=br['resources']
    report_text=(board/'outflow/ti60f225_oob.map.rpt').read_text()
    rows=hierarchy(report_text)
    synthesis_config={key:re.findall(r'^'+re.escape(key)+r'\s*:\s*([^\r\n]+)',report_text,re.M)
                      for key in ('mode','seq_opt','retiming','operator-sharing','pack-luts-to-comb4')}
    for key in ('mode','retiming','pack-luts-to-comb4'):
        assert len(synthesis_config[key])==1, (key,synthesis_config[key])
    xml=board/'ti60f225_oob.xml'
    tree=ET.parse(xml)
    requested={e.get('name'):e.get('value') for e in tree.findall(
        './/{http://www.efinixinc.com/enf_proj}synthesis/{http://www.efinixinc.com/enf_proj}param')}
    requested_seq='1' if br.get('sequential_opt',False) else '0'
    assert requested['seq_opt']==requested_seq
    # Efinity omits this documented default (1) from the map report header.
    # Accept omission only for an explicit XML + recorded-input request of 1.
    assert synthesis_config['seq_opt']==[requested_seq] or (
        requested_seq=='1' and synthesis_config['seq_opt']==[])
    assert bm.get('sequential_opt',False)==br.get('sequential_opt',False)
    assert synthesis_config['retiming']==['0']
    if br.get('operator_sharing',False):
        assert synthesis_config['operator-sharing']==['1']
    assert rows[0]['inclusive']['LUT4']==metrics['LUT4']
    assert rows[0]['inclusive']['RAM']==metrics['RAM10']+metrics['DPRAM10']
    assert metrics['DSP24']%2==0
    resource=dict(official_map=metrics, dumped_net_primitives=br['dumped_net_primitives'],
        physical_dsp=metrics['DSP24']//2+metrics['DSP48'],
        ram_blocks=metrics['RAM10']+metrics['DPRAM10'],
        device=dict(xlr=62016, ram_blocks=256, physical_dsp=160),
        placed_xlr=None,resource_fit_proven=False)
    # Prove the disabled legacy branch is no longer instantiated, while the
    # real fused residual and explicit illegal-mode rejection remain present.
    cluster=next(Path(r['snapshot']) for r in bm['inputs']
                 if Path(r['snapshot']).name=='compute_column_cluster.sv')
    ct=cluster.read_text()
    assert not re.search(r'\bresidual_column_fuse\s+\w+\s*\(',ct)
    assert 'pw_skip_fuse4 skip_fuse(' in ct
    assert 'if(cfg_residual || pc_residual || cfg_row_deltas!=0' in ct
    assert not any('/dut/residual' in row['path'] for row in rows)
    assert not br['physical_timing_pass'] and not br['board_download_allowed']
    gates=[]
    for name in ('pruned-guard1','pruned-guard-fourstate1','pruned-rejection1',
                 'rgb720-burst128-row1','video-output-ddr128-delay64',
                 'raw-capacity1','board-cpu-window2','apb-control2',
                 'program-fetch2','program-issue2','pair-owner1','rgb24-stream1'):
        root=ROOT/'tmp/frame-execution'/name
        rr=json.loads((root/'result.json').read_text())
        assert 'PASS' in rr['status'],name
        mm, sources=manifest(root)
        assert sha(root/'inputs.json')==rr['inputs_sha256'],name
        if name=='board-cpu-window2':
            for trial in rr['results']:
                log=root/f"seed{trial['seed']}-bad{trial['bad_mapping']}.log"
                assert sha(log)==trial['log_sha256'],(name,log)
            board_top=next(r['sha256'] for r in bm['inputs']
                           if Path(r['snapshot']).name=='ti60f225_oob_top.v')
            assert mm['board_sha256']==board_top,name
        else:
            log_key='simulation_sha256' if name in ('program-fetch2','program-issue2') else 'log_sha256'
            assert sha(root/'simulation.log')==rr[log_key],name
        checked=[]
        for n,h in sources.items():
            if n in bs:
                assert h==bs[n],(name,n)
                checked.append(n)
        assert checked, (name, 'gate has no shared implementation source')
        gates.append(dict(name=name,status=rr['status'],
            result_sha256=sha(root/'result.json'),matched_sources=checked))
    timing=None
    if not a.assembly_only:
        vr=json.loads((video/'result.json').read_text())
        assert sha(video/'inputs.json')==vr['inputs_sha256']
        assert sha(video/'simulation.log')==vr['log_sha256']
        body=(video/'simulation.log').read_text()
        assert 'PASS VIDEO_1080P frames=2' in body and 'PASS PAIRED_PIPELINE frames=2' in body
        commits=vr['paired_pipeline_commits']
        assert len(commits)==2
        interval=commits[1]['tick']-commits[0]['tick']
        assert interval==commits[1]['interval']
        events={label:[dict(frame=int(f),tick=int(t)) for f,t in re.findall(
            r'^'+label+r' frame=(\d+) tick=(\d+)(?: pixels=\d+)?$',body,re.M)]
            for label in ('CAMERA_START','CAPTURE_DONE','CNN_START','HDMI_START','HDMI_DONE')}
        assert all(len(v)==2 for v in events.values()), events
        per_frame=[]
        vector_manifest=json.loads((Path(vm['vectors'])/'frame_manifest.json').read_text())
        node_names=[n['id'] for n in vector_manifest['nodes']]
        assert len(node_names)==29
        node_rows=[tuple(map(int,row)) for row in re.findall(
            r'^FRAME_NODE frame=(\d+) node=(\d+) cumulative_cycles=(\d+) words=(\d+) input=(\d+) weight=(\d+) pool=(\d+) setup_drain=(\d+)$',body,re.M)]
        assert len(node_rows)==58
        previous={n:(0,)*6 for n in range(29)}
        node_frames=[]
        for i,commit in enumerate(commits):
            camera,capture,start,hdmi_start,hdmi_done=[events[k][i] for k in events]
            assert camera['frame']==capture['frame']==start['frame']==17+i
            assert hdmi_start['frame']==hdmi_done['frame']==commit['frame']==i+1
            assert camera['tick']<capture['tick']<=start['tick']<commit['tick']<hdmi_start['tick']<hdmi_done['tick']
            stage_cycles=dict(input_acquisition=capture['tick']-camera['tick'],
                capture_to_compute_start=start['tick']-capture['tick'],
                compute_start_to_commit=commit['tick']-start['tick'],
                result_to_hdmi_first_pixel=hdmi_start['tick']-commit['tick'],
                hdmi_first_to_done=hdmi_done['tick']-hdmi_start['tick'],
                input_first_to_result=commit['tick']-camera['tick'],
                input_first_to_hdmi_done=hdmi_done['tick']-camera['tick'])
            per_frame.append(dict(source_frame=camera['frame'],cycles=stage_cycles,
                ms_at_assumed_150mhz={k:v/150000 for k,v in stage_cycles.items()}))
            current=[r for r in node_rows if r[0]==i+1]
            assert {r[1] for r in current}==set(range(29))
            entries=[]
            for _,n,*totals in current:
                delta=[v-p for v,p in zip(totals,previous[n])]
                assert all(v>=0 for v in delta)
                previous[n]=tuple(totals)
                entries.append(dict(node=n,id=node_names[n],compute_command_cycles=delta[0],
                    compute_command_ms=delta[0]/150000,words=delta[1],input_wait=delta[2],
                    weight_wait=delta[3],pool_wait=delta[4],setup_drain=delta[5]))
            active=sum(r['compute_command_cycles'] for r in entries)
            assert active<=commit['cnn_cycles']
            node_frames.append(dict(frame=i+1,nodes=entries,
                command_cycles=active,outside_compute_command_cycles=commit['cnn_cycles']-active,
                note='Actual compute-command windows. DMA overlaps these windows; do not add DMA time again. NN/address fusion and fused residual may be charged to their consumer/producer node.'))
        timing=dict(cnn_frame_cycles=[r['cnn_cycles'] for r in commits],
            ddr_service_profile=vr['profile'],
            ddr_profile_description='ideal one-cycle response, no injected R/W stalls' if vr['profile']==0 else '12-cycle response delay and deterministic pseudo-random R/W stalls',
            cnn_frame_ms=[r['cnn_cycles']/150000 for r in commits],
            commit_ticks=[r['tick'] for r in commits],
            observed_commit_interval_cycles=interval,
            observed_commit_interval_ms=interval/150000,
            observed_commit_rate_fps=150000000/interval,
            hdmi_frames=vr['hdmi_completed'],events=events,per_frame=per_frame,node_frames=node_frames,
            assumed_process_clock_hz=150000000,
            assumption_sta_qualified=False,interval_sample_count=1,
            steady_state_proven=False,result_sha256=sha(video/'result.json'))
        assert int(vr['line'][0])==vr['profile']
        read_beats,write_beats=map(int,vr['line'][5:7])
        assert len(vr['hdmi_completed'])==2
        rgb_cnn_bytes=(read_beats+write_beats)*16//2
        top=next(Path(r['snapshot']) for r in bm['inputs']
                 if Path(r['snapshot']).name=='ti60f225_oob_top.v')
        assert '.WORDS_PER_FRAME(518400)' in top.read_text()
        raw_roundtrip_bytes=518400*4*2
        accounted_bytes=rgb_cnn_bytes+raw_roundtrip_bytes
        timing['ddr_payload_accounting']=dict(
            measured_two_frame_read_beats=read_beats,
            measured_two_frame_write_beats=write_beats,
            measured_rgb_cnn_bytes_per_frame=rgb_cnn_bytes,
            raw_roundtrip_bytes_per_frame_from_actual_top=raw_roundtrip_bytes,
            accounted_bytes_per_frame=accounted_bytes,
            payload_bytes_per_second_at_30fps=accounted_bytes*30,
            required_sustained_bytes_per_second_with_15pct_headroom=accounted_bytes*30/0.85,
            not_included='CPU execution/program upload, future gesture traffic, refresh/turnaround and real PHY service gaps',
            raw_concurrent_replay=False,sustained_bandwidth_proven=False)
    result=dict(status='ASSEMBLY_SOURCE_AUDIT_ONLY' if a.assembly_only else 'CONNECTED_DIGITAL_EVIDENCE_AUDIT_PASS_NOT_RELEASE',
        common_map=str(board),video_replay=str(video),resources=resource,
        synthesis_config_from_report=synthesis_config,
        requested_synthesis_config={k:requested.get(k) for k in synthesis_config},
        project_xml_sha256=sha(xml),
        omitted_seq_opt_default='1; local official doc/topics/syn-options.html',
        gate_level_simulation=False,
        shared_video_synthesis_sources=sorted(shared),local_gates=gates,
        hierarchy=rows,timing=timing,
        frozen_inputs_sha256=sha(board/'inputs.json'),
        video_inputs_sha256=sha(video/'inputs.json'),audit_sha256=sha(__file__),
        full_baseline=False,physical_timing_pass=False,board_download_allowed=False,
        missing=['CSI/RAW/debayer traffic concurrently with the complete CNN and display',
                 'CPU firmware boot and real program loading',
                 'gesture entry/exit, demon effect and real AI FPS overlay',
                 'complete physical fit and multi-seed STA',
                 'trained weights and board DDR/PHY calibration and sustained efficiency'],
        dynamic_boundary='Actual 1080 RGB input through downscale, DDR capture, production CNN, paired compare, upscale, async FIFO and HDMI scheduler. RAW/CPU AXI idle. AXI behavioral memory; no PHY simulation.')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],resources=resource,
        shared_sources=len(shared),gates=len(gates),timing=timing)))

if __name__=='__main__':
    main()
