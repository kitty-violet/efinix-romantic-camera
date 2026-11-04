"""Publish one fully reconciled CSI-to-TMDS run and its same-source whole-board map.

Refuses incomplete or failed runs. It never substitutes a local-core timing or
another candidate's resources. Numerical stage windows overlap and are kept
separate from the exhaustive, exclusive input-to-last-output latency ledger.
"""
import argparse
import json
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from run_units import ROOT, sha


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def build(board, mapped):
    run = read(board / 'run_result.json')
    timing = read(board / 'frame_measurement.json')
    traffic = read(board / 'traffic_profile.json')
    commands = read(board / 'command_profile.json')
    binding = read(board / 'board_source_audit.json')
    resources = read(mapped / 'resource_audit3.json')
    metadata = read(board / 'inputs.json')
    assert metadata['ddr516']
    partition=None
    if metadata.get('native'):
        from production_board_binding import validate
        partition=validate(board,metadata)
        assert binding['production_partition']==partition
    bench = (board / 'tb_board_startup.sv').read_text()
    assert sha(board / 'tb_board_startup.sv') == metadata['tb_sha256']
    assert '`timescale 1ns/1ps' in bench
    half_period = Decimal(re.search(r'always #([\d.]+) process_clk_150=', bench)[1])
    # Actual simulator time resolution, not the nominal 150 MHz label.
    cycle_ns = float(half_period.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP) * 2)
    contract = ROOT / 'network_ab/model_b_contract.json'
    assert sha(contract) == '5750c4668d75077746ec5d4accf8773ab437540cf933c202f153a8797c21dd01'
    sys.path.insert(0, str(contract.parent))
    from model_contract import compile_contract, load_contract
    nodes = compile_contract(load_contract())['nodes']
    assert len(nodes) == 29
    sim_inputs = sha(board / 'inputs.json')
    map_inputs = sha(mapped / 'inputs.json')
    assert run['status'] == 'BOARD_DIGITAL_MEASUREMENT_COMPLETE'
    assert timing['measurement_valid'] and len(timing['complete_frames']) >= 3
    assert run['returncode'] == 0 and not run['failures'] and run['ddr_model_error_total'] == 0
    assert sha(board / 'simulation.log') == run['log_sha256']
    assert binding['status'] == 'DDR516_BOARD_SOURCE_BINDING_PASS'
    assert binding['map_inputs_sha256'] == resources['inputs_sha256'] == map_inputs
    assert binding['simulation_inputs_sha256'] == sim_inputs
    for data in (run, timing, traffic, commands):
        assert data['inputs_sha256'] == sim_inputs
    for data in (timing, traffic, commands):
        assert data['run_sha256'] == sha(board / 'run_result.json')
    assert traffic['log_sha256'] == run['log_sha256'] and traffic['complete_run']
    assert commands['csv_sha256'] == sha(board / 'cnn_command_events.csv')
    assert commands['descriptor_hex_sha256'] == sha(board / 'compute.hex')
    assert resources['map_sha256'] == sha(mapped / 'outflow/ti60f225_oob.map.v')
    assert resources['report_sha256'] == sha(mapped / 'outflow/ti60f225_oob.map.rpt')
    for root in (board, mapped):
        for row in read(root / 'inputs.json')['inputs']:
            assert sha(row['snapshot']) == row['sha256']
    for frame in timing['complete_frames']:
        assert abs(frame['unaccounted_latency_ms']) < 1e-9
    assert all(r['completed_command_count'] == 8789 for r in commands['frames'] if r['committed'])
    counts = resources['official_resources']
    assert sum(v['LUT4'] for v in resources['partitions'].values()) == counts['LUT4']
    rate = timing['observed_rate']
    lines = [
        '# CSI至TMDS整板统一测量与资源分析', '',
        f"本报告只使用 `{board.name}` 的同次连续运行和 `{mapped.name}` 的共同综合。"
        f"共完成 {len(timing['complete_frames'])} 个不同 frame_id 的完整屏幕帧，"
        'Micron器件时序错误、板内协议/像素检查错误均为0。', '',
        f"全部CNN提交的平均间隔为 **{rate['ai_mean_interval_ms']:.6f} ms**，"
        f"对应 **{rate['ai_fps']:.6f} AI fps**；"
        f"不同结果首次显示的平均间隔为 **{rate['new_screen_mean_interval_ms']:.6f} ms**。"
        '重复扫描旧图像不计作新AI结果。', '',
        '| frame_id | 输入至首个显示像素/ms | 输入至最后显示像素/ms | 未归属耗时/ms |',
        '|---|---:|---:|---:|'
    ]
    for frame in timing['complete_frames']:
        lines.append(f"| {frame['frame']} | {frame['input_to_first_pixel_ms']:.6f} | "
                     f"{frame['input_to_last_pixel_ms']:.6f} | {frame['unaccounted_latency_ms']:.9f} |")
    lines += ['', '下表每列构成该帧完整、互不重叠的时延账；RAW读取、RGB缩放、DDR和CNN之间的并行不能重复相加。', '',
              '| 阶段 | ' + ' | '.join(f"帧{f['frame']}/ms" for f in timing['complete_frames']) + ' |',
              '|---|' + '---:|' * len(timing['complete_frames'])]
    names = ['相机端口开始至RAW完整写入', 'RAW读回、预处理、RGB写入及CNN准入',
             'CNN开始至结果提交', '提交至首次显示像素', '首个至最后显示像素']
    for index, label in enumerate(names):
        lines.append('| ' + label + ' | ' + ' | '.join(
            f"{f['exclusive_latency_ledger'][index]['duration_ms']:.6f}" for f in timing['complete_frames']) + ' |')
    lines += ['', '端到端时延和相邻新结果间隔是不同指标：采集、CNN与显示可处理不同帧，'
              '不能用端到端时延的倒数代替AI FPS。上表输入扫描、整帧准入、计算、显示等待和有效扫描已全部归账。', '',
              '| 前一frame_id → 后一frame_id | CNN新结果间隔/ms |', '|---|---:|']
    for interval in timing['all_ai_commit_intervals']:
        lines.append(f"| {interval['previous']} → {interval['frame']} | {interval['interval_ms']:.6f} |")
    command_frames = {r['frame']: r for r in commands['frames'] if r['committed']}
    measured_frames = [r['frame'] for r in timing['complete_frames']]
    lines += ['', '下面逐层列出实际计算指令占用窗口，包含窗口内供数、权重和结果等待。'
              'NN及残差融合没有独立指令窗口时，其代价归在使用它的指令内，不能将“无独立窗口”理解成没有硬件代价。', '',
              '| 层/节点 | ' + ' | '.join(f'帧{f}/ms' for f in measured_frames) + ' |',
              '|---|' + '---:|' * len(measured_frames)]
    by_node = {f: {r['node']: r['window_cycles'] for r in command_frames[f]['nodes']} for f in measured_frames}
    for index, node in enumerate(nodes):
        cells = [f"{by_node[f][index]*cycle_ns/1e6:.6f}" if index in by_node[f] else '无独立窗口' for f in measured_frames]
        lines.append('| ' + node['id'] + ' | ' + ' | '.join(cells) + ' |')
    lines.append('| 计算指令窗口之外的生产运行周期 | ' + ' | '.join(
        f"{command_frames[f]['outside_command_window_cycles']*cycle_ns/1e6:.6f}" for f in measured_frames) + ' |')
    lines += ['', f'逐指令周期按本次仿真实际时钟周期{cycle_ns:.6f}ns换算；'
              '完整端到端表直接使用事件时间戳。窗口外的生产控制、DMA等待等没有被丢弃，'
              '它们已包含在CNN开始至提交总时间中。', '',
              '| 相邻CNN提交间隔 | DDR读事务 | DDR写事务 | 读拍 | 写拍 | 合计总线流量/MB每秒 |',
              '|---|---:|---:|---:|---:|---:|']
    for interval in traffic['compute_commit_intervals']:
        count = interval['counters']
        lines.append(f"| {interval['previous_frame']} → {interval['frame']} | {count['AR']} | {count['AW']} | "
                     f"{count['R']} | {count['W']} | {interval['bus_payload_MB_per_s']:.3f} |")
    lines += ['', '这些流量计数包括同一时间段内相机、CNN、显示、取指和CPU的全部原生DDR事务。'
              '时间段中可能存在重复显示或被丢弃的采集帧，因此不将其冒充单个图像的独占流量。', '',
              '| 一次性启动事件 | 仿真时间戳/ms |', '|---|---:|']
    startup_names = dict(BOARD_CAL_DONE='真实DDR校准完成', PAYLOAD_PRELOADED='诊断包预置完成', BOARD_CPU_RUN='CPU装载/CRC/回读通过并启动视频')
    for name, value in timing['startup_ns'].items():
        lines.append(f'| {startup_names.get(name,name)} | {value/1e6:.6f} |')
    lines += ['', '启动与相机采集部分重叠；以上是时间戳，不能再相加到每帧稳态间隔。']
    lines += ['', f"共同综合为 **{counts['LUT4']} LUT4、{counts['FF']} FF、"
              f"{resources['ram_blocks']}/256 RAM、{resources['physical_dsp']}/160物理DSP**。"
              f"另有{counts['ADD']} ADD、{counts['COMB4']} COMB4、{counts['SRL8']} SRL8。"
              '这些是map统计，不是完成布局布线的XLR；不能把FF与LUT直接相加。'
              'Ti60为62,016逻辑单元（LE）、60,800 XLR。当前逻辑占用超容量，'
              'RAM仅剩3块且DSP已满，容量门禁未通过。', '',
              '| 排他资源分区 | LUT4 | 占整板比例 | RAM |', '|---|---:|---:|---:|']
    labels = dict(board_periphery='板级外围及顶层', other_cnn='CNN其余控制/参数/存储',
                  routing='CNN数据路由', reader='CNN读取', dispatch='CNN列调度', pool='计算及量化池', dma='CNN行DMA')
    for name, values in sorted(resources['partitions'].items(), key=lambda x: -x[1]['LUT4']):
        lines.append(f"| {labels[name]} | {values['LUT4']} | {values['LUT4']/counts['LUT4']:.2%} | {values['RAM']} |")
    lines += ['', '原始共同资源、逐指令窗口和DDR计数分别见 `resource_audit3.json`、'
              '`command_profile.json`、`traffic_profile.json`。逐指令窗口含真实等待；'
              'DDR读写等待会重叠，不能把这些计数再加到上面的时延账。', '',
              '测量从CSI输入端口开始，到TMDS编码输出端口结束，板内现有相机接收、RAW往返DDR、'
              '去马赛克、缩放、RGB帧缓存、真实CPU装载、生产CNN、显示读取及编码全部参与。'
              '本次固件选择同frame_id左右对比模式（APB控制值3）；'
              '运行时全屏选择逻辑也计入共同综合，但不冒充已经完成另一模式的动态测量。'
              '沿用已声明的热启动条件：跳过视频上电等待、通过Micron公开接口预置诊断包，'
              '实际DDR校准及CPU权重CRC/回读/RUN保留。权重是诊断权重；手势等交互叶项按用户要求本轮暂缓。'
              '结果不能作为训练画质、最终P&R/STA或实板校准/持续带宽验收。', '',
              f"同源绑定：仿真输入SHA `{sim_inputs}`；map输入SHA `{map_inputs}`；"
              f"运行日志SHA `{run['log_sha256']}`；网表SHA `{resources['map_sha256']}`。", '']
    if partition:
        lines += ['本次采用Questa与Verilator联合仿真：生产CNN按同源RTL在Verilator二态内核运行，'
                  '相机、CPU、视频、仲裁及厂商DDR/CSI在Questa运行。边界保留逐拍交互，'
                  '已通过同源四态接口对照、活跃计算对照、有效输入未知值拒绝及真实DDR四主并发短回归。'
                  '这不是整个CNN内部四态语义的全状态证明，亦不改变物理验收边界。','']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board', type=Path, required=True)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert output.is_relative_to(ROOT) and not output.exists()
    report = build(args.board.resolve(), args.map.resolve())
    output.write_text(report, encoding='utf8')
    print(json.dumps(dict(status='UNIFIED_BOARD_REPORT_PUBLISHED', output=str(output), sha256=sha(output))))


if __name__ == '__main__':
    main()
