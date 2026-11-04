from pathlib import Path
import hashlib,json,re
from analyze_trace import analyze
R=Path(__file__).resolve().parent

def measurement(folder,depfile):
 d=R/folder;p=json.loads((d/'result.json').read_text('utf8'))
 assert p['status']=='FULL_COMPUTE_FRAME_PASS_NOT_BOARD_RELEASE'
 for name,h in p['inputs'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
 body=(d/'test.log').read_text('utf8')
 m=re.search(r'PASS S4R1_FULL_COMPUTE_FRAME cycles=(\d+) outputs=(\d+) checked_bytes=(\d+)',body);assert m
 cycles,pixels,checked=map(int,m.groups());assert(pixels,checked)==(921600,6451200)
 overlap={int(k):int(v)for k,v in re.findall(r'AUDIT_OVERLAP (\d+) (\d+)',body)}
 analysis=analyze(d,depfile)
 return dict(cycles=cycles,ms_at_150mhz=cycles/150000,overlap=overlap,
  simultaneous_percent=overlap[3]/sum(overlap.values())*100,
  dependency_wait_cycles=analysis['compute_dependency_wait_cycles'],
  group_span_cycles=analysis['group_span_cycles'],checked_bytes=checked,outputs=pixels)

def main():
 for name,h in json.loads((R/'snapshot.json').read_text('utf8'))['files'].items():
  assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
 for name,v in json.loads((R/'candidate-vectors/result.json').read_text('utf8'))['files'].items():
  assert hashlib.sha256((R/'candidate-vectors'/name).read_bytes()).hexdigest()==v['sha256'],name
 before=measurement('trace1','dependency-audit.json');after=measurement('candidate-trace1','candidate-dependency-audit.json')
 saved=before['cycles']-after['cycles'];pct=saved/before['cycles']*100
 report=dict(status='ISOLATED_PROGRAM_ONLY_FULL_FRAME_NUMERIC_AND_SPEEDUP_PASS'if saved>0 else'ISOLATED_NUMERIC_PASS_NO_SPEEDUP',
  before=before,after=after,saved_cycles=saved,saved_ms_at_150mhz=saved/150000,latency_reduction_percent=pct,
  throughput_ratio=before['cycles']/after['cycles'],hardware_access=False,board_fps_measured=False,
  weights_changed=False,rtl_changed=False,physical_ram_added=False,
  limitation='One full PIXEL integer frame with transactional DDR backpressure. Five-slot program layout checked; no full five-style RTL matrix, no physical PHY/video-master contention, no board FPS or switch latency release.')
 (R/'performance-comparison.json').write_text(json.dumps(report,indent=2),'utf8')
 text=f'''
## 完整帧对照结果（本轮完成）

| 指标 | 参照 reorder4 | 提前预留候选 |
|---|---:|---:|
| 整帧生产周期 | {before['cycles']:,} | {after['cycles']:,} |
| 150MHz 换算时间 | {before['ms_at_150mhz']:.4f} ms | {after['ms_at_150mhz']:.4f} ms |
| 计算与 DMA 同时忙占比 | {before['simultaneous_percent']:.2f}% | {after['simultaneous_percent']:.2f}% |
| 计算等待 DMA 依赖周期 | {before['dependency_wait_cycles']:,} | {after['dependency_wait_cycles']:,} |
| 逐字节结果比对 | 6,451,200 字节 PASS | 6,451,200 字节 PASS |

本次整帧减少 **{saved:,} 拍（{saved/150000:.4f} ms，{pct:.2f}%）**。这是实际 RTL 仿真周期结果，不再只是调度模型估算。两次使用同一份输入、像素风权重和整数参考输出，完整 3×3 DW 保留。

编码、残差、解码三段的实际跨度（周期）分别由 {before['group_span_cycles']['encoder']:,} / {before['group_span_cycles']['residual']:,} / {before['group_span_cycles']['decoder']:,} 变为 {after['group_span_cycles']['encoder']:,} / {after['group_span_cycles']['residual']:,} / {after['group_span_cycles']['decoder']:,}。

候选增加的是提前预留的**已有 SRAM 地址范围**，最大占到每银行 1024 字上限；物理 RAM 数量、RTL 和 DDR 搬运总字节数不变。五个槽位原程序逐字相同，新程序仍保留每个风格独立系数；本轮完整帧数值试验实际只跑了 PIXEL。

结论：存在已验证可减的等待。有效改进是缓存生命周期与搬运顺序联动，而非直接删掉握手/完成条件。仍需主工程组合验证及监督板测，不能把 {after['ms_at_150mhz']:.4f} ms 的倒数当作真实摄像头到屏幕 FPS。

完整交接见 `接入说明.md`；机器可读比较见 `performance-comparison.json`。当前主工程、板上位流和 Flash 未改。
'''
 doc=R/'DMA与计算等待审查.md';old=doc.read_text('utf8').split('\n## 完整帧对照结果（本轮完成）')[0]
 doc.write_text(old+text,'utf8')
 print('FINAL_COMPARISON',json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
