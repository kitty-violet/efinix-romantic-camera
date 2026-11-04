"""Partition the frozen board at vendor IP ports, not inside the video pipeline.

Simulation-only transformation. Protected source is neither decoded nor changed.
The first candidate retains the complete Sapphire model as a conservative island;
its two protected arithmetic modules are recorded, not mistaken for public RTL.
No speculative time advance, ideal memory, or scoreboard removal is permitted.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
from probe_board_dpi_cost import BIN, VR, SIM, run
from run_units import ROOT, sha


def clean(text):
    return re.sub(r'/\*.*?\*/|//[^\n]*', lambda m: ' ' * len(m[0]), text, flags=re.S)


def closing(text, pos):
    assert text[pos] == '('
    depth = 0
    for i in range(pos, len(text)):
        depth += (text[i] == '(') - (text[i] == ')')
        if depth == 0:
            return i
    raise AssertionError('Unclosed port/parameter list')


def module_header(text, name):
    stripped = clean(text)
    m = re.search(r'\bmodule\s+' + re.escape(name) + r'\b', stripped)
    assert m, name
    opening = stripped.index('(', m.end())
    if '#' in stripped[m.end():opening]:
        opening = stripped.index('(', closing(stripped, opening) + 1)
    end = closing(stripped, opening)
    assert stripped[end + 1:].lstrip().startswith(';')
    return opening, end, stripped[opening + 1:end]


def constant(expression, params):
    expression = re.sub(r'`?(\b[A-Za-z_]\w*\b)', lambda m: str(params[m[1]]), expression)
    assert re.fullmatch(r'[\d\s+*/()\-]+', expression), expression
    return int(eval(expression, {'__builtins__': {}}, {}))


def ports(text, name, params):
    opening, end, header = module_header(text, name)
    header = re.sub(r'\(\*.*?\*\)', '', header, flags=re.S)
    out = []
    direction = width = None
    for entry in header.split(','):
        entry = entry.strip()
        m = re.fullmatch(r'(input|output|inout)\s*(?:(?:wire|reg|logic)\s+)?(?:\[([^:]+):([^]]+)\])?\s*(\w+)', entry)
        if m:
            direction = m[1]
            width = abs(constant(m[2], params) - constant(m[3], params)) + 1 if m[2] else 1
            signal = m[4]
        else:
            assert direction and re.fullmatch(r'\w+', entry), (name, entry)
            signal = entry
        out.append((direction, width, signal))
    return opening, end, out


def instance(text, module, name):
    stripped = clean(text)
    m = re.search(r'\b' + module + r'\s+' + name + r'\s*\(', stripped)
    assert m, (module, name)
    opening = stripped.index('(', m.start())
    end = closing(stripped, opening)
    assert stripped[end + 1:].lstrip().startswith(';')
    last = stripped.index(';', end) + 1
    connections = []
    pos = opening + 1
    while pos < end:
        p = re.search(r'\.(\w+)\s*\(', stripped[pos:end])
        if not p:
            break
        begin = pos + p.end() - 1
        finish = closing(stripped, begin)
        connections.append((p[1], stripped[begin + 1:finish].strip()))
        pos = finish + 1
    assert len(connections) == len({n for n, _ in connections})
    return m.start(), last, connections


def identifiers(expression):
    expression = re.sub(r"\d+'[sS]?[bBoOdDhH][0-9a-fA-F_xXzZ?]+", '', expression)
    return set(re.findall(r'\b[A-Za-z_]\w*\b', expression))


def declaration(direction, width, name):
    return f'{direction} wire [{width - 1}:0] {name}'


def generate_partition(base, root, cpu_alu_only=False):
    bm = json.loads((base / 'inputs.json').read_text())
    for row in bm['inputs']:
        old = Path(row['snapshot'])
        assert old.is_relative_to(base) and sha(old) == row['sha256'], old
        target = root / old.relative_to(base)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old, target)
    for old in list(base.glob('*.hex')) + list(base.glob('*.bin')):
        if not (root / old.name).exists():
            shutil.copy2(old, root / old.name)
    source = (root / 'rtl/ti60f225_oob_top.v').read_text(encoding='utf8')
    macros = (root / 'rtl/ddr3_controller/ddr3_parameter.vh').read_text()
    params = {m[1]: int(m[2]) for m in re.finditer(r'`define\s+(\w+)\s+(\d+)\b', macros)}
    params['RANK_RATIO'] = 1
    opening, end, top_ports = ports(source, 'ti60f225_oob_top', params)
    top_inputs = {n for d, w, n in top_ports if d == 'input'}
    top_outputs = {n for d, w, n in top_ports if d == 'output'}
    assert not any(d == 'inout' for d, w, n in top_ports)
    islands = [
        ('EfxSapphireSoc', 'cpu', 'EfxSapphireSoc.v'),
        ('csi_rx_controller', 'inst_efx_csi2_rx', 'csi_rx_controller_sim.sv'),
        ('ddr516_v4_controller', 'u_ddr3_top', 'ddr516_v4_controller.sv'),
        ('dvi_encoder', 'dvi_encoder_m0', 'rtl/dvi_tx/dvi_encoder.v_encrypted.v'),
    ]
    if cpu_alu_only:
        islands = [x for x in islands if x[1] != 'cpu']
    cuts = []
    body = source[end + 2:]
    for module, name, file in islands:
        first, last, connections = instance(body, module, name)
        _, _, ip_ports = ports((root / file).read_text(), module, params)
        cuts.append((module, name, dict((n, (d, w)) for d, w, n in ip_ports), connections))
        body = body[:first] + ' ' * (last - first) + body[last:]
    boundary = []
    assignments = []
    wrappers = []
    for module, name, ip_ports, connections in cuts:
        mapped = []
        for port, expression in connections:
            d, width = ip_ports[port]
            assert d != 'inout'
            if not expression:
                mapped.append((port, ''))
                continue
            # Pure top-level electrical passthrough stays entirely in Questa.
            direct = d == 'input' and identifiers(expression) <= top_inputs | set(params)
            if d == 'output' and identifiers(expression) <= top_outputs:
                names = identifiers(expression)
                direct = not any(re.search(r'\b' + n + r'\b', clean(body)) for n in names)
            if direct:
                mapped.append((port, expression))
                continue
            signal = 'cut__' + name + '__' + port
            nd = 'output' if d == 'input' else 'input'
            boundary.append((nd, width, signal))
            assignments.append(f'assign {signal}={expression};' if nd == 'output' else f'assign {expression}={signal};')
            mapped.append((port, signal))
        wrappers.append(module + ' ' + name + '(\n' + ',\n'.join('.' + p + '(' + e + ')' for p, e in mapped) + '\n);')
    if cpu_alu_only:
        cpu = (root/'EfxSapphireSoc.v').read_text()
        public_end = re.search(r'\bmodule\s+EfxCPUSp1\s*\(', cpu).start()
        public_cpu = cpu[:public_end]
        assert len(re.findall(r'\bEfxCPUSp1\s+\w+\s*\(', public_cpu)) == 1
        assert not re.search(r'\bEfxCPUSp2\s+\w+\s*\(', public_cpu)
        ao, ae, alu_ports = ports(cpu, 'EfxCPUSp1', params)
        stub = cpu[public_end:ae+2]+'\n'
        alu_connections=[]
        for direction, width, port in alu_ports:
            signal='cut__cpu_alu__'+port
            boundary.append(('input' if direction=='output' else 'output',width,signal))
            alu_connections.append('.'+port+'('+signal+')')
            if direction=='output':
                stub+=f'assign {port}=$root.board_public_native.{signal};\n'
            else:
                assignments.append(f'assign {signal}=cpu.system_cores_0_logic_cpu.EfxCPUSp1_inst.{port};')
        stub+='endmodule\n'
        (root/'EfxSapphireSoc_public.v').write_text(public_cpu+stub)
        wrappers.append('EfxCPUSp1 cpu_alu('+','.join(alu_connections)+');')
    # Warm video counter is the same already-declared initialization fixture.
    assert body.count('reg [26:0] dly_cnt') == 1
    body = body.replace('reg [26:0] dly_cnt', 'reg [26:0] dly_cnt /*verilator public_flat_rw*/')
    combined = clean(body + '\n'.join(assignments))
    used_ports = [(d, w, n) for d, w, n in top_ports if re.search(r'\b' + n + r'\b', combined)]
    # Keep every existing read-only measurement: export exact-width taps and
    # rewrite only their hierarchy, not their expressions, checks or timing.
    tb = (base / 'tb_board_startup.sv').read_text()
    assert sha(base / 'tb_board_startup.sv') == bm['tb_sha256']
    probe_file = base / 'board_measurement_probes.svh'
    probe = probe_file.read_text() if probe_file.exists() else ''
    refs = sorted(set(re.findall(r'\bdut\.([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)', clean(tb + '\n' + probe))))
    retained = tuple(n + '.' for m, n, f in islands)
    taps = {ref: 'probe__' + ref.replace('.', '__') for ref in refs if not ref.startswith(retained)}
    for ref, tap in taps.items():
        assignments.append(f'assign {tap}={ref};')
    native_ports = used_ports + boundary
    extra_decl = [f'output wire [$bits({ref})-1:0] {tap}' for ref, tap in taps.items()]
    native_header = source[:opening].replace('module ti60f225_oob_top', 'module board_public_native')
    native = native_header + '(\n' + ',\n'.join([declaration(*p) for p in native_ports] + extra_decl) + '\n);\n'
    native += body.replace('endmodule', '\n' + '\n'.join(assignments) + '\nendmodule')
    (root / 'board_public_native.sv').write_text(native, encoding='utf8')
    files = [root / Path(f).relative_to(base) for f in bm['source_files']]
    excluded_names = {'ti60f225_oob_top.v', 'cnn_production_frame_dpi.sv', 'tb_board_startup.sv',
                      'ddr3.v', 'camera_raw10_fixture.sv', 'ddr516_periphery_model.sv',
                      'ddr516_measurement_guard.sv', 'EFX_PLL_V3_FAST_PROTOCOL.v'}
    excluded_names.update(Path(f).name for m, n, f in islands)
    native_files = []
    protected = []
    for file in files:
        if file.name in {'tb_board_startup.sv', 'cnn_production_frame_dpi.sv'}:
            continue
        text = file.read_text(encoding='utf8', errors='replace')
        if re.search(r'`protected\b|\b(?:begin_protected|protected_cell)\b', text):
            protected.append(dict(path=str(file), sha256=sha(file)))
            continue
        if file.name not in excluded_names:
            native_files.append(file)
    # Restore the original production CNN, never nest the older DLL boundary.
    original_cnn = root / 'assembly/cnn_production_frame.sv'
    assert original_cnn.exists()
    native_files += [original_cnn, root / 'board_public_native.sv']
    if cpu_alu_only:
        native_files.append(root/'EfxSapphireSoc_public.v')
    inc = [root / Path(f).relative_to(base) for f in bm['include_dirs']]
    meta = dict(base=str(base), base_inputs_sha256=sha(base / 'inputs.json'), hardware_changed=False,
                whole_frame_pass=False, protected=protected, islands=islands,
                native_files=list(map(str, native_files)), include_dirs=list(map(str, inc)),
                native_ports=native_ports, taps=taps, wrapper_header=source[:end + 2],
                island_instances=wrappers, top_ports=top_ports, boundary=boundary,
                params=params, source_files=list(map(str, files)),cpu_alu_only=cpu_alu_only)
    (root / 'partition.json').write_text(json.dumps(meta, indent=2))
    return meta


def resolved_ports(header):
    found = []
    for m in re.finditer(r'VL_(IN|OUT)(?:8|16|64|W)?\s*\(\s*&?(\w+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*\d+)?\s*\)', header):
        found.append(('input' if m[1] == 'IN' else 'output', int(m[3]) - int(m[4]) + 1, m[2].replace('___05F', '__')))
    assert found
    return found


def packed_code(items, reading):
    code = []; offset = 0
    for name, width in items:
        name = name.replace('__', '___05F')
        if reading and width <= 64:
            code.append(f'm->{name}=get(data,{offset},{width});')
        else:
            for k in range((width + 31) // 32):
                count = min(32, width - 32 * k)
                value = f'm->{name}[{k}]' if width > 64 else f'(uint64_t(m->{name})>>{32*k})'
                code.append(f'm->{name}[{k}]=get(data,{offset+32*k},{count});' if reading else f'put(out,{offset+32*k},{count},{value});')
        offset += width
    return '\n'.join(code), offset


def bridge(root, meta):
    all_ports = resolved_ports((root / 'obj/Vboard_public_native.h').read_text())
    clock_names = {'process_clk_150', 'i_mipi_rx_pclk', 'vid_clk_dvi2', 'hdmi_tx_slow_clk',
                   'CLK_5M', 'CLK_25M', 'core_clk', 'jtag_inst1_TCK'}
    clocks = [(n, w) for d, w, n in all_ports if d == 'input' and n in clock_names]
    normal = [(n, w) for d, w, n in all_ports if d == 'input' and n not in clock_names]
    outputs = [(n, w) for d, w, n in all_ports if d == 'output']
    combo, ni = packed_code(normal, True); seq, nc = packed_code(clocks, True)
    publish, no = packed_code(outputs, False)
    cpp = r'''
#include "Vboard_public_native.h"
#include "Vboard_public_native___024root.h"
#include "verilated.h"
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <chrono>
using U=unsigned long long;
struct Model:Vboard_public_native {U sequence=0,combos=0,edges=0;double eval_seconds=0;Model(VerilatedContext*c):Vboard_public_native(c){};};
static uint64_t get(const unsigned*a,unsigned pos,unsigned width){
 uint64_t out=0;unsigned done=0;while(done<width){unsigned s=pos%32,n=std::min(width-done,32-s);
 out|=((uint64_t(a[pos/32])>>s)&((1ULL<<n)-1))<<done;done+=n;pos+=n;}return out;}
static void put(unsigned*a,unsigned pos,unsigned width,uint64_t value){
 unsigned done=0;while(done<width){unsigned s=pos%32,n=std::min(width-done,32-s);
 a[pos/32]|=unsigned((value>>done)&((1ULL<<n)-1))<<s;done+=n;pos+=n;}}
extern "C" __declspec(dllexport) void* board_public_create(){auto*c=new VerilatedContext;c->threads(1);return new Model(c);}
extern "C" __declspec(dllexport) void board_public_final(void*p){auto*m=static_cast<Model*>(p);auto*c=m->contextp();
 std::printf("BOARD_NATIVE_CALLS combo=%llu edge=%llu eval_seconds=%.9f\n",m->combos,m->edges,m->eval_seconds);m->final();delete m;delete c;}
extern "C" __declspec(dllexport) void board_public_warm(void*p){auto*m=static_cast<Model*>(p);m->rootp->board_public_native__DOT__dly_cnt=0x4000000;}
'''
    for kind, code, counter in [('combo', combo, 'combos'), ('seq', seq, 'edges')]:
        cpp += f'''
extern "C" __declspec(dllexport) U board_public_{kind}(void*p,U t,const unsigned*data,unsigned*out,U*due){{
 auto*m=static_cast<Model*>(p);++m->{counter};{code}
 if(t<m->contextp()->time()){{std::fprintf(stderr,"NATIVE_TIME_REVERSED\\n");std::abort();}}
 auto begin=std::chrono::steady_clock::now();
 m->contextp()->time(t);do{{m->eval();}}while(m->eventsPending()&&m->nextTimeSlot()==t);
 m->eval_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
 std::memset(out,0,{(no+31)//32*4});{publish}
 *due=m->eventsPending()?m->nextTimeSlot():~0ULL;return ++m->sequence;
}}
'''
    if (root/'obj/Vboard_public_native_board_public_native.h').exists():
        cpp='#include "Vboard_public_native_board_public_native.h"\n'+cpp
        cpp=cpp.replace('m->rootp->board_public_native__DOT__dly_cnt','m->rootp->board_public_native->dly_cnt')
    (root / 'board_public.cpp').write_text(cpp)
    sv = meta['wrapper_header'] + '\ntimeunit 1ps;timeprecision 1ps;\n'
    top_names = {n for d, w, n in meta['top_ports']}
    for d, w, n in all_ports:
        if n not in top_names:
            sv += f'wire [{w-1}:0] {n};\n'
    sv += '\n'.join(meta['island_instances']) + '\n'
    sv += '''import "DPI-C" function chandle board_public_create();
import "DPI-C" function void board_public_final(chandle h);
import "DPI-C" function void board_public_warm(chandle h);
'''
    for kind, width in [('combo', ni), ('seq', nc)]:
        sv += f'import "DPI-C" function longint unsigned board_public_{kind}(chandle h,longint unsigned t,input bit [{width-1}:0] data,output bit [{no-1}:0] result,output longint unsigned due);\n'
    sv += f'''
chandle h;
wire [{ni-1}:0] inputs={{{','.join(n for n,w in reversed(normal))}}};
wire [{nc-1}:0] clocks={{{','.join(n for n,w in reversed(clocks))}}};
bit [{no-1}:0] combo_out,seq_out,temp;
longint unsigned combo_num=0,seq_num=0,combo_due,seq_due,due,scheduled=64'hffffffffffffffff,wakeup=64'hffffffffffffffff;
reg tick=0;
assign {{{','.join(n for n,w in reversed(outputs))}}}=(seq_num>combo_num)?seq_out:combo_out;
initial h=board_public_create();
always_comb combo_num=board_public_combo(h,$time,inputs,combo_out,combo_due);
always @(clocks or tick)begin
 seq_num<=board_public_seq(h,$time,clocks,temp,seq_due);seq_out<=temp;
end
always @(combo_num or seq_num)begin
 due=(seq_num>combo_num)?seq_due:combo_due;
 if(due!=64'hffffffffffffffff && due!=scheduled)begin
  if(due<$time)$fatal(1,"Board native time reversed");
  scheduled=due;wakeup<=#(due-$time)due;
 end
end
always @(wakeup)if(wakeup==$time)tick=~tick;
task warm_video;board_public_warm(h);tick=~tick;endtask
// DDR response data may be X while invalid. Never coerce a valid response.
always @(posedge process_clk_150)if(probe__w_arstn && probe__cal_done)begin
 if($isunknown({{cut__u_ddr3_top__s_axi_rvalid,cut__u_ddr3_top__s_axi_bvalid,
 cut__u_ddr3_top__s_axi_awready,cut__u_ddr3_top__s_axi_arready,cut__u_ddr3_top__s_axi_wready}}))$fatal(1,"Board native DDR control X");
 if(cut__u_ddr3_top__s_axi_rvalid && $isunknown({{cut__u_ddr3_top__s_axi_rdata,cut__u_ddr3_top__s_axi_rid,
 cut__u_ddr3_top__s_axi_rresp,cut__u_ddr3_top__s_axi_rlast}}))$fatal(1,"Board native DDR valid R X");
 if(cut__u_ddr3_top__s_axi_bvalid && $isunknown({{cut__u_ddr3_top__s_axi_bid,cut__u_ddr3_top__s_axi_bresp}}))$fatal(1,"Board native DDR valid B X");
end
final board_public_final(h);
endmodule
'''
    (root / 'board_public_wrapper.sv').write_text(sv, encoding='utf8')
    base = Path(meta['base'])
    def rewrite(text):
        text = text.replace(base.as_posix(), root.as_posix()).replace(str(base), str(root))
        text = text.replace("dut.dly_cnt=27'h4000000;", 'dut.warm_video();')
        return re.sub(r'\bdut\.([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)',
                      lambda m: 'dut.' + meta['taps'].get(m[1], m[1]), text)
    (root / 'tb_board_startup.sv').write_text(rewrite((base / 'tb_board_startup.sv').read_text()), encoding='utf8')
    (root / 'board_measurement_probes.svh').write_text(rewrite((base / 'board_measurement_probes.svh').read_text()), encoding='utf8')
    meta['packing'] = dict(input_bits=ni, clock_bits=nc, output_bits=no)
    meta['resolved_ports'] = all_ports
    (root / 'partition.json').write_text(json.dumps(meta, indent=2))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True); p.add_argument('--resume', action='store_true')
    p.add_argument('--generate-only', action='store_true'); p.add_argument('--compile-only', action='store_true')
    p.add_argument('--cpu-alu-only',action='store_true')
    p.add_argument('--relink-only',action='store_true')
    a = p.parse_args(); root = a.root.resolve(); base = a.base.resolve()
    assert root.is_relative_to(ROOT / 'tmp')
    if not a.resume:
        assert not root.exists(); root.mkdir()
        meta = generate_partition(base, root,a.cpu_alu_only)
    elif (root / 'partition.json').exists():
        meta = json.loads((root / 'partition.json').read_text())
    else:
        assert not (root / 'result.json').exists()
        meta = generate_partition(base, root,a.cpu_alu_only)
    if a.generate_only:
        print('PUBLIC_BOARD_PARTITION_GENERATED_NOT_VALIDATED', flush=True); return
    env = os.environ.copy()
    env.update(PATH=str(BIN)+';E:/GIT/Git/usr/bin;'+env['PATH'], VERILATOR_ROOT=VR.as_posix(), MSYS2_ARG_CONV_EXCL='*',
               SALT_LICENSE_SERVER='E:/quatra/LR-195771_License.dat')
    obj = root / 'obj'
    if not a.relink_only:
        run([BIN/'verilator_bin.exe', '--cc', '--timing', '--sched-zero-delay', '--top-module', 'board_public_native',
             *['-I'+Path(x).as_posix() for x in meta['include_dirs']], '--timescale-override', '1ns/1ps',
             '--threads', '1', '--output-split', '15000', '--output-split-cfuncs', '500', '-Wno-fatal',
             '--Mdir', obj, '-CFLAGS', '-DVL_TIME_CONTEXT -fPIC', *meta['native_files']], root, 'verilate', env, 180)
    bridge(root, meta)
    if a.compile_only:
        print('PUBLIC_BOARD_VERILATE_PASS_NOT_VALIDATED', flush=True); return
    if not a.relink_only:
        run([BIN/'mingw32-make.exe', '-f', 'Vboard_public_native.mk', '-j4', 'Vboard_public_native__ALL.a',
             'verilated.o', 'verilated_threads.o', 'verilated_timing.o', 'CXX=g++.exe', 'AR=ar.exe',
             'PERL=E:/GIT/Git/usr/bin/perl.exe', 'SHELL=E:/GIT/Git/usr/bin/sh.exe', 'VERILATOR_ROOT='+VR.as_posix(),
             'OPT_FAST=-O2', 'OPT_GLOBAL=-O2', 'CXXFLAGS=-std=gnu++20 -DVL_TIME_CONTEXT'], obj, 'build', env, 600)
    run([BIN/'g++.exe', '-O2', '-std=c++20', '-DVL_TIME_CONTEXT', '-I'+obj.as_posix(), '-I'+(VR/'include').as_posix(),
         '-I'+(VR/'include/vltstd').as_posix(), root/'board_public.cpp', obj/'Vboard_public_native__ALL.a',
         obj/'verilated.o', obj/'verilated_threads.o', obj/'verilated_timing.o', '-pthread', '-latomic',
         '-shared', '-o', root/'board_public.dll'], root, 'link', env, 120)
    sources = [f for f in meta['source_files'] if Path(f).name not in {'ti60f225_oob_top.v','cnn_production_frame_dpi.sv'}]
    sources.append(str(root/'board_public_wrapper.sv'))
    run([SIM/'vlib.exe', 'work'], root, 'vlib', env, 20)
    run([SIM/'vlog.exe', '-sv', '-timescale', '1ns/1ps', '+define+den2048Mb+sg125+x16',
         *['+incdir+'+x for x in meta['include_dirs']], *sources], root, 'compile', env, 120)
    result = dict(status='PUBLIC_BOARD_PARTITION_BUILT_NOT_VALIDATED', packing=meta['packing'],
                  partition_sha256=sha(root/'partition.json'), dll_sha256=sha(root/'board_public.dll'))
    (root/'result.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
