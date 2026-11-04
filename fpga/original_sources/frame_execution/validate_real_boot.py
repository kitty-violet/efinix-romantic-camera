"""Real vendor CPU + real JTAG + width adapter + APB + coefficient RAM test.

The external 128-bit memory is explicitly a stalled AXI testbench model, not
the DDR controller/PHY. All bytes enter through actual external JTAG signals.
No testbench APB commands, CPU replacement, hierarchical writes or RAM preload.
"""
import argparse
import os
import json
import re
import shutil
import sys
import traceback
from pathlib import Path
import zlib
from run_units import ROOT, HERE, EF, MS, sha, run
sys.path.insert(0, str(HERE / "board_integration"))
from boot_image import write_image
from observe_board_boot import MODEL


def make_tb():
    old = (HERE / "tb_cnn_apb_control.sv").read_text()
    declarations = old[old.index(" wire pready,"):old.index(" reg [63:0] words")]
    store = old[old.index(" cnn_apb_control dut("):old.index(" task access(")]
    store = store.replace("cnn_apb_control dut(", "cnn_apb_control control(")
    store = store.replace(".committed_frames(32'd7)", ".committed_frames(commits)")
    store = store.replace(".pipeline_error(1'b0)", ".pipeline_error(1'b0)")
    store = store.replace(".cancel(1'b0)", ".cancel(cancel)")
    cpu_old = (HERE.parent / "sapphire_memory/tb_cpu_diagnostic.sv").read_text()
    cpu = cpu_old[cpu_old.index("    EfxSapphireSoc dut ("):cpu_old.index("    initial begin")]
    cpu = cpu.replace(".jtagCtrl_tck(clk)", ".jtagCtrl_tck(jtck)")
    for name, wire in (("tdi", "jtdi"), ("enable", "jsel"), ("capture", "jcapture"), ("shift", "jshift"), ("update", "jupdate"), ("reset", "jreset")):
        cpu = cpu.replace(f".jtagCtrl_{name}(1'b0)", f".jtagCtrl_{name}({wire})")
    cpu = cpu.replace(".userInterruptA", ".jtagCtrl_tdo(jtdo),.axiA_awburst(awburst),.axiA_arburst(arburst),.userInterruptA")
    cpu = cpu.replace(".io_apbSlave_0_PRDATA(32'h0)", ".io_apbSlave_0_PRDATA(prdata)")
    cpu = cpu.replace(".io_apbSlave_0_PSLVERROR(1'b0)", ".io_apbSlave_0_PSLVERROR(pslverror)")
    cpu = cpu.replace(".axiA_bresp(2'b00)", ".axiA_bresp(bresp)").replace(".axiA_rresp(2'b00)", ".axiA_rresp(rresp)").replace(".axiA_rlast(1'b1)", ".axiA_rlast(rlast)")
    # Explicitly leave unused output ports open. Missing inputs are an error.
    header = MODEL.read_text().split(");", 1)[0]
    for direction, name in re.findall(r"\b(input|output)\s+wire\s*(?:\[[^]]+\]\s*)?(\w+)", header):
        if not re.search(r"\." + name + r"\(", cpu):
            assert direction == "output", name
            cpu = cpu.replace(".userInterruptA", f".{name}(),.userInterruptA")
    # Instantiate the exact adapter, deriving links from its ANSI header.
    adapter = (HERE.parent / "ddr_fabric/sapphire_axi32_to_axi128.sv").read_text().split(");", 1)[0]
    ports = re.findall(r"\b(input|output)\s+(?:wire|reg)\s*(\[[^]]+\])?\s*(\w+)", adapter)
    decl, links = [], []
    for direction, width, name in ports:
        if name == "clk": target = "clk"
        elif name == "rst_n": target = "!reset&&!sys_reset"
        elif name.startswith("s_axi_"): target = name[6:]
        else:
            target = name
            kind = "reg" if name in ("m_axi_rdata", "m_axi_rvalid", "m_axi_bvalid") else "wire"
            decl.append(f"{kind} {width} {name}" + ("=0" if kind == "reg" else "") + ";")
        links.append(f".{name}({target})")
    return '''`timescale 1ns/1ps
module tb_real_boot;
reg clk=0,reset=1;always #3.333333 clk=~clk;
wire sys_reset;
reg jtck=0,jtdi=0,jsel=0,jcapture=0,jshift=0,jupdate=0,jreset=0;wire jtdo;
wire awvalid,wvalid,wlast,bready,arvalid,rready,awready,wready,arready,bvalid,rvalid,rlast;
wire [31:0] awaddr,wdata,araddr,rdata;wire [7:0] awid,awlen,arid,arlen,bid,rid;
wire [2:0] awsize,arsize;wire [1:0] awburst,arburst,bresp,rresp;wire [3:0] wstrb;
wire [15:0] paddr;wire psel,penable,pwrite;wire [31:0] pwdata;
reg [31:0] commits=0;
''' + declarations + cpu + store + "\n" + "\n".join(decl) + '''
sapphire_axi32_to_axi128 #(.ADDRESS_BASE(32'he1000000),.ADDRESS_LIMIT(32'he1ffffff),.DDR_BASE(32'h04000000)) adapter(
''' + ",".join(links) + ''');
// Small, disjoint memory windows; no aliases and no preinitialized DDR bytes.
reg [7:0] memory[0:20479];reg [127:0] descriptor_probe0,descriptor_probe1;
reg [31:0] rng=32'h13059ade,wa=0,ra=0;reg have_aw=0,have_w=0,read_pending=0;
reg [127:0] wd=0;reg [15:0] ws=0;
integer cycles=0,loads=0,diags=0,verifies=0,ddr_writes=0,ddr_reads=0,stalls=0,apb=0;
reg [31:0] phase=0;integer smoke=0;
assign m_axi_awready=!have_aw&&!m_axi_bvalid&&rng[0];
assign m_axi_wready=!have_w&&!m_axi_bvalid&&rng[1];
assign m_axi_arready=!read_pending&&!m_axi_rvalid&&rng[2];
assign m_axi_rid=0;assign m_axi_bid=0;assign m_axi_rlast=1;
assign m_axi_rresp=0;assign m_axi_bresp=0;
wire [31:0] cpu_da=dut.system_cores_0_logic_cpu_dBus_cmd_payload_address;
wire [31:0] cpu_dd=dut.system_cores_0_logic_cpu_dBus_cmd_payload_data;
wire cpu_store=dut.system_cores_0_logic_cpu_dBus_cmd_valid&&dut.system_cores_0_dBus_cmd_ready&&dut.system_cores_0_logic_cpu_dBus_cmd_payload_wr;
function automatic bit allowed(input [31:0] address);
 allowed=(address>=32'h04200000&&address<32'h04205000)||address==32'h04000000||address==32'h04100000;
endfunction
always @(posedge clk)begin
 cycles<=cycles+1;rng<={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};
 if(cycles>40000000)$fatal(1,"real boot watchdog phase=%h loads=%0d diags=%0d",phase,loads,diags);
 if(reset||sys_reset)begin have_aw<=0;have_w<=0;read_pending<=0;m_axi_bvalid<=0;m_axi_rvalid<=0;phase<=0;end
 else begin
  if(cpu_store&&cpu_da==32'hf9000f04)begin phase<=cpu_dd;$display("BOOT_PHASE cycle=%0d value=%h loads=%0d diags=%0d",cycles,cpu_dd,loads,diags);$fflush();end
  if(cpu_store&&(cpu_da==32'hf9000f20||cpu_da==32'hf9000f24))begin $display("TRAP_MAILBOX addr=%h data=%h",cpu_da,cpu_dd);$fflush();end
  if(psel&&penable&&pready)begin apb<=apb+1;if(pslverror)$fatal(1,"APB firmware rejected addr=%h",paddr);end
  if(load_valid&&load_ready)loads<=loads+1;
  if(diag_valid&&diag_accept)diags<=diags+1;
  if(verify_valid&&verify_ready)verifies<=verifies+1;
  if(protocol_error)$fatal(1,"width adapter protocol error");
  if((m_axi_awvalid&&!m_axi_awready)||(m_axi_wvalid&&!m_axi_wready)||(m_axi_arvalid&&!m_axi_arready))stalls<=stalls+1;
  if(m_axi_awvalid&&m_axi_awready)begin
   if(!allowed(m_axi_awaddr)||m_axi_awlen!=0||m_axi_awsize!=4)$fatal(1,"bounded AW %h",m_axi_awaddr);
   wa<=m_axi_awaddr;have_aw<=1;
  end
  if(m_axi_wvalid&&m_axi_wready)begin if(!m_axi_wlast)$fatal(1,"WLAST");wd<=m_axi_wdata;ws<=m_axi_wstrb;have_w<=1;end
  if(have_aw&&have_w&&!m_axi_bvalid)begin
   for(integer b=0;b<16;b=b+1)if(ws[b])begin
    if(wa==32'h04000000)descriptor_probe0[b*8+:8]<=wd[b*8+:8];
    else if(wa==32'h04100000)descriptor_probe1[b*8+:8]<=wd[b*8+:8];
    else memory[wa-32'h04200000+b]<=wd[b*8+:8];
   end
   have_aw<=0;have_w<=0;m_axi_bvalid<=1;ddr_writes<=ddr_writes+1;
  end
  if(m_axi_bvalid&&m_axi_bready)m_axi_bvalid<=0;
  if(m_axi_arvalid&&m_axi_arready)begin
   if(!allowed(m_axi_araddr)||m_axi_arlen!=0||m_axi_arsize!=4)$fatal(1,"bounded AR %h",m_axi_araddr);
   ra<=m_axi_araddr;read_pending<=1;ddr_reads<=ddr_reads+1;
  end
  if(read_pending&&!m_axi_rvalid&&rng[3])begin
   for(integer b=0;b<16;b=b+1)begin
    if(ra==32'h04000000)m_axi_rdata[b*8+:8]<=descriptor_probe0[b*8+:8];
    else if(ra==32'h04100000)m_axi_rdata[b*8+:8]<=descriptor_probe1[b*8+:8];
    else m_axi_rdata[b*8+:8]<=memory[ra-32'h04200000+b];
   end
   m_axi_rvalid<=1;read_pending<=0;
  end
  if(m_axi_rvalid&&m_axi_rready)m_axi_rvalid<=0;
 end
end
`include "jtag_memory_tasks.svh"
`include "boot_vectors.vh"
task reboot;
 begin
  @(negedge clk);reset=1;commits=0;host_halted=0;host_address_valid=0;host_data_valid=0;repeat(20)@(negedge clk);reset=0;
  wait(phase==1);repeat(20)@(negedge clk);
  jexpect(32'hf9000f04,1);jresume();
  if(enable||coeff_ready)$fatal(1,"reset eligibility");
 end
endtask
initial begin
 if($value$plusargs("SMOKE=%d",smoke))begin end
 reboot();
 jwrite(32'he1000000,32'h12345678);jexpect(32'he1000000,32'h12345678);
 jwrite(32'he1100000,32'h87654321);jexpect(32'he1100000,32'h87654321);
 if(descriptor_probe0[31:0]!==32'h12345678||descriptor_probe1[31:0]!==32'h87654321)$fatal(1,"descriptor map");
 $display("PASS JTAG_REAL_PATH descriptor_addresses=2 actual_cpu=1 actual_adapter=1");
 // Invalid command must never enable or access a coefficient byte.
 jwrite(32'hf9000f00,32'hbad00001);jresume();wait(phase[31]);
 jexpect(32'hf9000f04,32'h80000001);
 if(enable||loads!=0)$fatal(1,"bad header accepted");
 if(smoke)begin $display("PASS REAL_BOOT_SMOKE");$finish;end
 reboot();
 // Host payload enters through the hard-JTAG boundary, including zero padding.
 for(integer i=0;i<5120;i=i+1)begin
  jwrite(32'he1200000+4*i,upload_word(i));
  if(i%512==0)begin $display("UPLOAD_PROGRESS words=%0d cycles=%0d",i,cycles);$fflush();end
 end
 jwrite(32'hf9000f08,32'hc321);jwrite(32'hf9000f0c,EXPECTED_CRC);
 jwrite(32'hf9000f10,32'h12345678);jwrite(32'hf9000f14,32'h89abcdef);
 jwrite(32'hf9000f18,3);jwrite(32'hf9000f00,32'h434e4e31);jresume();
 wait(phase==4||phase[31]);
 if(phase!=4||!enable||!coeff_ready||loads!=2560||diags!=2560||verifies!=1)$fatal(1,"real boot phase=%h enable=%b coeff_ready=%b load=%0d readback=%0d verify=%0d",phase,enable,coeff_ready,loads,diags,verifies);
 if(compute_crc32!==32'h12345678||dma_crc32!==32'h89abcdef)$fatal(1,"CRC registers");
 jexpect(32'hf9000f1c,EXPECTED_CRC);jresume();
 @(negedge clk);commits=3;wait(phase==5);
 if(enable)$fatal(1,"bounded frame stop");
 $display("PASS REAL_BOOT_VALID words=2560 readbacks=2560 frame_auto_stop=1 cycles=%0d stalls=%0d",cycles,stalls);
 // CPU/common reset must revoke old permission. DDR contents persist, as real
 // external DDR does; mismatch the expected CRC and prove no second enable.
 reboot();
 jwrite(32'hf9000f08,32'hc322);jwrite(32'hf9000f0c,EXPECTED_CRC^1);
 jwrite(32'hf9000f18,3);jwrite(32'hf9000f00,32'h434e4e31);jresume();
 wait(phase[31]);jexpect(32'hf9000f04,32'h80000002);
 if(enable||coeff_ready||loads!=5120||diags!=2560||verifies!=1)$fatal(1,"bad CRC eligibility");
 if(stalls==0||ddr_writes!=5122)$fatal(1,"coverage");
 $display("PASS REAL_BOOT_COMPLETE crc_reject=1 reset_revoke=1 actual_jtag=1 actual_cpu=1 ddr_phy=0 cycles=%0d apb=%0d",cycles,apb);
 $finish;
end
endmodule
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--sim-bin", type=Path)
    p.add_argument("--license-file", type=Path)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--cpu-source", type=Path)
    p.add_argument("--frozen-base", type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    assert root.is_relative_to(ROOT / "tmp") and not root.exists()
    root.mkdir(parents=True)
    sim=a.sim_bin.resolve() if a.sim_bin else MS
    if a.license_file:
        assert a.license_file.is_file()
        os.environ['SALT_LICENSE_SERVER']=str(a.license_file.resolve())
    try:
        image = write_image(root / "firmware")
        board = HERE / 'board_integration/ti60f225_oob_top.v'
        assert ".CPU_ADDRESS_BASE(32'he1000000),.CPU_ADDRESS_LIMIT(32'he1ffffff),.CPU_DDR_BASE(32'h04000000)" in board.read_text()
        for f in (root / "firmware").glob("*.bin"):
            if f.name != "boot.bin": shutil.copy2(f, root / f.name)
        frozen = ROOT / "tmp/frame-execution/connected-board-sharing1/assembly/EfxSapphireSoc.v"
        attr = '(* syn_ramstyle = "registers" *) '
        assert frozen.read_text().split("module EfxCPUSp1")[0].replace(attr, "") == MODEL.read_text().split("module EfxCPUSp1")[0].replace(attr, "")
        sources = [MODEL, EF / "sim_models/verilog/efx_ram10.v",
                   ROOT / "tmp/frame-execution/program-map2/resident_weight_bank64.sv",
                   ROOT / "tmp/frame-execution/program-map2/coefficient_bank_store_fast.sv",
                   HERE / "cnn_apb_control.sv", HERE.parent / "ddr_fabric/sapphire_axi32_to_axi128.sv",
                   HERE / "board_integration/jtag_memory_tasks.svh"]
        if a.frozen_base:
            from frozen_sources import frozen_paths
            sources = frozen_paths(sources, a.frozen_base)
            # The official CPU simulation view is qualified separately below.
            sources[0] = MODEL
        cpu_qualification = None
        if a.cpu_source:
            from prune_unused_sapphire import prune
            assert a.frozen_base is not None
            cpu = a.cpu_source.resolve()
            assert cpu.is_relative_to(HERE/'variants/unused_peripherals/model')
            board_cpu = cpu.parent.parent/'EfxSapphireSoc.v'
            original = a.frozen_base/'assembly/EfxSapphireSoc.v'
            assert board_cpu.read_text() == prune(original.read_text())
            assert cpu.read_text() == prune(MODEL.read_text())
            def public(t): return t.split('module EfxCPUSp1')[0].replace(attr, '')
            assert public(board_cpu.read_text()) == public(cpu.read_text())
            sources[0] = cpu
            cpu_qualification = dict(board_source=str(board_cpu), board_sha256=sha(board_cpu),
                reference=str(original), reference_sha256=sha(original), transform_sha256=sha(HERE/'prune_unused_sapphire.py'),
                public_wrapper_equal=True, encrypted_cpu_text_unchanged=True)
        records = []
        for src in sources:
            dst = root / src.name
            shutil.copy2(src, dst)
            records.append(dict(source=str(src), snapshot=str(dst), sha256=sha(src)))
        words = [int(x, 16) for x in (ROOT / "tmp/frame-execution/joint-reorder-vectors1/load_words.hex").read_text().split()]
        assert len(words) == 2560 and all(w == 0 for w in words[2224:])
        packed = b"".join(w.to_bytes(8, "little") for w in words)
        crc = zlib.crc32(packed)
        # Case ROM is a HOST fixture, never a synthesized coefficient source.
        vector = f"localparam [31:0] EXPECTED_CRC=32'h{crc:08x};\nfunction automatic [31:0] upload_word(input integer i);\nbegin case(i)\n"
        vector += "\n".join(f"{i}:upload_word=32'h{int.from_bytes(packed[i*4:i*4+4],'little'):08x};" for i in range(5120))
        vector += "\ndefault:upload_word=32'hx;endcase end endfunction\n"
        (root / "boot_vectors.vh").write_text(vector)
        (root / "tb_real_boot.sv").write_text(make_tb())
        (root / "inputs.json").write_text(json.dumps(dict(inputs=records, image=image, runner_sha256=sha(__file__),
            tb_sha256=sha(root / "tb_real_boot.sv"), fixture_sha256=sha(root / "boot_vectors.vh"), smoke=a.smoke,simulator=str(sim),
            cpu_qualification=cpu_qualification), indent=2))
        if a.prepare_only:
            print(json.dumps(dict(status='PREPARED_NOT_RUN',image=image)),flush=True);return
        run([sim / "vlib.exe", "work"], root, "vlib.log")
        run([sim / "vlog.exe", "-sv", "-timescale", "1ns/1ps", *[r["snapshot"] for r in records if not r["snapshot"].endswith(".svh")], root / "tb_real_boot.sv"], root, "compile.log")
        body, seconds = run([sim / "vsim.exe", "-c", *([] if a.sim_bin else ['-novopt']), "work.tb_real_boot", "+SMOKE=" + str(int(a.smoke)),
             "-do", "onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0"], root, "simulation.log", timeout=10800)
        marker = "PASS REAL_BOOT_SMOKE" if a.smoke else "PASS REAL_BOOT_COMPLETE"
        assert marker in body
        for r in records: assert sha(r["source"]) == sha(r["snapshot"]) == r["sha256"]
        result = dict(status=marker.replace(" ", "_"), lines=re.findall(r"^# (PASS .*)$", body, re.M), seconds=seconds,
                      inputs_sha256=sha(root / "inputs.json"), log_sha256=sha(root / "simulation.log"), full_board=False, ddr_phy=False, board_release=False)
        (root / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    except Exception as e:
        (root / "failure.json").write_text(json.dumps(dict(error=repr(e), traceback=traceback.format_exc()), indent=2))
        raise


if __name__ == "__main__": main()
