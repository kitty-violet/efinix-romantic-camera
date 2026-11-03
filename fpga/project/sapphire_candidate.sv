module sapphire_candidate (
input wire clk, reset, interrupt_request,
output wire system_reset,
output wire [15:0] paddr, output wire [31:0] pwdata,
output wire psel, penable, pwrite,
input wire pready, pslverror, input wire [31:0] prdata
);
EfxSapphireSoc soc (
.io_systemClk(clk), .io_asyncReset(reset), .io_systemReset(system_reset),
.jtagCtrl_tck(clk), .jtagCtrl_tdi(1'b0), .jtagCtrl_enable(1'b0),
.jtagCtrl_capture(1'b0), .jtagCtrl_shift(1'b0), .jtagCtrl_update(1'b0),
.jtagCtrl_reset(1'b1), .jtagCtrl_tdo(),
.userInterruptA(interrupt_request),
.io_apbSlave_0_PADDR(paddr), .io_apbSlave_0_PSEL(psel),
.io_apbSlave_0_PENABLE(penable), .io_apbSlave_0_PWRITE(pwrite),
.io_apbSlave_0_PWDATA(pwdata), .io_apbSlave_0_PREADY(pready),
.io_apbSlave_0_PRDATA(prdata), .io_apbSlave_0_PSLVERROR(pslverror)
);
endmodule
