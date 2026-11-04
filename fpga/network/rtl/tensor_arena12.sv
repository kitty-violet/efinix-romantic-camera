// Actual Ti60 true-dual-port word bank. EFX_DPRAM10 has at most10 bits
// per port; seven primitives hold64 bits. Both ports can read/write, with
// explicit same-address write collision serialization. No fictitious third port.
module tensor_dual_bank #(parameter integer ZERO_WRITE_RESPONSE=1)(
 input wire clk,reset,cancel,
 input wire [1:0] req_valid,output wire [1:0] req_ready,input wire [1:0] req_write,
 input wire [19:0] req_address,input wire [127:0] req_data,input wire [31:0] req_tag,
 output wire [1:0] rsp_valid,input wire [1:0] rsp_ready,
 output wire [127:0] rsp_data,output wire [31:0] rsp_tag,
 output reg [1:0] rsp_was_write
);
 assign rsp_tag=32'd0;
 reg [1:0] valid_q;
 wire active=!reset && !cancel;
 assign rsp_valid=valid_q & {2{active}};
 wire a_ready=active && (!valid_q[0] || rsp_ready[0]);
 wire fire_a=req_valid[0] && a_ready;
 wire collision=fire_a && req_address[9:0]==req_address[19:10] && (req_write[0] || req_write[1]);
 assign req_ready[0]=a_ready;
 assign req_ready[1]=active && (!valid_q[1] || rsp_ready[1]) && !collision;
 wire [1:0] fire=req_valid & req_ready;
 wire [69:0] a_data={6'd0,req_data[63:0]},b_data={6'd0,req_data[127:64]};
 wire [69:0] a_out,b_out;
 assign rsp_data[63:0]=(ZERO_WRITE_RESPONSE!=0 && rsp_was_write[0]) ? 64'd0 : a_out[63:0];
 assign rsp_data[127:64]=(ZERO_WRITE_RESPONSE!=0 && rsp_was_write[1]) ? 64'd0 : b_out[63:0];
 for(genvar n=0;n<7;n=n+1)begin:RAM
  EFX_DPRAM10 #(.READ_WIDTH_A(10),.WRITE_WIDTH_A(10),.READ_WIDTH_B(10),.WRITE_WIDTH_B(10),
   .OUTPUT_REG_A(1'b0),.OUTPUT_REG_B(1'b0),.WRITE_MODE_A("READ_FIRST"),.WRITE_MODE_B("READ_FIRST")) mem(
   .CLKA(clk),.WEA(req_write[0]),.CLKEA(fire[0]),.RSTA(1'b0),.ADDRENA(1'b1),
   .WDATAA(a_data[n*10+:10]),.ADDRA(req_address[9:0]),.RDATAA(a_out[n*10+:10]),
   .CLKB(clk),.WEB(req_write[1]),.CLKEB(fire[1]),.RSTB(1'b0),.ADDRENB(1'b1),
   .WDATAB(b_data[n*10+:10]),.ADDRB(req_address[19:10]),.RDATAB(b_out[n*10+:10]));
 end
 always @(posedge clk)begin
  if(reset || cancel)begin valid_q<=0;rsp_was_write<=0;end
  else for(integer p=0;p<2;p=p+1)begin
   if(rsp_ready[p])valid_q[p]<=0;
   if(fire[p])begin valid_q[p]<=1;rsp_was_write[p]<=req_write[p];end
  end
 end
endmodule
// Fixed local response contract: one accepted transaction per physical port.
// The occupied port and retained router owner identify the response. No echo tag.
// Read/write direction, collision serialization and held response remain checked.
module tensor_arena12 #(parameter integer ZERO_WRITE_RESPONSE=1)(
 input wire clk,reset,cancel,
 input wire [23:0] req_valid,output wire [23:0] req_ready,input wire [23:0] req_write,
 input wire [239:0] req_address,input wire [1535:0] req_data,input wire [383:0] req_tag,
 output wire [23:0] rsp_valid,input wire [23:0] rsp_ready,
 output wire [1535:0] rsp_data,output wire [383:0] rsp_tag,output wire [23:0] rsp_was_write
);
 for(genvar b=0;b<12;b=b+1)begin:BANK
  tensor_dual_bank #(.ZERO_WRITE_RESPONSE(ZERO_WRITE_RESPONSE)) bank(.clk(clk),.reset(reset),.cancel(cancel),
   .req_valid(req_valid[b*2+:2]),.req_ready(req_ready[b*2+:2]),.req_write(req_write[b*2+:2]),
   .req_address(req_address[b*20+:20]),.req_data(req_data[b*128+:128]),.req_tag(req_tag[b*32+:32]),
   .rsp_valid(rsp_valid[b*2+:2]),.rsp_ready(rsp_ready[b*2+:2]),.rsp_data(rsp_data[b*128+:128]),
   .rsp_tag(rsp_tag[b*32+:32]),.rsp_was_write(rsp_was_write[b*2+:2]));
 end
endmodule
