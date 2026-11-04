// Structural candidate: six row banks, each a pair of 64-bit pixel banks.
// The total storage and primitive count remain 6*2*1024*64 and84 DPRAM10.
// Port A reads both adjacent pixel halves; port B reads/writes with independent
// half enables. No third RAM port and no private payload response register.
// The caller must retain valid/address/data until acceptance. Cancel is a
// shared arena/owner cancellation; it never revokes an external DDR transfer.
module row_pair_arena6 #(parameter integer EXTERNAL_WRITE_COLLISION=0)(
 input wire clk,reset,cancel,
 input wire [5:0] aux_write_collision,
 input wire [5:0] read_valid,output wire [5:0] read_ready,
 input wire [59:0] read_address,
 output wire [5:0] read_response,input wire [5:0] read_response_ready,
 output wire [767:0] read_data,
 input wire [5:0] aux_valid,output wire [5:0] aux_ready,
 input wire [59:0] aux_address,input wire [11:0] aux_write_half,
 input wire [767:0] aux_data,
 output wire [5:0] aux_response,input wire [5:0] aux_response_ready,
 output wire [767:0] aux_response_data,output reg [5:0] aux_was_write
);
 reg [5:0] read_full,aux_full;
 wire active=!reset && !cancel;
 assign read_response=read_full & {6{active}};
 assign aux_response=aux_full & {6{active}};
 for(genvar row=0;row<6;row=row+1)begin:ROW
  assign read_ready[row]=active && (!read_full[row] || read_response_ready[row]);
  wire rfire=read_valid[row] && read_ready[row];
  wire selected_collision=rfire && (|aux_write_half[row*2+:2]) &&
   read_address[row*10+:10]==aux_address[row*10+:10];
  wire write_collision=EXTERNAL_WRITE_COLLISION!=0 ? aux_write_collision[row]:selected_collision;
  // synthesis translate_off
  always @(posedge clk)if(active && EXTERNAL_WRITE_COLLISION!=0)begin
   if(write_collision!==selected_collision)$fatal(1,"ARENA_EARLY_COLLISION_MISMATCH row=%0d",row);
  end
  // synthesis translate_on
  assign aux_ready[row]=active && (!aux_full[row] || aux_response_ready[row]) && !write_collision;
  wire afire=aux_valid[row] && aux_ready[row];
  for(genvar half=0;half<2;half=half+1)begin:HALF
   wire [69:0] wd={6'd0,aux_data[row*128+half*64+:64]};
   wire [69:0] ra,rb;
   assign read_data[row*128+half*64+:64]=ra[63:0];
   assign aux_response_data[row*128+half*64+:64]=rb[63:0];
   for(genvar part=0;part<7;part=part+1)begin:SLICE
    EFX_DPRAM10 #(.READ_WIDTH_A(10),.WRITE_WIDTH_A(10),.READ_WIDTH_B(10),.WRITE_WIDTH_B(10),
     .OUTPUT_REG_A(1'b0),.OUTPUT_REG_B(1'b0),.WRITE_MODE_A("READ_FIRST"),.WRITE_MODE_B("READ_FIRST")) mem(
     .CLKA(clk),.WEA(1'b0),.CLKEA(rfire),.RSTA(1'b0),.ADDRENA(1'b1),
     .WDATAA(10'd0),.ADDRA(read_address[row*10+:10]),.RDATAA(ra[part*10+:10]),
     .CLKB(clk),.WEB(aux_write_half[row*2+half]),.CLKEB(afire),.RSTB(1'b0),.ADDRENB(1'b1),
     .WDATAB(wd[part*10+:10]),.ADDRB(aux_address[row*10+:10]),.RDATAB(rb[part*10+:10]));
   end
  end
  always @(posedge clk)begin
   if(reset || cancel)begin read_full[row]<=0;aux_full[row]<=0;aux_was_write[row]<=0;end
   else begin
    if(read_response_ready[row])read_full[row]<=0;
    if(aux_response_ready[row])aux_full[row]<=0;
    if(rfire)read_full[row]<=1;
    if(afire)begin aux_full[row]<=1;aux_was_write[row]<=|aux_write_half[row*2+:2];end
   end
  end
 end
endmodule
