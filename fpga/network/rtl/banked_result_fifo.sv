//16 finite entries, one real synchronous read and write port,104-bit payload.
// Six official RAM10 blocks. Credit for in-flight RQ is managed by the caller.
module banked_result_fifo #(parameter integer WIDTH=104)(
 input wire clk,reset,
 input wire push,input wire [WIDTH-1:0] in_data,
 output wire out_valid,input wire out_ready,output wire [WIDTH-1:0] out_data,
 output wire empty,output reg error
);
 reg [3:0] write_ptr,read_ptr;reg [4:0] stored;
 reg valid_q;
 wire read_fire=stored!=0 && (!valid_q || out_ready) && !reset;
 localparam integer BLOCKS=(WIDTH+19)/20;
 wire [BLOCKS*20-1:0] wide_input={{(BLOCKS*20-WIDTH){1'b0}},in_data};
 wire [BLOCKS*20-1:0] wide_output;
 assign out_data=wide_output[WIDTH-1:0];assign out_valid=valid_q && !reset && !error;
 assign empty=stored==0 && !valid_q;
 for(genvar b=0;b<BLOCKS;b=b+1)begin:RAM
  EFX_RAM10 #(.READ_WIDTH(20),.WRITE_WIDTH(20),.OUTPUT_REG(1'b0),.WRITE_MODE("READ_FIRST")) mem(
   .WCLK(clk),.WE(2'b11),.WCLKE(push && !reset && !error),.WADDREN(1'b1),.WDATA(wide_input[b*20+:20]),.WADDR({5'd0,write_ptr}),
   .RCLK(clk),.RE(read_fire),.RST(1'b0),.RADDREN(1'b1),.RDATA(wide_output[b*20+:20]),.RADDR({5'd0,read_ptr}));
 end
 always @(posedge clk)begin
  if(reset)begin write_ptr<=0;read_ptr<=0;stored<=0;valid_q<=0;error<=0;end
  else begin
   if(out_ready)valid_q<=0;
   if(read_fire)begin read_ptr<=read_ptr+1'b1;valid_q<=1;end
   if(push)begin
    if(stored==16 || (read_fire && write_ptr==read_ptr))error<=1;
    write_ptr<=write_ptr+1'b1;
   end
   case({push,read_fire})2'b10:stored<=stored+1'b1;2'b01:stored<=stored-1'b1;default:stored<=stored;endcase
  end
 end
endmodule
