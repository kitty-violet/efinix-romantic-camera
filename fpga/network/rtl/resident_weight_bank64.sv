// Four official 512x16 synchronous RAMs form each resident 512x64 bank.
module cnn_weight_bank64 #(parameter DEPTH=512,ADDR_WIDTH=9)(
 input wire clk,write_enable,input wire [ADDR_WIDTH-1:0] write_address,input wire [63:0] write_data,
 input wire read_enable,input wire [ADDR_WIDTH-1:0] read_address,output wire [63:0] read_data
);
 for(genvar n=0;n<4;n=n+1)begin:RAM
  EFX_RAM10 #(.READ_WIDTH(16),.WRITE_WIDTH(16),.OUTPUT_REG(1'b0),.WRITE_MODE("READ_FIRST")) mem(
   .WCLK(clk),.WE(2'b11),.WCLKE(write_enable),.WADDREN(1'b1),.WDATA(write_data[n*16+:16]),.WADDR(write_address),
   .RCLK(clk),.RE(read_enable),.RST(1'b0),.RADDREN(1'b1),.RDATA(read_data[n*16+:16]),.RADDR(read_address));
 end
endmodule

