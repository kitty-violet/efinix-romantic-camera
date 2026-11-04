// Eight stored entries plus the existing output register. The integrated
// callers reserve at most eight TOTAL in-flight words, including this register
// and upstream RQ stages. Keep the original registered-output timing.
module shallow_result_fifo #(parameter integer WIDTH=80)(
 input wire clk,reset,input wire push,input wire [WIDTH-1:0] in_data,
 output wire out_valid,input wire out_ready,output reg [WIDTH-1:0] out_data,
 output wire empty,output reg error
);
 reg [3:0] stored;reg valid_q;
 wire read_fire=stored!=0 && (!valid_q || out_ready) && !reset;
 wire [2:0] front=stored[2:0]-3'd1;
 wire [WIDTH-1:0] front_data;
 assign out_valid=valid_q && !reset && !error;
 assign empty=stored==0 && !valid_q;
 for(genvar b=0;b<WIDTH;b=b+1)begin:BIT
  wire word_q;
  EFX_SRL8 storage(.D(in_data[b]),.A(~front),.CLK(clk),
   .CE(push && !reset && !error),.Q(word_q),.Q7());
  assign front_data[b]=~word_q;
 end
 always @(posedge clk)begin
  if(reset)begin stored<=0;valid_q<=0;error<=0;end
  else begin
   if(out_ready)valid_q<=0;
   if(read_fire)begin out_data<=front_data;valid_q<=1;end
   if(push && stored==8)error<=1;
   case({push,read_fire})
    2'b10:stored<=stored+4'd1;
    2'b01:stored<=stored-4'd1;
    default:stored<=stored;
   endcase
  end
 end
endmodule
