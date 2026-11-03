// Fixed RGB byte order: R in bits7:0, then G, B. No per-pixel metadata RAM.
// Frame length must be a multiple of16 RGB pixels; 1280x720 meets that contract.
module rgb24_stream_pack(
 input wire clk,reset,
 input wire s_valid,output wire s_ready,input wire [23:0] s_rgb,
 output wire m_valid,input wire m_ready,output wire [127:0] m_data,
 output wire empty
);
 reg [143:0] buffer;reg [4:0] count;
 wire pop=m_valid && m_ready;
 assign m_valid=!reset && count>=16;
 assign m_data=buffer[127:0];
 assign s_ready=!reset && (count<16 || m_ready);
 assign empty=count==0;
 wire push=s_valid && s_ready;
 wire [4:0] remaining=count-(pop ? 5'd16:5'd0);
 wire [143:0] kept=pop ? {128'd0,buffer[143:128]}:buffer;
 wire [143:0] appended={120'd0,s_rgb} << {remaining,3'd0};
 always @(posedge clk)begin
  if(reset)begin buffer<=0;count<=0;end
  else begin
   if(pop || push)buffer<=kept | (push ? appended:144'd0);
   count<=remaining+(push ? 5'd3:5'd0);
  end
 end
endmodule

