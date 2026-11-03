// Fixed 128-bit RGB stream: at most two retained bytes when a new beat arrives.
module rgb24_stream_unpack(
 input wire clk,reset,
 input wire s_valid,output wire s_ready,input wire [127:0] s_data,
 output wire m_valid,input wire m_ready,output wire [23:0] m_rgb,
 output wire empty
);
 reg [143:0] buffer;reg [4:0] count;
 wire pop=m_valid && m_ready;
 assign m_valid=!reset && count>=3;
 assign m_rgb=buffer[23:0];assign empty=count==0;
 wire [4:0] remaining=count-(pop ? 5'd3:5'd0);
 assign s_ready=!reset && ((count<3) || (count<6 && m_ready));
 wire push=s_valid && s_ready;
 wire [15:0] retained=pop ? buffer[39:24]:buffer[15:0];
 // Concatenate disjoint byte fields. There is no full-width old/new OR.
 // The first accepted beat overwrites the complete payload before VALID.
 always @(posedge clk)begin
  if(push)begin
   case(remaining)
    0:buffer<={16'd0,s_data};
    1:buffer<={8'd0,s_data,retained[7:0]};
    default:buffer<={s_data,retained};
   endcase
  end else if(pop)buffer<={24'd0,buffer[143:24]};
  if(reset)count<=0;
  else count<=remaining+(push ? 5'd16:5'd0);
 end
 // synthesis translate_off
 always @(posedge clk)if(push && remaining>2)$fatal(1,"RGB_APPEND_CAPACITY");
 // synthesis translate_on
endmodule
