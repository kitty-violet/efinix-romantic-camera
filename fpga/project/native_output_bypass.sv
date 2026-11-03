// The same DDR reader feeds either the existing 720p scaler or the final
// 1080p FIFO. A native frame waits for any previous scaled frame to finish.
// A source-mode change cannot mix their pixels inside one HDMI frame.
module native_output_bypass(
 input wire clk,reset,
 input wire native_active,
 input wire s_valid,output wire s_ready,input wire[23:0]s_rgb,
 input wire[15:0]s_id,input wire s_sof,s_eol,s_eof,
 output wire scale_valid,input wire scale_ready,
 input wire up_valid,output wire up_ready,input wire[23:0]up_rgb,
 input wire[15:0]up_id,input wire up_sof,up_eol,up_eof,
 output wire m_valid,input wire m_ready,output wire[23:0]m_rgb,
 output wire[15:0]m_id,output wire m_sof,m_eol,m_eof
);
 reg style_inflight;
 // Two pixel registers isolate downstream tag/marker admission from the
 // scaler's ready chain. Capacity alone controls upstream ready. There is
 // no combinational path from m_ready (or output payload) to up_ready.
 reg front_valid,spare_valid;
 reg[42:0]front_pixel,spare_pixel;
 wire accept_ready=!spare_valid;
 wire use_native=native_active && !style_inflight && !up_valid;
 assign scale_valid=s_valid && !native_active;
 assign s_ready=native_active ? (use_native && accept_ready):scale_ready;
 assign up_ready=accept_ready;
 wire select_valid=up_valid || (use_native && s_valid);
 wire[42:0]select_pixel=use_native?{s_eof,s_eol,s_sof,s_id,s_rgb}:{up_eof,up_eol,up_sof,up_id,up_rgb};
 assign m_valid=front_valid;
 assign {m_eof,m_eol,m_sof,m_id,m_rgb}=front_pixel;
 always @(posedge clk)begin
  if(reset)begin front_valid<=0;spare_valid<=0;end
  else if(select_valid && accept_ready)begin
   if(!front_valid || m_ready)begin
    front_pixel<=select_pixel;front_valid<=1;
   end else begin
    spare_pixel<=select_pixel;spare_valid<=1;
   end
  end else if(m_ready)begin
   front_pixel<=spare_pixel;front_valid<=spare_valid;spare_valid<=0;
  end
 end
 always @(posedge clk)begin
  if(reset)style_inflight<=0;
  else begin
   if(up_valid && up_ready && up_eof)style_inflight<=0;
   if(scale_valid && scale_ready && s_sof)style_inflight<=1;
  end
 end
endmodule
