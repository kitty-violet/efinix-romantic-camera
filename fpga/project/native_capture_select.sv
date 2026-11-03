// Switch at an accepted input SOF, never in the middle of a captured frame.
// Drain the preceding downscaled frame before handing the shared writer a
// native frame. No extra frame buffer or independently refreshed video path.
module native_capture_select(
 input wire clk,reset,abort_frame,want_native,
 input wire s_valid,output wire s_ready,input wire[23:0]s_rgb,
 input wire[15:0]s_id,input wire s_sof,s_eol,
 output wire down_input_valid,input wire down_input_ready,
 input wire down_valid,output wire down_ready,input wire[23:0]down_rgb,
 input wire[15:0]down_id,input wire down_sof,down_eol,down_eof,
 output wire m_valid,input wire m_ready,output wire[23:0]m_rgb,
 output wire[15:0]m_id,output wire m_sof,m_eol,m_eof,m_native
);
 reg native_route,style_inflight;
 reg[10:0]source_x,source_y;
 // On a stalled SOF the decision must also stay put. Latch the offer before
 // acceptance, just as the existing frame admission locks its grant.
 reg offered,offered_native;
 wire requested=s_sof ? (offered?offered_native:want_native):native_route;
 wire native_select=requested && !style_inflight;
 wire enabled=!reset && !abort_frame;
 wire input_boundary_ready=!s_sof || !style_inflight;
 assign down_input_valid=enabled && s_valid && !requested && input_boundary_ready;
 assign s_ready=enabled && input_boundary_ready && (requested ? (!style_inflight && m_ready):down_input_ready);
 assign down_ready=enabled && !native_select && m_ready;
 assign m_valid=enabled && (native_select?s_valid:down_valid);
 assign m_rgb=native_select?s_rgb:down_rgb;
 assign m_id=native_select?s_id:down_id;
 assign m_sof=native_select?s_sof:down_sof;
 // The released ISP carries SOF/ID but has no exported EOL. Reconstruct
 // native row markers from accepted pixels, never from wall-clock cycles.
 wire native_eol=source_x==11'd1919;
 assign m_eol=native_select?native_eol:down_eol;
 assign m_eof=native_select?(native_eol && source_y==11'd1079):down_eof;
 assign m_native=native_select;
 always @(posedge clk)begin
  if(reset || abort_frame)begin
   native_route<=0;style_inflight<=0;source_x<=0;source_y<=0;offered<=0;offered_native<=0;
  end else begin
   if(s_valid && s_sof && !offered)begin offered<=1;offered_native<=want_native;end
   if(down_valid && down_ready && down_eof)style_inflight<=0;
   if(s_valid && s_ready)begin
    if(s_sof)begin
     native_route<=requested;source_y<=0;offered<=0;
     if(!requested)style_inflight<=1;
    end
    if(native_eol)begin
     source_x<=0;source_y<=source_y==11'd1079 ? 11'd0:source_y+1'b1;
    end else source_x<=source_x+1'b1;
   end
  end
 end
endmodule
