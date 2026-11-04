// Hardware issuer for the versioned 0x55/0x3 row-pair two-stream descriptor ABI.
// The finite transport owns records until completion (cready/dready).
// All cross-engine dependencies count completed operations, not issued requests.
// RGB publication is gated by CRC, full coverage, all compute nodes and final B.
module frame_program_issue #(parameter integer AUTO_DW_GROUPS=0)(
 input wire clk,reset,cancel,start,
 input wire discard,discard_release,output reg discarded,output wire discard_boundary,
 input wire [13:0] compute_records,dma_records,
 input wire cvalid,output wire cready,input wire [511:0] cdata,
 input wire next_valid,input wire [511:0] next_data,
 input wire dvalid,output wire dready,input wire [127:0] ddata,
 input wire fetch_verified,fetch_error,
 output wire cfg_valid,input wire cfg_ready,done_valid,output wire done_ready,
 input wire datapath_busy,datapath_error,
 output wire cfg_pw_fuse,cfg_main_scaled,cfg_stride2,cfg_stride4,cfg_nn4,cfg_stem,cfg_mode_pw,cfg_nn_view,
 output wire [11:0] cfg_skip_banks,output wire [39:0] cfg_skip_bases,output wire [10:0] cfg_skip_delta,
 output wire [17:0] align_multiplier,post_multiplier,
 output wire [8:0] cfg_pixels,output wire [2:0] cfg_rows,cfg_in_groups,cfg_groups,
 output wire [1:0] cfg_group,output wire [7:0] cfg_mask,
 output wire [10:0] cfg_output_x,cfg_source_width,cfg_source_origin,
 output wire [17:0] cfg_row_banks,output wire [59:0] cfg_row_bases,
 output wire [11:0] out_banks,output wire [39:0] out_bases,output wire [14:0] cfg_weight_offset,
 output wire pcv,input wire pcr,
 output wire hv,input wire hready,output reg [14:0] hbias,hmult,hrelu,
 output reg [7:0] hlayer,output reg [2:0] hgroups,
 input wire hsvalid,hssuccess,output wire hsready,input wire hbusy,herror,
 output wire dma_valid,input wire dma_ready,output wire dstore,drgb,output wire [31:0] daddress,
 output wire [8:0] dcolumns,output wire [2:0] dgroups,output wire [2:0] dbank,output wire [9:0] dbase,
 input wire ddone,output wire ddready,input wire derror,dbusy,
 output reg busy,frame_valid,error,output reg [3:0] fault_code,
 output reg [13:0] compute_completed,dma_completed,
 output reg [28:0] nodes_completed,output reg [19:0] rgb_pixels,
 output reg [31:0] frame_cycles,
 output wire compute_begin,compute_end,dma_end,
 output wire cfg_storage_phase,output wire [3:0] cfg_out_phase,
 output wire [14:0] debug_program_index
);
 wire [4:0] c1=cdata[29+:5];
 wire [8:0] c2=cdata[34+:9];
 wire [2:0] c3=cdata[43+:3];
 wire [2:0] c4=cdata[46+:3];
 wire [2:0] c5=cdata[49+:3];
 wire [5:0] c6=cdata[52+:6];
 wire [10:0] c7=cdata[58+:11];
 wire [10:0] c8=cdata[69+:11];
 wire [10:0] c9=cdata[80+:11];
 wire [17:0] c10=cdata[91+:18];
 wire [31:0] c11=cdata[109+:32];
 wire [27:0] c12=cdata[141+:28];
 wire [11:0] c13=cdata[169+:12];
 wire [31:0] c14=cdata[181+:32];
 wire [7:0] c15=cdata[213+:8];
 wire [14:0] c16=cdata[221+:15];
 wire [14:0] c17=cdata[236+:15];
 wire [14:0] c18=cdata[251+:15];
 wire [14:0] c19=cdata[266+:15];
 wire [11:0] c20=cdata[281+:12];
 wire [31:0] c21=cdata[293+:32];
 wire [7:0] c22=cdata[325+:8];
 wire [10:0] c23=cdata[333+:11];
 wire [17:0] c24=cdata[344+:18];
 wire [17:0] c25=cdata[362+:18];
 wire [7:0] c28=cdata[380+:8];
 wire [28:0] c29=cdata[388+:29];
 wire [9:0] c30=cdata[417+:10];
 wire [14:0] c31=cdata[427+:15];
 wire [0:0] d0=ddata[14+:1];
 wire [4:0] d1=ddata[15+:5];
 wire [8:0] d2=ddata[20+:9];
 wire [2:0] d3=ddata[29+:3];
 wire [2:0] d4=ddata[32+:3];
 wire [9:0] d5=ddata[35+:10];
 wire [0:0] d6=ddata[45+:1];
 wire [27:0] d7=ddata[46+:28];
 wire [0:0] d8=ddata[74+:1];
 wire [9:0] d9=ddata[75+:10];
 wire [10:0] d10=ddata[85+:11];
 wire [9:0] d30=ddata[96+:10];
 wire [14:0] d31=ddata[106+:15];
 localparam [2:0] C_IDLE=0,C_HEADER=1,C_APPLY=2,C_START=3,C_RUN=4,C_FINISH=5;
 localparam [1:0] H_EMPTY=0,H_REQUEST=1,H_WAIT=2,H_VALID=3;
 reg [2:0] cstate;reg [1:0] hstate;
 reg dwait;reg [1:0] group_index;
 reg discard_pending,dma_offer;
 wire stop_new=discard_pending || discard;
 reg [13:0] ctotal,dtotal;
 reg [9:0] rgb_y;reg [10:0] rgb_x;
 // Finite one-frame run. Timeout stops new commands; downstream AXI drains
 // through the transport and datapath. A wedged bus requires common reset.
 wire watchdog_expired=frame_cycles>=32'd33554432;
 wire stopped=cancel || error || fetch_error || datapath_error || derror || herror || watchdog_expired;
 wire active=busy && !stopped && !reset;
 assign discard_boundary=active && stop_new && cstate==C_IDLE && !dwait && !dma_offer &&
  hstate!=H_REQUEST && hstate!=H_WAIT && !datapath_busy && !dbusy && !hbusy;
 wire cbad=cdata[511:504]!=8'h55 || cdata[503:449]!=0 || c1>=13 || cdata[13:0]>dtotal || cdata[28:14]>=14823;
 wire dbad=ddata[127:124]!=4'h3 || ddata[123:121]!=0 || d1>=13 || ddata[13:0]>ctotal;
 wire [31:0] rgb_offset=((({22'd0,rgb_y}<<10)+({22'd0,rgb_y}<<8))+{21'd0,rgb_x})*3;
 wire rgb_bad=d8 && (!d0 || !d6 || d2!=256 || d9!=rgb_y || d10!=rgb_x ||
  {4'd0,d7}!=32'h00800000+rgb_offset);
 wire final_compute=done_valid && (AUTO_DW_GROUPS!=0 || c6[0] || {1'b0,group_index}+3'd1==c5);
 assign cready=active && cstate==C_RUN && final_compute;
 assign dready=active && dwait && ddone;
 assign cfg_valid=active && cstate==C_START;
 assign done_ready=active && cstate==C_RUN;
 assign pcv=active && cstate==C_APPLY;
 assign hv=active && hstate==H_REQUEST;
 assign hsready=active && hstate==H_WAIT;
 assign dma_valid=active && !dwait && (!stop_new || dma_offer) && dvalid && ddata[13:0]<=compute_completed && !dbad && !rgb_bad;
 assign ddready=active && dwait;
 assign cfg_stride4=cdata[447];assign cfg_nn4=cdata[448];
 assign cfg_storage_phase=cdata[442];assign cfg_out_phase=cdata[443+:4];
 assign cfg_pw_fuse=c6[4];assign cfg_main_scaled=c6[5];assign cfg_stride2=c6[3];
 assign cfg_stem=c6[2];assign cfg_mode_pw=c6[0];assign cfg_nn_view=c6[1];
 assign cfg_pixels=c2;assign cfg_rows=c3;assign cfg_in_groups=c4;assign cfg_groups=c5;
 assign cfg_group=group_index;assign cfg_mask=c28;
 assign cfg_output_x=c7;assign cfg_source_width=c8;assign cfg_source_origin=c9;
 assign cfg_row_banks=c10;assign cfg_row_bases={c12,c11};
 assign out_banks=c13;assign out_bases={c15,c14};
 assign cfg_weight_offset=c16+(c6[0] ? 15'd0:({13'd0,group_index}<<6)+({13'd0,group_index}<<3));
 assign cfg_skip_banks=c20;assign cfg_skip_bases={c22,c21};assign cfg_skip_delta=c23;
 assign align_multiplier=c24;assign post_multiplier=c25;
 assign dstore=d0;assign drgb=d6;assign daddress={4'd0,d7};assign dcolumns=d2;
 assign dgroups=d3;assign dbank=d4;assign dbase=d5;
 assign compute_begin=active && !stop_new && cstate==C_IDLE && cvalid && cdata[13:0]<=dma_completed && !cbad;
 assign compute_end=cready;assign dma_end=dready;
 assign debug_program_index=cdata[28:14];
 wire prefetch=active && !stop_new && (cstate==C_START || cstate==C_RUN || cstate==C_FINISH) &&
  next_valid && hstate==H_VALID && next_data[236+:15]!=hbias;
 always @(posedge clk)begin
  if(reset || start)begin
   cstate<=C_IDLE;hstate<=H_EMPTY;dwait<=0;group_index<=0;
   discard_pending<=0;discarded<=0;dma_offer<=0;
   hbias<=0;hmult<=0;hrelu<=0;hlayer<=0;hgroups<=0;
   busy<=!reset && start;frame_valid<=0;error<=0;fault_code<=0;
   compute_completed<=0;dma_completed<=0;ctotal<=compute_records;dtotal<=dma_records;
   nodes_completed<=0;rgb_pixels<=0;frame_cycles<=0;rgb_y<=0;rgb_x<=0;
  end else if(busy)begin
   if(discard)discard_pending<=1;
   // Preserve an already offered command until acceptance, including stalls.
   if(dma_valid)dma_offer<=!dma_ready;
   frame_cycles<=frame_cycles+1'b1;
   if(stopped)begin
    busy<=0;frame_valid<=0;error<=1;
    fault_code<=error ? fault_code:cancel ? 4'd1:fetch_error ? 4'd2:watchdog_expired ? 4'd8:4'd3;
   end else begin
    if(cvalid && cbad || dvalid && (dbad || rgb_bad))begin error<=1;fault_code<=4'd4;end
    case(hstate)
     H_REQUEST:if(hready)hstate<=H_WAIT;
     H_WAIT:if(hsvalid)begin
      if(hssuccess)hstate<=H_VALID;else begin error<=1;fault_code<=4'd5;end
     end
     default:;
    endcase
    case(cstate)
     C_IDLE:if(compute_begin)begin cstate<=(hstate==H_VALID && hbias==c17) ? C_APPLY:C_HEADER;group_index<=0;end
     C_HEADER:begin
      if(hstate==H_VALID && hbias==c17)cstate<=C_APPLY;
      else if(hstate==H_EMPTY || hstate==H_VALID)begin
       hbias<=c17;hmult<=c18;hrelu<=c19;hlayer<={3'd0,c1};hgroups<=c5;hstate<=H_REQUEST;
      end
     end
     C_APPLY:if(pcr)cstate<=C_START;
     C_START:if(cfg_ready)cstate<=C_RUN;
     C_RUN:if(done_valid)begin
      if(final_compute)begin
       cstate<=C_IDLE;compute_completed<=compute_completed+1'b1;nodes_completed<=nodes_completed|c29;
      end
      else begin group_index<=group_index+1'b1;cstate<=C_START;end
     end
     default:begin error<=1;fault_code<=4'd6;end
    endcase
    if(prefetch)begin
     hbias<=next_data[236+:15];hmult<=next_data[251+:15];hrelu<=next_data[266+:15];
     hlayer<={3'd0,next_data[29+:5]};hgroups<=next_data[49+:3];hstate<=H_REQUEST;
    end
    if(dma_valid && dma_ready)dwait<=1;
    if(dready)begin
     dwait<=0;dma_completed<=dma_completed+1'b1;
     if(d8)begin
      rgb_pixels<=rgb_pixels+{11'd0,d2};
      if(rgb_y==719)begin rgb_y<=0;rgb_x<=rgb_x+11'd256;end else rgb_y<=rgb_y+1'b1;
     end
    end
    // Finish current compute/header/DMA commands. Never abort the AXI owner.
    // An intentional discard is not frame completion and is not a fault.
    // Keep busy/watchdog alive until the program transport also drained.
    if(discard_boundary && discard_release)begin
     busy<=0;frame_valid<=0;discarded<=1;
    end else if(!stop_new && compute_completed==ctotal && dma_completed==dtotal && fetch_verified &&
       !datapath_busy && !dbusy && !hbusy)begin
     busy<=0;
     if(nodes_completed==29'h0001fff && rgb_pixels==921600 && rgb_x==1280 && rgb_y==0)frame_valid<=1;
     else begin error<=1;fault_code<=4'd7;end
    end
   end
  end
 end
endmodule
