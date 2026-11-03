// Fixed 2:3 RGB upscaler. Store two original rows, not widened intermediate
// rows. Two read ports per row supply all four bilinear taps each clock. Horizontal
// rounding still precedes vertical rounding, exactly matching the old contract.
module rgb_bilinear_scaler_prototype #(
 parameter integer UPSCALE=1,SRC_WIDTH=1280,SRC_HEIGHT=720,
 DST_WIDTH=1920,DST_HEIGHT=1080,FRAME_ID_WIDTH=16
)(
 input wire clk,rst_n,input wire s_valid,output wire s_ready,
 input wire [23:0] s_rgb,input wire [FRAME_ID_WIDTH-1:0] s_frame_id,
 input wire s_sof,s_eol,output wire m_valid,input wire m_ready,
 output wire [23:0] m_rgb,output wire [FRAME_ID_WIDTH-1:0] m_frame_id,
 output wire m_sof,m_eol,m_eof,output reg protocol_error,output wire[11:0]graft_x,graft_y,output wire[2:0]graft_state
);
 localparam LOAD=0,CHECK=1,START=2,OUTPUT_ROW=3,ADVANCE=4;
 reg [2:0] state;
 reg [11:0] source_x,source_y,output_x,output_y;
 reg [FRAME_ID_WIDTH-1:0] frame_id;
 wire [24:0] q00,q01,q10,q11;
 wire xv,xlast,yv,ylast;wire [11:0] x0,x1,y0,y1;wire [2:0] xw,yw;
 wire load=s_valid && s_ready;
 wire ystart=load && source_x==0 && source_y==0;
 reg issuing;
 reg rv,sv,hv,ov;
 wire oready=!ov || m_ready,hready=!hv || oready,sready=!sv || hready,rready=!rv || sready;
 wire issue=state==OUTPUT_ROW && issuing && xv && rready;
 resize_axis_up_2_to_3 #(.SRC_SIZE(SRC_WIDTH),.DST_SIZE(DST_WIDTH),.INDEX_WIDTH(12),.COUNT_WIDTH(12)) xa(
  .clk(clk),.rst_n(rst_n),.start(state==START),.step(issue && !xlast),.valid(xv),.last(xlast),.index0(x0),.index1(x1),.weight1_num(xw));
 resize_axis_up_2_to_3 #(.SRC_SIZE(SRC_HEIGHT),.DST_SIZE(DST_HEIGHT),.INDEX_WIDTH(12),.COUNT_WIDTH(12)) ya(
  .clk(clk),.rst_n(rst_n),.start(ystart),.step(state==ADVANCE),.valid(yv),.last(ylast),.index0(y0),.index1(y1),.weight1_num(yw));
 reg ryodd,rysame;reg [2:0] rxw,ryw;
 reg [2:0] rmark,smark,hmark,omark;
 reg [23:0] s00,s01,s10,s11,h0,h1,orgb;reg [2:0] sxw,syw,hyw;
 wire [23:0] rx00=q00[23:0],rx01=q01[23:0],rx10=q10[23:0],rx11=q11[23:0];
 // 5-bit x2048 true-dual-port mode packs each 1280x24 row into five RAM10.
 // Port A writes only in LOAD; both ports read only in OUTPUT_ROW.
 wire [24:0] pixel25={1'b0,s_rgb};
 for(genvar row=0;row<2;row=row+1)begin:ROW
  for(genvar lane=0;lane<5;lane=lane+1)begin:LANE
   wire [4:0] qa,qb;
   EFX_DPRAM10 #(.READ_WIDTH_A(5),.WRITE_WIDTH_A(5),.READ_WIDTH_B(5),.WRITE_WIDTH_B(5),
    .OUTPUT_REG_A(0),.OUTPUT_REG_B(0),.WRITE_MODE_A("READ_FIRST"),.WRITE_MODE_B("READ_FIRST")) mem(
    .CLKA(clk),.CLKEA(issue || (load && source_y[0]==1'(row))),.WEA(load && source_y[0]==1'(row)),
    .RSTA(1'b0),.ADDRENA(1'b1),.ADDRA(load ? source_x[10:0]:x0[10:0]),.WDATAA(pixel25[lane*5+:5]),.RDATAA(qa),
    .CLKB(clk),.CLKEB(issue),.WEB(1'b0),.RSTB(1'b0),.ADDRENB(1'b1),.ADDRB(x1[10:0]),.WDATAB(5'd0),.RDATAB(qb));
   if(row==0)begin assign q00[lane*5+:5]=qa;assign q01[lane*5+:5]=qb;end
   else begin assign q10[lane*5+:5]=qa;assign q11[lane*5+:5]=qb;end
  end
 end
 // Divide floor(n/2) by three using five radix-four remainder steps.
 // Each step sees only {remainder[1:0], digit[1:0]}, never a wide divider.
 function automatic [7:0] div6(input [10:0] n);
  reg [9:0] half_n,quotient;
  reg [1:0] remainder;
  reg [3:0] digit,qr;
  begin
   half_n=n[10:1];quotient=0;remainder=0;
   for(integer j=4;j>=0;j=j-1)begin
    digit={remainder,half_n[j*2+:2]};
    case(digit)
     0:qr=4'b0000;1:qr=4'b0001;2:qr=4'b0010;
     3:qr=4'b0100;4:qr=4'b0101;5:qr=4'b0110;
     6:qr=4'b1000;7:qr=4'b1001;8:qr=4'b1010;
     9:qr=4'b1100;10:qr=4'b1101;11:qr=4'b1110;
     default:qr=4'b0000;
    endcase
    quotient[j*2+:2]=qr[3:2];remainder=qr[1:0];
   end
   div6=quotient[7:0];
  end
 endfunction
 function automatic [7:0] lerp(input [7:0] a,b,input [2:0] w);
  reg [7:0] base_pixel,other_pixel;
  reg [10:0] total;
  reg [8:0] middle;
  begin
   // Weights one and five are the same datapath with exchanged operands.
   // Zero, one-half, and one bypass the weighted numerator entirely.
   base_pixel=w==5 ? b:a;other_pixel=w==5 ? a:b;
   total=({3'd0,base_pixel}<<2)+{3'd0,base_pixel}+{3'd0,other_pixel}+11'd3;
   middle={1'b0,a}+{1'b0,b}+9'd1;
   case(w)
    0:lerp=a;
    1,5:lerp=div6(total);
    3:lerp=middle[8:1];
    default:lerp=b;
   endcase
  end
 endfunction
 function automatic [23:0] blend(input [23:0] a,b,input [2:0] w);
  begin for(integer c=0;c<3;c=c+1)blend[c*8+:8]=lerp(a[c*8+:8],b[c*8+:8],w);end
 endfunction
 assign graft_x=output_x;assign graft_y=output_y;assign graft_state=state;
 assign s_ready=rst_n && state==LOAD;
 assign m_valid=rst_n && ov;assign m_rgb=orgb;assign m_frame_id=frame_id;
 assign {m_sof,m_eol,m_eof}={3{m_valid}}&omark;
 // Only pipeline valid/ownership resets; payloads are overwritten before use.
 always @(posedge clk)begin
  if(sready && rv)begin
   s00<=ryodd?rx10:rx00;s01<=ryodd?rx11:rx01;
   s10<=rysame?(ryodd?rx10:rx00):(ryodd?rx00:rx10);
   s11<=rysame?(ryodd?rx11:rx01):(ryodd?rx01:rx11);
   sxw<=rxw;syw<=ryw;smark<=rmark;
  end
  if(hready && sv)begin h0<=blend(s00,s01,sxw);h1<=blend(s10,s11,sxw);hyw<=syw;hmark<=smark;end
  if(oready && hv)begin orgb<=blend(h0,h1,hyw);omark<=hmark;end
 end
 always @(posedge clk or negedge rst_n)begin
  if(!rst_n)begin
   state<=LOAD;source_x<=0;source_y<=0;output_x<=0;output_y<=0;
   frame_id<=0;protocol_error<=0;rv<=0;sv<=0;hv<=0;ov<=0;issuing<=0;
   ryodd<=0;rysame<=0;rxw<=0;ryw<=0;rmark<=0;
  end else begin
   if(oready)ov<=hv;
   if(hready)hv<=sv;
   // Each stage may drain independently when its downstream slot is free.
   if(sready)sv<=rv;
   if(rready)rv<=issue;
   if(issue)begin
    ryodd<=y0[0];rysame<=y0==y1;rxw<=xw;ryw<=yw;
    rmark<={output_y==0 && output_x==0,xlast,xlast && ylast};
    if(xlast)issuing<=0;else output_x<=output_x+1'b1;
   end
   case(state)
    LOAD:if(load)begin
     if(s_sof)frame_id<=s_frame_id;else if(s_frame_id!=frame_id)protocol_error<=1;
     if(s_sof!=(source_x==0 && source_y==0) || s_eol!=(source_x==SRC_WIDTH-1))protocol_error<=1;
     if(source_x==SRC_WIDTH-1)begin source_x<=0;state<=CHECK;end
     else source_x<=source_x+1'b1;
    end
    CHECK:if(yv)begin
     if(y1<=source_y)begin state<=START;output_x<=0;end
     else if(source_y<SRC_HEIGHT-1)begin source_y<=source_y+1'b1;state<=LOAD;end
     else protocol_error<=1;
    end
    START:begin state<=OUTPUT_ROW;issuing<=1;end
    OUTPUT_ROW:if(m_valid && m_ready && m_eol)begin
     if(m_eof)begin source_x<=0;source_y<=0;output_y<=0;state<=LOAD;end
     else begin output_y<=output_y+1'b1;state<=ADVANCE;end
    end
    ADVANCE:state<=CHECK;
    default:begin state<=LOAD;protocol_error<=1;end
   endcase
  end
 end
 initial if(UPSCALE!=1 || SRC_WIDTH<2 || SRC_WIDTH>2048 || SRC_HEIGHT<2 || DST_WIDTH*2!=SRC_WIDTH*3 || DST_HEIGHT*2!=SRC_HEIGHT*3)
  $fatal(1,"RAW_ROW_UPSCALE_REQUIRES_FIXED_2_TO_3");
endmodule
