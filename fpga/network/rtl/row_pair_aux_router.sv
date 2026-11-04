// Three fixed clients (writer, skip, DMA) share each of six auxiliary RAM ports.
// Responses remain in BRAM and route by a two-bit owner; no payload FIFO.
// Main compute reads use the separate A ports and never enter this crossbar.
module row_pair_aux_router #(parameter integer READ_ONLY_SKIP=0)(
 input wire clk,reset,cancel,
 input wire [5:0] protected_read_fire,input wire [59:0] protected_read_address,
 output wire [5:0] aux_write_collision,
 input wire [17:0] request_valid,output wire [17:0] request_ready,
 input wire [179:0] request_address,input wire [35:0] request_half,input wire [2303:0] request_data,
 output wire [17:0] response_valid,input wire [17:0] response_ready,
 output wire [2303:0] response_data,output wire [17:0] response_was_write,
 output wire [5:0] aux_valid,input wire [5:0] aux_ready,
 output wire [59:0] aux_address,output wire [11:0] aux_write_half,output wire [767:0] aux_data,
 input wire [5:0] aux_response,output wire [5:0] aux_response_ready,
 input wire [767:0] aux_response_data,input wire [5:0] aux_was_write,
 output wire busy,output wire error
);
 reg [5:0] pending,held,bad,pending_write;
 reg [1:0] owner[0:5],held_owner[0:5];
 wire active=!reset && !cancel && !(|bad);
 assign busy=|pending || |request_valid;assign error=|bad;
 for(genvar b=0;b<6;b=b+1)begin:BANK
  wire [2:0] valid={request_valid[12+b],request_valid[6+b],request_valid[b]};
  reg [2:0] choice;
  always @*begin
   choice=0;
   case(held_owner[b])
    0:if(valid[1])choice=3'b010;else if(valid[2])choice=3'b100;else if(valid[0])choice=3'b001;
    1:if(valid[2])choice=3'b100;else if(valid[0])choice=3'b001;else if(valid[1])choice=3'b010;
    default:if(valid[0])choice=3'b001;else if(valid[1])choice=3'b010;else if(valid[2])choice=3'b100;
   endcase
  end
  // Registered selection stays on an idle client or selects the next pending
  // client at the accepted edge. No combinational grant enters the payload.
  // An unaccepted valid request retains this owner and its producer payload.
  wire [2:0] select={held_owner[b]==2,held_owner[b]==1,held_owner[b]==0};
  // Compare client addresses in parallel with arbitration; only a one-bit
  // collision result then crosses the owner mux. Keep the actual A handshake.
  (* syn_keep = "true" *) wire [2:0] client_collision;
  for(genvar c=0;c<3;c=c+1)begin:COLLISION
   assign client_collision[c]=(|request_half[(c*6+b)*2+:2]) &&
    protected_read_address[b*10+:10]==request_address[(c*6+b)*10+:10];
  end
  assign aux_write_collision[b]=protected_read_fire[b] && (|(select & client_collision));
  wire have=|(select & valid);
  wire [1:0] grant=held_owner[b];
  wire [9:0] address=({10{select[0]}} & request_address[b*10+:10]) |
    ({10{select[1]}} & request_address[60+b*10+:10]) |
    ({10{select[2]}} & request_address[120+b*10+:10]);
  wire [1:0] half=({2{select[0]}} & request_half[b*2+:2]) |
    ({2{select[1]}} & request_half[12+b*2+:2]) |
    ({2{select[2]}} & request_half[24+b*2+:2]);
  wire [127:0] data;
  if(READ_ONLY_SKIP!=0)begin:VIDEO_DATA
   // Only clients 0 and 2 write. Data is immaterial on a read or idle port.
   // Keep owner/arbitration/ACK unchanged; remove a full-width zeroing stage.
   assign data=select[2] ? request_data[1536+b*128+:128]:request_data[b*128+:128];
  end else begin:GENERIC_DATA
   assign data=({128{select[0]}} & request_data[b*128+:128]) |
    ({128{select[1]}} & request_data[768+b*128+:128]) |
    ({128{select[2]}} & request_data[1536+b*128+:128]);
  end
  wire ready=(owner[b]==0 && response_ready[b]) ||
    (owner[b]==1 && response_ready[6+b]) ||
    (owner[b]==2 && response_ready[12+b]);
  assign aux_valid[b]=active && have;
  assign aux_address[b*10+:10]=address;assign aux_write_half[b*2+:2]=half;assign aux_data[b*128+:128]=data;
  assign aux_response_ready[b]=active && pending[b] && ready;
  wire fire=aux_valid[b] && aux_ready[b];
  for(genvar m=0;m<3;m=m+1)begin:CLIENT
   assign request_ready[m*6+b]=active && have && grant==2'(m) && aux_ready[b];
   assign response_valid[m*6+b]=active && pending[b] && owner[b]==2'(m) && aux_response[b];
   // Ready/valid qualifies data; broadcasting costs no three-way output mux.
   assign response_data[m*768+b*128+:128]=aux_response_data[b*128+:128];
   assign response_was_write[m*6+b]=aux_was_write[b];
  end
  always @(posedge clk)begin
   if(reset || cancel)begin pending[b]<=0;held[b]<=0;bad[b]<=0;pending_write[b]<=0;owner[b]<=0;held_owner[b]<=0;end
   else if(active)begin
    if(aux_response[b] && (!pending[b] || aux_was_write[b]!=pending_write[b]))bad[b]<=1;
    if(held[b] && !have)bad[b]<=1;
    if(READ_ONLY_SKIP!=0 && valid[1] && (|request_half[12+b*2+:2]))bad[b]<=1;
    if(aux_response[b] && aux_response_ready[b])pending[b]<=0;
    // held now records a real stalled request, not an idle parked owner.
    held[b]<=have && !aux_ready[b];
    if((!have || fire) && (|choice))
     held_owner[b]<=choice[2] ? 2'd2:choice[1] ? 2'd1:2'd0;
    if(fire)begin
     if(pending[b] && !(aux_response[b] && aux_response_ready[b]))bad[b]<=1;
     pending[b]<=1;pending_write[b]<=|half;owner[b]<=grant;held[b]<=0;
    end
   end
  end
 end
endmodule
