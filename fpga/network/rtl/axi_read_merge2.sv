// Two logical readers share ONE CNN master. Hold an offered owner through
// AR backpressure, then lock it until accepted RLAST. Common reset only.
module axi_read_merge2(
 input wire clk,reset,
 input wire [63:0] s_araddr,input wire [15:0] s_arlen,input wire [1:0] s_arvalid,
 output wire [1:0] s_arready,output wire [255:0] s_rdata,output wire [3:0] s_rresp,
 output wire [1:0] s_rlast,s_rvalid,input wire [1:0] s_rready,
 output wire [31:0] m_araddr,output wire [7:0] m_arlen,output wire m_arvalid,
 input wire m_arready,input wire [127:0] m_rdata,input wire [1:0] m_rresp,
 input wire m_rlast,m_rvalid,output wire m_rready,output wire busy
);
 reg held,held_owner,occupied,owner,round_robin;
 wire chosen=held ? held_owner:s_arvalid[1] && (!s_arvalid[0] || round_robin);
 assign m_araddr=chosen ? s_araddr[63:32]:s_araddr[31:0];
 assign m_arlen=chosen ? s_arlen[15:8]:s_arlen[7:0];
 assign m_arvalid=!reset && !occupied && s_arvalid[chosen];
 assign s_arready={2{!reset && !occupied && m_arready}} & (chosen ? 2'b10:2'b01);
 assign s_rvalid={2{!reset && occupied && m_rvalid}} & (owner ? 2'b10:2'b01);
 assign s_rdata={2{m_rdata}};assign s_rresp={2{m_rresp}};assign s_rlast={2{m_rlast}};
 assign m_rready=!reset && occupied && s_rready[owner];
 assign busy=held || occupied;
 always @(posedge clk)begin
  if(reset)begin held<=0;held_owner<=0;occupied<=0;owner<=0;round_robin<=0;end
  else begin
   if(m_arvalid && !m_arready)begin held<=1;held_owner<=chosen;end
   if(m_arvalid && m_arready)begin held<=0;occupied<=1;owner<=chosen;end
   if(m_rvalid && m_rready && m_rlast)begin occupied<=0;round_robin<=!owner;end
  end
 end
endmodule

