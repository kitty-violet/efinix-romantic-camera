// Inactive lanes are suppressed by the registered return_mask at the sole
// externally visible result/error boundary. Do not mask wide operands twice.
// Shared modulo-21 prefix subtraction: keep exact fixed Q17 return latency.
// A/B constant-index routing candidate; same module and fixed latency.
// Eight parallel existing Q17 pipelines. A granted final MAC term reserves this
// cluster's input exactly three edges later. Every grant returns once, even
// when a local owner cancellation flushed its MAC before the fixed tail.
module shared_rq_oc8_cluster #(
    // Production pool rejects any bias-overflow packet before queue commit.
    // Its invalid payload need not be saturated in all 32 lanes.
    parameter integer REJECT_BIAS_OVERFLOW=0,
    parameter integer BANK_PREFIX_HISTORY=0,
    parameter integer PRESELECTED_HEADER=0,
    parameter integer DIRECT_DOT_OWNER=0,
    parameter integer NUM_FRONTS=9,
    parameter integer OWNER_WIDTH=(NUM_FRONTS>1 ? $clog2(NUM_FRONTS) : 1)
) (
    input wire clk,reset,epoch_clear,
    input wire bypass_mode,input wire [71:0] bypass_value,input wire [17:0] bypass_multiplier,
    input wire issue_valid,input wire [OWNER_WIDTH-1:0] issue_owner,
    input wire [1:0] issue_slot,issue_group,
    input wire [7:0] issue_generation,issue_layer_id,issue_expected_mask,
    input wire [15:0] issue_image_tag,
    input wire [NUM_FRONTS-1:0] owner_cancel_pending,dot_valid,
    input wire [NUM_FRONTS*256-1:0] dot_data,
    input wire [NUM_FRONTS*168-1:0] dot_previous,
    input wire [NUM_FRONTS*2-1:0] dot_slot,dot_group,
    input wire [NUM_FRONTS*8-1:0] dot_generation,dot_layer_id,dot_expected_mask,
    input wire [NUM_FRONTS*16-1:0] dot_image_tag,
    input wire [1023:0] header_bias,header_multiplier,
    input wire [31:0] header_relu,input wire [2:0] header_groups,
    input wire [7:0] header_layer_id,input wire [15:0] header_image_tag,
    output wire return_valid,return_killed,
    output wire [OWNER_WIDTH-1:0] return_owner,
    output wire [1:0] return_slot,output wire [7:0] return_generation,
    output wire [63:0] return_data,
    output wire [7:0] return_mask,return_bias_overflow,return_range_error,
    output wire busy,output reg protocol_error
);
    localparam integer TOKEN_WIDTH=OWNER_WIDTH+44;
    localparam integer RETURN_WIDTH=OWNER_WIDTH+18;
    localparam signed [32:0] I32_MAX=33'sd2147483647;
    localparam signed [32:0] I32_MIN=-33'sd2147483648;
    // Match the MAC's three-edge tail without holding an upstream RAM response.
    // Mailbox data is already complete and stable at issue. Requantize now;
    // no three-cycle prediction of an unfinished MAC is needed.
    wire [215:0] bypass_pipe={3{bypass_value}};
    wire [2:0] tail_valid={3{issue_valid}};
    wire [3*TOKEN_WIDTH-1:0] tail_token={3{issue_owner,issue_slot,issue_generation,issue_group,issue_layer_id,issue_image_tag,issue_expected_mask}};
    wire [OWNER_WIDTH-1:0] tail_owner;
    wire [1:0] tail_slot,tail_group;
    wire [7:0] tail_generation,tail_layer,tail_mask;
    wire [15:0] tail_image;
    assign {tail_owner,tail_slot,tail_generation,tail_group,tail_layer,tail_image,tail_mask}=
        tail_token[3*TOKEN_WIDTH-1 -: TOKEN_WIDTH];
    // Decode each owner once and use constant slices. A dynamic 256-bit
    // part-select replicated across 3 clusters mapped to large mux networks.
    wire [NUM_FRONTS-1:0] owner_select;
    genvar selected_front;
    generate for(selected_front=0;selected_front<NUM_FRONTS;selected_front=selected_front+1)begin:GEN_OWNER_SELECT
        assign owner_select[selected_front]=DIRECT_DOT_OWNER ? (bypass_mode ? selected_front==0:dot_valid[selected_front]) : tail_owner==selected_front;
    end endgenerate
    reg [167:0] selected_previous;
    reg [300:0] selected_packet;
    reg [519:0] selected_header;
    integer select_index,header_index;
    always @* begin

        selected_packet=0;
        for(select_index=0;select_index<NUM_FRONTS;select_index=select_index+1)
            selected_packet=selected_packet | ({301{owner_select[select_index]}} &
                {dot_valid[select_index],dot_data[select_index*256 +: 256],
                 dot_slot[select_index*2 +: 2],dot_generation[select_index*8 +: 8],
                 dot_group[select_index*2 +: 2],dot_layer_id[select_index*8 +: 8],
                 dot_image_tag[select_index*16 +: 16],dot_expected_mask[select_index*8 +: 8]});
        selected_header=0;
        for(header_index=0;header_index<4;header_index=header_index+1)
            selected_header=selected_header | ({520{(PRESELECTED_HEADER ? header_index==0 : tail_group==header_index)}} &
                {header_bias[header_index*256 +: 256],header_multiplier[header_index*256 +: 256],
                 header_relu[header_index*8 +: 8]});
    end
    // Production banks return each active front again within eight MAC returns.
    // Store the last eight selected prefixes ONCE at the RQ bank, rather than
    // 8 private copies plus a second 8-to-1 168-bit selection network.
    // Four-bit saturating ages reject a reuse after eviction. First uses zero;
    // pool configuration resets both DSP accumulators and this validity state.
    wire history_bad;
    wire rq_issue;
    wire [255:0] selected_dots=selected_packet[299:44];
    generate if(BANK_PREFIX_HISTORY!=0)begin:BANK_HISTORY
        reg [NUM_FRONTS-1:0] seen;
        reg [3:0] age[0:NUM_FRONTS-1];
        reg selected_seen;reg [3:0] selected_age;
        wire push_history=rq_issue && !bypass_mode;
        wire [167:0] prior;
        always @*begin
            selected_seen=0;selected_age=0;
            for(integer p=0;p<NUM_FRONTS;p=p+1)begin
                selected_seen=selected_seen | (seen[p] && owner_select[p]);
                selected_age=selected_age | (age[p] & {4{owner_select[p]}});
            end
        end
        always @* selected_previous=selected_seen ? prior:168'd0;
        assign history_bad=!bypass_mode && selected_seen && selected_age>=8;
        for(genvar l=0;l<8;l=l+1)begin:LANE
            for(genvar k=0;k<21;k=k+1)begin:BIT
                wire q;
                EFX_SRL8 history(.D(selected_dots[l*32+k]),.A(~selected_age[2:0]),
                    .CLK(clk),.CE(push_history),.Q(q),.Q7());
                assign prior[l*21+k]=~q;
            end
        end
        always @(posedge clk)begin
            if(reset || epoch_clear)begin
                seen<=0;
                for(integer p=0;p<NUM_FRONTS;p=p+1)age[p]<=0;
            end else if(push_history)begin
                for(integer p=0;p<NUM_FRONTS;p=p+1)begin
                    if(owner_select[p])begin seen[p]<=1;age[p]<=0;end
                    else if(seen[p] && age[p]<8)age[p]<=age[p]+4'd1;
                end
            end
        end
    end else begin:PRIVATE_HISTORY
        always @*begin
            selected_previous=0;
            for(integer p=0;p<NUM_FRONTS;p=p+1)
                selected_previous=selected_previous | ({168{owner_select[p]}} & dot_previous[p*168+:168]);
        end
        assign history_bad=1'b0;
    end endgenerate

    wire owner_in_range=|owner_select;
    wire owner_cancel=|(owner_select & owner_cancel_pending);
    wire tuple_bad=history_bad || !owner_in_range || (!bypass_mode && (!selected_packet[300] ||
        selected_packet[43:0]!={tail_slot,tail_generation,tail_group,tail_layer,tail_image,tail_mask})) ||
        tail_layer!=header_layer_id || tail_image!=header_image_tag || {1'b0,tail_group}>=header_groups;
    wire tail_bad=tail_valid[2] && !owner_cancel && tuple_bad;
    wire tail_killed=owner_cancel || tail_bad;
    assign rq_issue=!reset && tail_valid[2] && !tail_killed;
    wire [255:0] selected_bias=selected_header[519:264];
    wire [255:0] selected_multiplier=selected_header[263:8];
    wire [7:0] selected_relu=selected_header[7:0];
    wire [7:0] rq_valid,rq_range,rq_bias_overflow;
    wire [63:0] rq_data;
    genvar lane;
    generate for(lane=0;lane<8;lane=lane+1)begin:GEN_RQ
        wire [20:0] delta=selected_dots[lane*32+:21]-selected_previous[lane*21+:21];
        wire signed [31:0] raw_dot=$signed({{11{delta[20]}},delta});
        wire signed [31:0] bias=$signed(selected_bias[lane*32 +: 32]);
        wire signed [32:0] biased=$signed({raw_dot[31],raw_dot})+$signed({bias[31],bias});
        wire signed [31:0] clamped=REJECT_BIAS_OVERFLOW!=0 ? biased[31:0] : biased>I32_MAX ? 32'sh7fffffff :
            biased<I32_MIN ? 32'sh80000000 : biased[31:0];
        assign rq_bias_overflow[lane]=!bypass_mode && tail_mask[lane] && ((biased>I32_MAX)||(biased<I32_MIN));
        requantize_i32_mul18_q17_pipeline requant (
            .clk(clk),.rst_n(!reset),.in_valid(rq_issue),
            .value(bypass_mode ? {{23{bypass_pipe[144+lane*9+8]}},bypass_pipe[144+lane*9+:9]} : clamped),
            .multiplier_field(bypass_mode ? {{14{bypass_multiplier[17]}},bypass_multiplier} : selected_multiplier[lane*32 +: 32]),
            .relu_enable(!bypass_mode && selected_relu[lane]),
            .out_valid(rq_valid[lane]),.result(rq_data[lane*8 +: 8]),.multiplier_range_error(rq_range[lane])
        );
    end endgenerate
    reg [3:0] return_valid_q,return_killed_q;
    reg [4*RETURN_WIDTH-1:0] return_token_q;
    reg [31:0] bias_overflow_q;
    assign return_valid=!reset && return_valid_q[3];
    assign {return_owner,return_slot,return_generation,return_mask}=
        return_token_q[4*RETURN_WIDTH-1 -: RETURN_WIDTH];
    assign return_killed=return_killed_q[3] || owner_cancel_pending[return_owner];
    assign return_bias_overflow=return_killed ? 8'd0 : bias_overflow_q[31:24] & return_mask;
    assign return_range_error=return_killed ? 8'd0 : rq_range & return_mask;
    generate for(lane=0;lane<8;lane=lane+1)begin:GEN_RETURN
        assign return_data[lane*8 +: 8]=(return_killed || !return_mask[lane] || rq_range[lane]) ?
            8'd0 : rq_data[lane*8 +: 8];
    end endgenerate
    // With zero input-tail delay, ISSUE is a combinational handshake, not
    // resident state. Including it in BUSY loops through cfg_ready -> run.
    // Configuration has priority before issue; accepted work appears in q.
    assign busy=!reset && ((|return_valid_q) || (|rq_valid));
    always @(posedge clk)begin
        if(reset)begin
            return_valid_q<=0;return_killed_q<=0;
            return_token_q<=0;bias_overflow_q<=0;protocol_error<=0;
        end else begin
            return_valid_q<={return_valid_q[2:0],tail_valid[2]};
            return_killed_q<={return_killed_q[2:0],tail_killed};
            return_token_q<={return_token_q[3*RETURN_WIDTH-1:0],tail_owner,tail_slot,tail_generation,tail_mask};
            bias_overflow_q<={bias_overflow_q[23:0],rq_bias_overflow};
            if(tail_bad || (rq_valid != {8{return_valid_q[3] && !return_killed_q[3]}}))protocol_error<=1;
        end
    end
endmodule
