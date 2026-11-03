module axi_frame_write_dma #(
    parameter integer AXI_ADDR_WIDTH = 32,
    parameter integer AXI_DATA_WIDTH = 128,
    parameter integer AXI_ID_WIDTH = 4,
    parameter integer FRAME_ID_WIDTH = 16,
    parameter integer MAX_BURST_BEATS = 128
) (
    input  wire                         clk,
    input  wire                         rst_n,

    input  wire                         start_valid,
    output wire                         start_ready,
    input  wire [AXI_ADDR_WIDTH-1:0]    start_addr,
    input  wire [31:0]                  start_beats,
    input  wire [FRAME_ID_WIDTH-1:0]    start_frame_id,

    input  wire                         s_valid,
    output wire                         s_ready,
    input  wire [AXI_DATA_WIDTH-1:0]    s_data,
    input  wire                         abort,

    output wire [AXI_ID_WIDTH-1:0]      m_axi_awid,
    output wire [AXI_ADDR_WIDTH-1:0]    m_axi_awaddr,
    output wire [7:0]                   m_axi_awlen,
    output wire [2:0]                   m_axi_awsize,
    output wire [1:0]                   m_axi_awburst,
    output wire                         m_axi_awvalid,
    input  wire                         m_axi_awready,

    output wire [AXI_DATA_WIDTH-1:0]    m_axi_wdata,
    output wire [(AXI_DATA_WIDTH/8)-1:0] m_axi_wstrb,
    output wire                         m_axi_wlast,
    output wire                         m_axi_wvalid,
    input  wire                         m_axi_wready,

    input  wire [AXI_ID_WIDTH-1:0]      m_axi_bid,
    input  wire [1:0]                   m_axi_bresp,
    input  wire                         m_axi_bvalid,
    output wire                         m_axi_bready,

    output wire                         busy,
    output reg                          frame_done,
    output reg                          frame_failed,
    output reg [FRAME_ID_WIDTH-1:0]     completed_frame_id,
    output reg                          config_error,
    output reg                          protocol_error,
    output wire [1:0]                   graft_state
);

    localparam integer BYTES_PER_BEAT = AXI_DATA_WIDTH / 8;
    localparam integer ADDR_LSB = $clog2(BYTES_PER_BEAT);
    localparam [1:0] STATE_IDLE = 2'd0;
    localparam [1:0] STATE_AW   = 2'd1;
    localparam [1:0] STATE_W    = 2'd2;
    localparam [1:0] STATE_B    = 2'd3;

    reg [1:0] state;
    reg [AXI_ADDR_WIDTH-1:0] current_addr;
    reg [31:0] remaining_beats;
    reg [FRAME_ID_WIDTH-1:0] active_frame_id;
    reg [8:0] burst_beats;
    reg [8:0] burst_beat_index;
    reg abort_pending;
    reg stalled_wvalid;
    reg [AXI_DATA_WIDTH-1:0] stalled_wdata;

    function automatic [8:0] choose_burst_beats;
        input [AXI_ADDR_WIDTH-1:0] addr;
        input [31:0] beats_left;
        reg [12:0] bytes_to_4k;
        reg [8:0] beats_to_4k;
        reg [31:0] limited_beats;
        begin
            bytes_to_4k = 13'd4096 - {1'b0, addr[11:0]};
            beats_to_4k = bytes_to_4k >> ADDR_LSB;
            limited_beats = beats_left;
            if (limited_beats > MAX_BURST_BEATS)
                limited_beats = MAX_BURST_BEATS;
            if (limited_beats > beats_to_4k)
                limited_beats = beats_to_4k;
            choose_burst_beats = limited_beats[8:0];
        end
    endfunction

    assign start_ready = (state == STATE_IDLE);
    assign busy = (state != STATE_IDLE);

    assign m_axi_awid = {AXI_ID_WIDTH{1'b0}};
    assign m_axi_awaddr = current_addr;
    assign m_axi_awlen = burst_beats[7:0] - 1'b1;
    assign m_axi_awsize = ADDR_LSB[2:0];
    assign m_axi_awburst = 2'b01;
    assign m_axi_awvalid = (state == STATE_AW);

    // Preserve a beat already offered before abort.  Subsequent beats may be
    // zero-filled, but a stalled AXI offer cannot be changed or cancelled.
    assign m_axi_wdata = stalled_wvalid ? stalled_wdata :
                        ((abort_pending || abort) ?
                         {AXI_DATA_WIDTH{1'b0}} : s_data);
    assign m_axi_wstrb = {(AXI_DATA_WIDTH/8){1'b1}};
    assign m_axi_wlast = (burst_beat_index == (burst_beats - 1'b1));
    assign m_axi_wvalid = (state == STATE_W) &&
                          (stalled_wvalid || abort_pending || abort || s_valid);
    assign s_ready = (state == STATE_W) && !abort_pending && !abort &&
                     m_axi_wready;

    assign m_axi_bready = (state == STATE_B);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            current_addr <= {AXI_ADDR_WIDTH{1'b0}};
            remaining_beats <= 32'd0;
            active_frame_id <= {FRAME_ID_WIDTH{1'b0}};
            burst_beats <= 9'd0;
            burst_beat_index <= 9'd0;
            abort_pending <= 1'b0;
            stalled_wvalid <= 1'b0;
            stalled_wdata <= {AXI_DATA_WIDTH{1'b0}};
            frame_done <= 1'b0;
            frame_failed <= 1'b0;

            completed_frame_id <= {FRAME_ID_WIDTH{1'b0}};
            config_error <= 1'b0;
            protocol_error <= 1'b0;
        end else begin
            frame_done <= 1'b0;
            frame_failed <= 1'b0;

            // No extra latency or streaming bubble is introduced: the normal
            // data path remains a pass-through.  Capture only a blocked offer.
            if ((state == STATE_W) && m_axi_wvalid && !m_axi_wready) begin
                if (!stalled_wvalid)
                    stalled_wdata <= m_axi_wdata;
                stalled_wvalid <= 1'b1;
            end else begin
                stalled_wvalid <= 1'b0;
            end

            if ((state != STATE_B) && m_axi_bvalid)
                protocol_error <= 1'b1;

            case (state)
                STATE_IDLE: begin
                    if (start_valid) begin
                        config_error <= 1'b0;
                        protocol_error <= 1'b0;
                        if ((start_beats == 0) ||
                            (start_addr[ADDR_LSB-1:0] != {ADDR_LSB{1'b0}})) begin
                            config_error <= 1'b1;
                        end else begin
                            current_addr <= start_addr;
                            remaining_beats <= start_beats;
                            active_frame_id <= start_frame_id;
                            burst_beats <= choose_burst_beats(start_addr, start_beats);
                            burst_beat_index <= 9'd0;
                            abort_pending <= 1'b0;
                            state <= STATE_AW;
                        end
                    end
                end

                STATE_AW: begin
                    // AWVALID has already been offered.  Cancellation must
                    // keep AW stable, finish this burst and collect B before
                    // reporting failure; it must never withdraw AWVALID.
                    if (abort)
                        abort_pending <= 1'b1;
                    if (m_axi_awready) begin
                        burst_beat_index <= 9'd0;
                        state <= STATE_W;
                    end
                end

                STATE_W: begin
                    if (abort)
                        abort_pending <= 1'b1;
                    if (m_axi_wvalid && m_axi_wready) begin
                        current_addr <= current_addr + BYTES_PER_BEAT;
                        remaining_beats <= remaining_beats - 1'b1;
                        if (m_axi_wlast) begin
                            burst_beat_index <= 9'd0;
                            state <= STATE_B;
                        end else begin
                            burst_beat_index <= burst_beat_index + 1'b1;
                        end
                    end
                end

                STATE_B: begin
                    if (abort)
                        abort_pending <= 1'b1;
                    if (m_axi_bvalid) begin
                        if (abort_pending || abort) begin
                            frame_failed <= 1'b1;
                            abort_pending <= 1'b0;
                            state <= STATE_IDLE;
                        end else if ((m_axi_bid != {AXI_ID_WIDTH{1'b0}}) ||
                            (m_axi_bresp != 2'b00)) begin
                            protocol_error <= 1'b1;
                            frame_failed <= 1'b1;
                            abort_pending <= 1'b0;
                            state <= STATE_IDLE;
                        end else if (remaining_beats == 0) begin
                            completed_frame_id <= active_frame_id;
                            frame_done <= 1'b1;
                            abort_pending <= 1'b0;
                            state <= STATE_IDLE;
                        end else begin
                            burst_beats <= choose_burst_beats(current_addr,
                                                              remaining_beats);
                            state <= STATE_AW;
                        end
                    end
                end

                default: begin
                    protocol_error <= 1'b1;
                    state <= STATE_IDLE;
                end
            endcase
        end
    end

 assign graft_state=state;
endmodule
