// Coefficient image v1 transport. Software owns schema, CRC/SHA and semantic
// validation. load_complete is deliberately NOT permission to compute.
// Isolated extension: the same five resident banks expose five pipelined reads.
// Host load/verify/read barriers drain all fast tokens before changing state.
module coefficient_bank_store_fast (
    input wire clk, input wire reset, input wire cancel,
    input wire begin_valid, output wire begin_ready,
    input wire [31:0] begin_version, input wire [31:0] begin_image_bytes,
    input wire [15:0] begin_tag,
    input wire load_valid, output wire load_ready,
    input wire [11:0] load_index, input wire [63:0] load_data,
    input wire load_last,
    input wire verify_valid, output wire verify_ready,
    input wire [15:0] verify_tag, input wire verify_ok,
    input wire read_valid, output wire read_ready,
    input wire read_for_compute, input wire [2:0] read_bank,
    input wire [9:0] read_address,
    output wire response_valid, input wire response_ready,
    output wire [63:0] response_data,
    output reg [3:0] response_error,
    output reg [2:0] response_bank, output reg [9:0] response_address,
    output reg [15:0] response_tag,
    output reg loading, output reg load_complete,
    output wire coeff_ready, output reg error_valid,
    output reg [3:0] error_code, output reg [15:0] image_tag,
    output reg [11:0] loaded_words,
    input wire [4:0] fast_read_valid,output wire [4:0] fast_read_ready,
    input wire [44:0] fast_read_address,input wire [15:0] fast_read_tag,
    output wire [4:0] fast_response_valid,input wire [4:0] fast_response_ready,
    output wire [319:0] fast_response_data,output wire [44:0] fast_response_address,
    output wire [79:0] fast_response_tag,output wire [19:0] fast_response_error
);
    localparam [3:0] ERR_HEADER=1, ERR_ORDER=2, ERR_LAST=3,
        ERR_PADDING=4, ERR_VERIFY=5, ERR_STATE=6, ERR_ADDRESS=7;
    reg verified;
    reg [2:0] load_bank_q;
    reg [8:0] load_address_q;
    reg response_valid_q, pending;
    reg [2:0] pending_bank;
    reg [9:0] pending_address;
    reg [15:0] pending_tag;
    reg [3:0] pending_error;
    wire active = !reset && !cancel;
    wire [4:0] fast_pending,fast_held;
    wire fast_idle=!(|fast_pending) && !(|fast_held);
    wire idle_read = !pending && !response_valid_q && fast_idle;
    // Begin/verify are control barriers; held responses cannot be overwritten.
    assign begin_ready = active && idle_read;
    assign verify_ready = active && idle_read && !begin_valid;
    assign read_ready = active && fast_idle && !pending &&
        (!response_valid_q || response_ready) && !begin_valid && !verify_valid;
    assign load_ready = active && loading && !begin_valid && !verify_valid && !read_valid;
    assign response_valid = active && response_valid_q;
    assign coeff_ready = active && verified && load_complete && !error_valid;
    wire begin_fire = begin_valid && begin_ready;
    wire verify_fire = verify_valid && verify_ready;
    wire read_fire = read_valid && read_ready;
    wire load_fire = load_valid && load_ready;
    wire load_order_ok = load_index == loaded_words;
    wire load_last_ok = load_last == (loaded_words == 12'd2559);
    wire load_padding_ok = (loaded_words < 12'd2224) || (load_data == 64'd0);
    wire write_fire = load_fire && load_order_ok && load_last_ok && load_padding_ok;
    wire read_state_ok = load_complete && !error_valid &&
        (!read_for_compute || verified);
    wire read_address_ok = read_bank < 3'd5 && !read_address[9];
    wire read_memory = read_fire && read_state_ok && read_address_ok;
    wire [319:0] bank_data;
    // Host response owns the same RAM outputs until accepted; no payload copy.
    assign response_data=response_error==0 ? bank_data[response_bank*64+:64]:64'd0;
    wire fast_enabled=coeff_ready && !begin_valid && !verify_valid && !read_valid && !pending && !response_valid_q;
    genvar bank;
    generate for (bank=0; bank<5; bank=bank+1) begin : banks
        cnn_weight_bank64 #(.DEPTH(512), .ADDR_WIDTH(9)) storage (
            .clk(clk), .write_enable(write_fire && load_bank_q == bank),
            .write_address(load_address_q), .write_data(load_data),
            .read_enable((read_memory && read_bank == bank) || (fast_read_valid[bank] && fast_read_ready[bank] && fast_read_tag==image_tag)),
            .read_address(fast_enabled ? fast_read_address[bank*9+:9] : read_address[8:0]),
            .read_data(bank_data[bank*64+:64])
        );
        // A single elastic token lives in the synchronous RAM output itself.
        // Consume-and-replace still accepts every bank on every cycle (II=1).
        // No new read can overwrite a stalled response; barriers include this
        // token. A rejected tag emits zero without reading retained contents.
        reg out_valid_q,out_bad_tag;
        reg [8:0] out_addr;
        wire take=fast_read_valid[bank] && fast_read_ready[bank];
        assign fast_pending[bank]=1'b0;
        assign fast_held[bank]=out_valid_q;
        assign fast_read_ready[bank]=fast_enabled && (!out_valid_q || fast_response_ready[bank]);
        assign fast_response_valid[bank]=active && out_valid_q;
        assign fast_response_data[bank*64+:64]=out_bad_tag ? 64'd0:bank_data[bank*64+:64];
        assign fast_response_address[bank*9+:9]=out_addr;
        // image_tag cannot change until every fast token has drained.
        assign fast_response_tag[bank*16+:16]=image_tag;
        assign fast_response_error[bank*4+:4]=out_bad_tag ? 4'd8:4'd0;
        always @(posedge clk)begin
            if(reset || cancel)begin out_valid_q<=0;out_addr<=0;out_bad_tag<=0;end
            else begin
                if(out_valid_q && fast_response_ready[bank])out_valid_q<=0;
                if(take)begin
                    out_valid_q<=1;out_addr<=fast_read_address[bank*9+:9];
                    out_bad_tag<=fast_read_tag!=image_tag;
                end
            end
        end
    end endgenerate

    // No RAM reset. Eligibility, all pending reads and held responses are
    // synchronously revoked; stale physical bytes are inaccessible.
    always @(posedge clk) begin
        if (reset || cancel) begin
            loading <= 0; load_complete <= 0; verified <= 0;
            error_valid <= 0; error_code <= 0; image_tag <= 0;
            loaded_words <= 0; load_bank_q <= 0; load_address_q <= 0;
            pending <= 0; pending_bank <= 0; pending_address <= 0;
            pending_tag <= 0; pending_error <= 0;
            response_valid_q <= 0; response_error <= 0;
            response_bank <= 0; response_address <= 0; response_tag <= 0;
        end else begin
            if (response_valid_q && response_ready) response_valid_q <= 0;
            if (pending) begin
                pending <= 0;
                response_valid_q <= 1;
                response_error <= pending_error;
                response_bank <= pending_bank;
                response_address <= pending_address;
                response_tag <= pending_tag;
            end
            if (begin_fire) begin
                image_tag <= begin_tag;
                loaded_words <= 0; load_bank_q <= 0; load_address_q <= 0;
                verified <= 0; load_complete <= 0;
                if (begin_version != 32'd1 || begin_image_bytes != 32'd17792) begin
                    loading <= 0; error_valid <= 1; error_code <= ERR_HEADER;
                end else begin
                    loading <= 1; error_valid <= 0; error_code <= 0;
                end
            end else if (verify_fire) begin
                if (load_complete && !error_valid && verify_ok && verify_tag == image_tag) begin
                    verified <= 1;
                end else begin
                    loading <= 0; load_complete <= 0; verified <= 0;
                    error_valid <= 1; error_code <= ERR_VERIFY;
                end
            end else if (read_fire) begin
                pending <= 1; pending_bank <= read_bank;
                pending_address <= read_address; pending_tag <= image_tag;
                pending_error <= !read_state_ok ? ERR_STATE :
                    (!read_address_ok ? ERR_ADDRESS : 4'd0);
                if (!read_state_ok || !read_address_ok) begin
                    loading <= 0; load_complete <= 0; verified <= 0;
                    error_valid <= 1;
                    error_code <= !read_state_ok ? ERR_STATE : ERR_ADDRESS;
                end
            end else if (load_fire) begin
                if (!load_order_ok || !load_last_ok || !load_padding_ok) begin
                    loading <= 0; load_complete <= 0; verified <= 0;
                    error_valid <= 1;
                    error_code <= !load_order_ok ? ERR_ORDER :
                        (!load_last_ok ? ERR_LAST : ERR_PADDING);
                end else begin
                    loaded_words <= loaded_words + 1'b1;
                    if (load_bank_q == 3'd4) begin
                        load_bank_q <= 0;
                        load_address_q <= load_address_q + 1'b1;
                    end else load_bank_q <= load_bank_q + 1'b1;
                    if (load_last) begin loading <= 0; load_complete <= 1; end
                end
            end
        end
    end
endmodule
