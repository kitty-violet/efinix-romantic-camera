module resize_axis_up_2_to_3 #(
    parameter integer SRC_SIZE = 1280,
    parameter integer DST_SIZE = 1920,
    parameter integer INDEX_WIDTH = 11,
    parameter integer COUNT_WIDTH = 11
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,
    input  wire                     step,
    output reg                      valid,
    output reg                      last,
    output reg [INDEX_WIDTH-1:0]    index0,
    output reg [INDEX_WIDTH-1:0]    index1,
    output reg [2:0]                weight1_num
);

    // Half-pixel coordinate for a 2:3 source/destination ratio:
    // source = (4 * destination - 1) / 6.
    // Destination zero is the replicated left/top border.  Thereafter a
    // Bresenham-style +4/6 accumulator avoids a general divider.
    reg [INDEX_WIDTH-1:0] source_index;
    reg [2:0]             fraction;
    reg [COUNT_WIDTH-1:0] destination_index;

    wire [3:0] fraction_plus_four = {1'b0, fraction} + 4'd4;
    wire       advance_source = (fraction_plus_four >= 4'd6);
    wire [3:0] reduced_fraction = fraction_plus_four - 4'd6;
    wire [2:0] advanced_fraction = advance_source
        ? reduced_fraction[2:0]
        : fraction_plus_four[2:0];
    wire [INDEX_WIDTH-1:0] advanced_index = source_index + advance_source;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            source_index      <= 0;
            fraction          <= 0;
            destination_index <= 0;
            valid             <= 1'b0;
            last              <= 1'b0;
            index0            <= 0;
            index1            <= 0;
            weight1_num       <= 0;
        end else if (start) begin
            source_index      <= 0;
            fraction          <= 0;
            destination_index <= 0;
            valid             <= 1'b1;
            last              <= (DST_SIZE == 1);
            index0            <= 0;
            index1            <= 0;
            weight1_num       <= 0;
        end else if (step && valid) begin
            if (last) begin
                valid <= 1'b0;
                last  <= 1'b0;
            end else begin
                destination_index <= destination_index + 1'b1;
                last              <= (destination_index == DST_SIZE - 2);

                if (destination_index == 0) begin
                    source_index <= 0;
                    fraction     <= 3;
                    index0       <= 0;
                    index1       <= (SRC_SIZE == 1) ? 0 : 1;
                    weight1_num  <= (SRC_SIZE == 1) ? 0 : 3;
                end else begin
                    source_index <= advanced_index;
                    fraction     <= advanced_fraction;
                    index0       <= advanced_index;
                    if (advanced_index >= SRC_SIZE - 1) begin
                        index1          <= SRC_SIZE - 1;
                        weight1_num     <= 0;
                    end else begin
                        index1          <= advanced_index + 1'b1;
                        weight1_num     <= advanced_fraction;
                    end
                end
            end
        end
    end

endmodule

