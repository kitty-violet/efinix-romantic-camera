// Continuous circular last-term admission with constant-index priority.
// The initial cursor splits the circle into [cursor,N) then [0,cursor).
// Removing each winner preserves that order for the remaining output lanes;
// no dynamic request[index] scan or fixed partition of owners is needed.
// A disabled output does not consume a request or advance the cursor.
module finish_rr_arbiter #(
    parameter integer NUM_FRONTS=9,
    parameter integer NUM_GRANTS=1,
    parameter integer OWNER_WIDTH=(NUM_FRONTS>1 ? $clog2(NUM_FRONTS) : 1)
) (
    input wire clk,reset,
    input wire [NUM_FRONTS-1:0] request,
    input wire [NUM_GRANTS-1:0] grant_enable,
    output reg [NUM_FRONTS-1:0] grant,
    output reg [NUM_GRANTS-1:0] selected_valid,
    output reg [NUM_GRANTS*OWNER_WIDTH-1:0] selected_owner
);
    reg [OWNER_WIDTH-1:0] cursor,next_cursor,winner;
    reg [NUM_FRONTS-1:0] order_mask,pending,eligible,take;
    reg found;
    integer owner,lane;
    always @* begin
        order_mask=0;
        for(owner=0;owner<NUM_FRONTS;owner=owner+1)
            order_mask[owner]=(owner>=cursor);
        grant=0;selected_valid=0;selected_owner=0;next_cursor=cursor;
        pending=request;eligible=0;take=0;found=0;winner=0;
        for(lane=0;lane<NUM_GRANTS;lane=lane+1)begin
            eligible=pending & order_mask;
            if(eligible==0)eligible=pending;
            take=0;found=0;winner=0;
            for(owner=0;owner<NUM_FRONTS;owner=owner+1)begin
                if(eligible[owner] && !found)begin
                    take[owner]=1;found=1;winner=owner[OWNER_WIDTH-1:0];
                end
            end
            if(!reset && grant_enable[lane] && found)begin
                grant=grant | take;pending=pending & ~take;
                selected_valid[lane]=1;
                selected_owner[lane*OWNER_WIDTH +: OWNER_WIDTH]=winner;
                next_cursor=(winner==NUM_FRONTS-1) ? {OWNER_WIDTH{1'b0}} : winner+1'b1;
            end
        end
    end
    always @(posedge clk)begin
        if(reset)cursor<=0;
        else cursor<=next_cursor;
    end
endmodule
