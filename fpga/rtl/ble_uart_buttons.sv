`timescale 1ns/1ps
// UART command source only. All permission/ready/fault policy stays in the
// existing post-debounce button event logic. No PLL, DMA, or frame ownership.
module ble_uart_buttons #(
    parameter integer CLOCK_HZ = 150000000,
    parameter integer BAUD = 115200
)(
    input wire clk, input wire reset, input wire uart_rx,
    output reg key_start, output reg key_style, output reg key_view,
    output reg bad_frame
);
    localparam integer BIT_CYCLES=(CLOCK_HZ+BAUD/2)/BAUD;
    localparam integer CW=$clog2(BIT_CYCLES+1);
    localparam integer TIMEOUT=BIT_CYCLES*40;
    localparam integer TW=$clog2(TIMEOUT+1);
    (* async_reg="true", syn_preserve="true", syn_srlstyle="registers" *) reg rx_meta=1'b1, rx_sync=1'b1;
    always @(posedge clk) begin
        if(reset) begin rx_meta<=1'b1;rx_sync<=1'b1;end
        else begin rx_meta<=uart_rx;rx_sync<=rx_meta;end
    end
    localparam IDLE=0,START=1,DATA=2,STOP=3,RECOVER=4;
    reg [2:0] state;
    reg [CW-1:0] timer;
    reg [2:0] bit_index;
    reg [7:0] shift,rx_data;
    reg rx_valid,rx_error;
    always @(posedge clk) begin
        rx_valid<=0;rx_error<=0;
        if(reset)begin state<=IDLE;timer<=0;bit_index<=0;shift<=0;rx_data<=0;end
        else case(state)
            IDLE:if(!rx_sync)begin timer<=BIT_CYCLES/2-1;state<=START;end
            START:if(timer!=0)timer<=timer-1'b1;
                  else if(rx_sync)state<=IDLE;
                  else begin timer<=BIT_CYCLES-1;bit_index<=0;state<=DATA;end
            DATA:if(timer!=0)timer<=timer-1'b1;
                 else begin shift[bit_index]<=rx_sync;timer<=BIT_CYCLES-1;
                     if(bit_index==7)state<=STOP;else bit_index<=bit_index+1'b1;end
            STOP:if(timer!=0)timer<=timer-1'b1;
                 else if(rx_sync)begin rx_data<=shift;rx_valid<=1;state<=IDLE;end
                 else begin rx_error<=1;state<=RECOVER;timer<=BIT_CYCLES-1;end
            RECOVER:if(!rx_sync)timer<=BIT_CYCLES-1;
                    else if(timer!=0)timer<=timer-1'b1;else state<=IDLE;
            default:state<=IDLE;
        endcase
    end
    reg [1:0] packet_state;
    reg [7:0] command;
    reg [TW-1:0] age;
    wire [TW-1:0] age_next;
    // UART timeout is a slow-control counter. Use parallel Boolean toggle
    // terms instead of a dedicated carry chain that requires consecutive sites.
    assign age_next[0]=~age[0];
    generate for(genvar k=1;k<TW;k=k+1)begin:age_toggle
        assign age_next[k]=age[k] ^ (&age[k-1:0]);
    end endgenerate
    always @(posedge clk)begin
        key_start<=0;key_style<=0;key_view<=0;bad_frame<=0;
        if(reset)begin packet_state<=0;command<=0;age<=0;end
        else if(rx_error)begin packet_state<=0;age<=0;bad_frame<=1;end
        else if(rx_valid)begin
            age<=0;
            case(packet_state)
                0:if(rx_data==8'ha5)packet_state<=1;
                1:if(rx_data>=1 && rx_data<=3)begin command<=rx_data;packet_state<=2;end
                  else begin packet_state <= (rx_data == 8'ha5) ? 1 : 0;bad_frame<=1;end
                2:begin
                    packet_state <= (rx_data == 8'ha5) ? 1 : 0;
                    if(rx_data==~command)begin
                        key_start<=command==1;key_style<=command==2;key_view<=command==3;
                    end else bad_frame<=1;
                end
                default:packet_state<=0;
            endcase
        end else if(packet_state!=0)begin
            if(age==TIMEOUT-1)begin age<=0;packet_state<=0;bad_frame<=1;end
            else age<=age_next;
        end
    end
endmodule
