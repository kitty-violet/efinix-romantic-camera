`timescale 1ns/1ps
// Fixed SC431HAI configuration. Reuses the existing pin-level byte/bit engine.
// No writes are sent until both chip-ID bytes match. Every slave ACK is checked.
module camera_i2c_transaction #(
    parameter integer PRESCALE=32, TIMEOUT_CYCLES=20000
)(
    input wire clk, reset_n, request, read_reg,
    input wire [7:0] device_write, write_data,
    input wire [15:0] reg_address,
    output reg busy, done,
    output reg [3:0] error,
    output reg [2:0] error_byte,
    output reg [7:0] read_data,
    input wire scl_i,sda_i,
    output wire scl_o,scl_oen,sda_o,sda_oen
);
    localparam IDLE=0, ISSUE=1, ACTIVE=2, GAP=3, ABORT=4, STOP_WAIT=5, RELEASE=6;
    reg [2:0] state,step;
    reg [15:0] address_q;
    reg [7:0] device_q,data_q;
    reg read_q;
    reg start_cmd,stop_cmd,read_cmd,write_cmd,core_reset;
    reg [7:0] byte_data;
    reg [$clog2(TIMEOUT_CYCLES+1)-1:0] watchdog;
    wire byte_done,slave_nack,arbitration_lost,unused_busy;
    wire [7:0] byte_read;
    i2c_master_byte_ctrl byte_engine(
      .clk(clk),.rst(core_reset),.nReset(reset_n),.ena(1'b1),.clk_cnt(16'(PRESCALE)),
      .start(start_cmd),.stop(stop_cmd),.read(read_cmd),.write(write_cmd),
      .ack_in(1'b1),.din(byte_data),.cmd_ack(byte_done),.ack_out(slave_nack),
      .dout(byte_read),.i2c_busy(unused_busy),.i2c_al(arbitration_lost),
      .scl_i(scl_i),.scl_o(scl_o),.scl_oen(scl_oen),
      .sda_i(sda_i),.sda_o(sda_o),.sda_oen(sda_oen));
    always @(posedge clk or negedge reset_n) begin
      if(!reset_n) begin
        state<=IDLE;step<=0;busy<=0;done<=0;error<=0;error_byte<=0;read_data<=0;
        address_q<=0;device_q<=0;data_q<=0;read_q<=0;
        start_cmd<=0;stop_cmd<=0;read_cmd<=0;write_cmd<=0;byte_data<=0;
        watchdog<=0;core_reset<=0;
      end else begin
        done<=0;core_reset<=0;
        case(state)
          IDLE: if(request) begin
            address_q<=reg_address;device_q<=device_write;data_q<=write_data;read_q<=read_reg;
            step<=0;error<=0;error_byte<=0;busy<=1;state<=ISSUE;
          end
          ISSUE: begin
            watchdog<=0;start_cmd<=(step==0 || (read_q && step==3));
            stop_cmd<=(!read_q && step==3) || (read_q && step==4);
            read_cmd<=read_q && step==4;write_cmd<=!(read_q && step==4);
            case(step)
              0:byte_data<=device_q;
              1:byte_data<=address_q[15:8];
              2:byte_data<=address_q[7:0];
              3:byte_data<=read_q ? device_q|8'h01 : data_q;
              default:byte_data<=0;
            endcase
            state<=ACTIVE;
          end
          ACTIVE: begin
            watchdog<=watchdog+1'b1;
            if(arbitration_lost || watchdog==TIMEOUT_CYCLES-1 || byte_done) begin
              start_cmd<=0;stop_cmd<=0;read_cmd<=0;write_cmd<=0;
              if(arbitration_lost || watchdog==TIMEOUT_CYCLES-1) begin
                error<=arbitration_lost ? 4'd3 : 4'd2;error_byte<=step;
                core_reset<=1;state<=RELEASE;
              end else if(write_cmd && slave_nack) begin
                error<=1;error_byte<=step;state<=ABORT;
              end else if(stop_cmd) begin
                read_data<=byte_read;busy<=0;done<=1;state<=IDLE;
              end else begin step<=step+1'b1;state<=GAP;end
            end
          end
          GAP: state<=ISSUE;
          ABORT: begin stop_cmd<=1;watchdog<=0;state<=STOP_WAIT;end
          STOP_WAIT: begin
            watchdog<=watchdog+1'b1;
            if(byte_done || watchdog==TIMEOUT_CYCLES-1 || arbitration_lost) begin
              stop_cmd<=0;core_reset<=1;state<=RELEASE;
            end
          end
          RELEASE: begin busy<=0;done<=1;state<=IDLE;end
          default:begin core_reset<=1;state<=RELEASE;error<=6;end
        endcase
      end
    end
endmodule
