// Test bench wrapper: spi_flash_reader + behavioural flash, joined through delayed tri-state
// lines. t_out models the chip's output pad plus board (SCK, CS#, IO towards the flash),
// t_in the way back (flash output through the input pad), t_clqv the flash clock-to-output.
// All three are in ps and are set by the cocotb test.
`timescale 1ps/1ps
module flash_tb #(
    parameter IOW  = 4,
    parameter AW   = 24,
    parameter HALF = 1
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          start,
    input  wire          stop,
    input  wire [AW-1:0] addr,
    input  wire [2:0]    mode,
    input  wire [4:0]    dummy,
    input  wire [1:0]    rx_dly,
    output wire          busy,
    output wire [7:0]    out_data,
    output wire          out_valid,
    input  wire          out_ready,
    input  wire [31:0]   t_out,
    input  wire [31:0]   t_in,
    input  wire [31:0]   t_clqv,
    output wire          bad_cmd,
    output wire [7:0]    last_cmd,
    output wire [31:0]   n_selects,
    output reg           contention,
    output wire          sck_pin,
    output wire          cs_n_pin
);
    wire           sck_o, cs_n_o;
    wire [IOW-1:0] io_o, io_oe;
    reg  [IOW-1:0] io_i;

    spi_flash_reader #(.IOW(IOW), .AW(AW), .HALF(HALF)) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .stop(stop), .addr(addr), .mode(mode),
        .dummy(dummy), .rx_dly(rx_dly), .busy(busy),
        .out_data(out_data), .out_valid(out_valid), .out_ready(out_ready),
        .sck_o(sck_o), .cs_n_o(cs_n_o), .io_o(io_o), .io_oe(io_oe), .io_i(io_i));

    // towards the flash
    reg            sck_f, cs_n_f;
    reg  [IOW-1:0] h_o_f, h_oe_f;
    initial begin sck_f = 0; cs_n_f = 1; h_o_f = 0; h_oe_f = 0; io_i = {IOW{1'b1}}; contention = 0; end
    always @(sck_o)  sck_f  <= #(t_out) sck_o;
    always @(cs_n_o) cs_n_f <= #(t_out) cs_n_o;
    always @(io_o)   h_o_f  <= #(t_out) io_o;
    always @(io_oe)  h_oe_f <= #(t_out) io_oe;

    wire [IOW-1:0] f_o, f_oe_raw;
    wire [IOW-1:0] f_oe = f_oe_raw & {IOW{~cs_n_f}};
    // the line at the flash: host drives, else flash drives, else pulled up
    wire [IOW-1:0] line = (h_oe_f & h_o_f) | (~h_oe_f & f_oe & f_o) | (~h_oe_f & ~f_oe);

    flash_model #(.IOW(IOW), .AW(AW)) flash (
        .sck(sck_f), .cs_n(cs_n_f), .io_in(line), .io_out(f_o), .io_oe(f_oe_raw),
        .t_clqv(t_clqv), .bad_cmd(bad_cmd), .last_cmd(last_cmd), .n_selects(n_selects));

    // back to the controller
    always @(line) io_i <= #(t_in) line;

    always @(h_oe_f or f_oe) if ((h_oe_f & f_oe) != {IOW{1'b0}}) contention = 1;

    assign sck_pin  = sck_f;
    assign cs_n_pin = cs_n_f;
endmodule
