// Decimate-by-2 FIR with one multiplier and a latch-array delay line.
// y[m] = sat_OW(round((sum_i coef[i] * x[2m+1-i]) >>> (SHIFT - gain)))
//
// Low-power structure: samples are written into a circular buffer of latches, one word per
// input, through a per-word clock gate, so storage only switches when it is written. The
// multiply-accumulate registers sit behind one more clock gate that is open only while a
// sample is being taken in or an output is being computed. The per-word gates take that
// gated clock, not the free-running one (it always pulses on the write edge, because
// in_valid opens it), so between samples the clock does not reach them at all. (The first
// version rotated a shift register through the multiplier: every storage flop was clocked
// every cycle.)
//
// Timing: one product per clock, NT + 5 clocks from an odd input to its output; the caller
// must leave that many clocks before the next input (overrun goes high and stays high if
// not). SEL picks the coefficient table in pdm_coefs.svh: 0 = half-band, 1 = compensation.
// Reference: rtl/model/pdm_model.py fir_decim2() (arithmetic unchanged, still bit-exact).
module fir_decim2 #(
    parameter NT    = 23,
    parameter DW    = 20,
    parameter OW    = 20,
    parameter SHIFT = 15,
    parameter SEL   = 0
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 in_valid,
    input  wire signed [DW-1:0] in_data,
    input  wire [2:0]           gain,
    output reg                  out_valid,
    output reg  signed [OW-1:0] out_data,
    output reg                  overrun
);
`include "pdm_coefs.svh"

    localparam PW = DW + 16;
    localparam AW = PW + $clog2(NT);

    // ---- control and datapath registers, all on the gated clock gclk
    reg signed [DW-1:0] xin;        // input sample, held for the latch write
    reg [6:0]           wp;         // next word to write
    reg [6:0]           wsel;       // newest word
    reg [6:0]           rp;         // word being read
    reg [6:0]           idx;        // coefficient index of the word being read
    reg [6:0]           fill;       // words written since reset, saturates at NT
    reg                 wr, phase, start, busy, pv, fin;
    reg signed [PW-1:0] prod;
    reg signed [AW-1:0] acc;

    wire taking = in_valid && !(wr | start | busy);
    wire active = in_valid | wr | start | busy | pv | fin | out_valid;
    wire gclk;
    clk_gate u_cg (.clk(clk), .en(active), .gclk(gclk));

    // ---- latch array with one clock gate per word.
    // Cycle 0 (taking): xin and the word enable are captured. Cycle 1: the word's gated clock
    // is high for the first half cycle and the latch takes xin. Cycle 2 on: it can be read.
    reg  signed [DW-1:0] mem [0:NT-1];
    wire [NT-1:0]        wclk;
    genvar g;
    generate
        for (g = 0; g < NT; g = g + 1) begin : g_word
            clk_gate u_wg (.clk(gclk), .en(taking && (wp == g)), .gclk(wclk[g]));
            /* verilator lint_off LATCH */
            always @* begin
                if (wclk[g]) mem[g] = xin;
            end
            /* verilator lint_on LATCH */
        end
    endgenerate

    // words not written since reset read as zero (the model's x[n < 0] = 0)
    wire signed [DW-1:0] tap  = (idx < fill) ? mem[rp] : {DW{1'b0}};
    wire signed [15:0]   coef = (SEL == 0) ? pdm_hb_coef(idx) : pdm_fir_coef(idx);

    // rounding shift and saturation
    wire [4:0]            sh   = SHIFT - gain;
    wire signed [AW:0]    one  = {{AW{1'b0}}, 1'b1};
    wire signed [AW:0]    half = one <<< (sh - 5'd1);
    wire signed [AW:0]    accx = {acc[AW-1], acc};
    wire signed [AW:0]    rnd  = (accx + half) >>> sh;
    wire                  ovf  = (rnd[AW:OW-1] != {(AW-OW+2){1'b0}}) && (rnd[AW:OW-1] != {(AW-OW+2){1'b1}});
    wire signed [OW-1:0]  satv = ovf ? {rnd[AW], {(OW-1){~rnd[AW]}}} : rnd[OW-1:0];

    always @(posedge gclk or negedge rst_n) begin
        if (!rst_n) begin
            xin       <= {DW{1'b0}};
            wp        <= 7'd0;
            wsel      <= 7'd0;
            rp        <= 7'd0;
            idx       <= 7'd0;
            fill      <= 7'd0;
            wr        <= 1'b0;
            phase     <= 1'b0;
            start     <= 1'b0;
            busy      <= 1'b0;
            pv        <= 1'b0;
            fin       <= 1'b0;
            prod      <= {PW{1'b0}};
            acc       <= {AW{1'b0}};
            out_valid <= 1'b0;
            out_data  <= {OW{1'b0}};
            overrun   <= 1'b0;
        end else begin
            out_valid <= 1'b0;
            wr        <= 1'b0;
            start     <= 1'b0;
            pv        <= busy;
            fin       <= pv & ~busy;

            if (in_valid && !taking) overrun <= 1'b1;

            if (taking) begin
                xin   <= in_data;
                wsel  <= wp;
                wp    <= (wp == NT - 1) ? 7'd0 : wp + 7'd1;
                if (fill != NT) fill <= fill + 7'd1;
                wr    <= 1'b1;
                phase <= ~phase;
                start <= phase;
            end

            if (start) begin
                busy <= 1'b1;
                idx  <= 7'd0;
                rp   <= wsel;
                acc  <= {AW{1'b0}};
            end

            if (busy) begin
                prod <= coef * tap;
                rp   <= (rp == 7'd0) ? NT - 1 : rp - 7'd1;
                if (idx == NT - 1) busy <= 1'b0;
                else               idx  <= idx + 7'd1;
            end

            if (pv) acc <= acc + prod;

            if (fin) begin
                out_data  <= satv;
                out_valid <= 1'b1;
            end
        end
    end
endmodule
