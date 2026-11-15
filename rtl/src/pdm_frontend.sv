// PDM microphone front end: PDM clock out, 1-bit data in, 16-bit PCM out at f_pdm / (4*R).
// Default: clk 49.152 MHz, DIV 16 -> PDM clock 3.072 MHz, R 48 -> 16 kHz PCM.
// (24.576 MHz with DIV 8, or 12.288 MHz with DIV 4, give the same rates.)
// The FIR coefficients in pdm_coefs.svh are designed for R = 48, N = 5.
// gain: output gain of 2^gain (0..7), saturating. dc_en: 20 Hz DC blocker.
// Reference: rtl/model/pdm_model.py pdm_decimate().
module pdm_frontend #(
    parameter DIV       = 16,
    parameter R         = 48,
    parameter N         = 5,
    parameter CIC_W     = 29,
    parameter CIC_SHIFT = 9,
    parameter DW        = 20,
    parameter HB_TAPS   = 23,
    parameter FIR_TAPS  = 63
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               en,
    input  wire [2:0]         gain,
    input  wire               dc_en,
    output wire               pdm_clk_o,
    input  wire               pdm_dat_i,
    output wire               pcm_valid,
    output wire signed [15:0] pcm_data,
    output wire               overrun
);
    wire bit_valid, bit_data;
    pdm_clkgen #(.DIV(DIV)) u_clk (
        .clk(clk), .rst_n(rst_n), .en(en), .pdm_dat_i(pdm_dat_i),
        .pdm_clk_o(pdm_clk_o), .bit_valid(bit_valid), .bit_data(bit_data));

    wire                    cic_valid;
    wire signed [CIC_W-1:0] cic_data;
    cic_decim #(.R(R), .N(N), .W(CIC_W)) u_cic (
        .clk(clk), .rst_n(rst_n), .in_valid(bit_valid), .in_bit(bit_data),
        .out_valid(cic_valid), .out_data(cic_data));

    // round CIC_W -> DW bits (cannot overflow by more than the rounding step; saturate anyway)
    wire signed [CIC_W:0] cic_x  = {cic_data[CIC_W-1], cic_data};
    wire signed [CIC_W:0] cic_h  = {{(CIC_W-CIC_SHIFT+1){1'b0}}, 1'b1, {(CIC_SHIFT-1){1'b0}}};
    wire signed [CIC_W:0] cic_r  = (cic_x + cic_h) >>> CIC_SHIFT;
    wire                  a_ovf  = (cic_r[CIC_W:DW-1] != {(CIC_W-DW+2){1'b0}}) && (cic_r[CIC_W:DW-1] != {(CIC_W-DW+2){1'b1}});
    wire signed [DW-1:0]  a_data = a_ovf ? {cic_r[CIC_W], {(DW-1){~cic_r[CIC_W]}}} : cic_r[DW-1:0];

    wire                 hb_valid;
    wire signed [DW-1:0] hb_data;
    wire                 ovr_a, ovr_b;
    fir_decim2 #(.NT(HB_TAPS), .DW(DW), .OW(DW), .SHIFT(15), .SEL(0)) u_hb (
        .clk(clk), .rst_n(rst_n), .in_valid(cic_valid), .in_data(a_data), .gain(3'd0),
        .out_valid(hb_valid), .out_data(hb_data), .overrun(ovr_a));

    wire               fir_valid;
    wire signed [15:0] fir_data;
    fir_decim2 #(.NT(FIR_TAPS), .DW(DW), .OW(16), .SHIFT(DW - 1), .SEL(1)) u_fir (
        .clk(clk), .rst_n(rst_n), .in_valid(hb_valid), .in_data(hb_data), .gain(gain),
        .out_valid(fir_valid), .out_data(fir_data), .overrun(ovr_b));

    dc_block #(.W(16), .K(7)) u_dc (
        .clk(clk), .rst_n(rst_n), .en(dc_en), .in_valid(fir_valid), .in_data(fir_data),
        .out_valid(pcm_valid), .out_data(pcm_data));

    assign overrun = ovr_a | ovr_b;
endmodule
