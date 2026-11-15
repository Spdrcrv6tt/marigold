// First-order DC blocker: y = x - x[-1] + (1 - 2^-K) y[-1], corner about fs / (2*pi*2^K).
// The state keeps K fraction bits. With en = 0 the sample passes through (one clock later)
// and the state is held at zero.
// Reference: rtl/model/pdm_model.py dc_block().
module dc_block #(
    parameter W = 16,
    parameter K = 7
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                en,
    input  wire                in_valid,
    input  wire signed [W-1:0] in_data,
    output reg                 out_valid,
    output reg  signed [W-1:0] out_data
);
    localparam SW = W + K + 2;
    reg  signed [SW-1:0] s;
    reg  signed [W-1:0]  xp;

    wire signed [W:0]    dx   = {in_data[W-1], in_data} - {xp[W-1], xp};
    wire signed [SW-1:0] dxs  = {{(SW-W-1-K){dx[W]}}, dx, {K{1'b0}}};
    wire signed [SW-1:0] sn   = s - (s >>> K) + dxs;
    wire signed [SW-1:0] half = {{(SW-K){1'b0}}, 1'b1, {(K-1){1'b0}}};
    wire signed [SW-1:0] rnd  = (sn + half) >>> K;
    wire                 ovf  = (rnd[SW-1:W-1] != {(SW-W+1){1'b0}}) && (rnd[SW-1:W-1] != {(SW-W+1){1'b1}});
    wire signed [W-1:0]  satv = ovf ? {rnd[SW-1], {(W-1){~rnd[SW-1]}}} : rnd[W-1:0];

    wire gclk;
    clk_gate u_cg (.clk(clk), .en(in_valid | out_valid), .gclk(gclk));

    always @(posedge gclk or negedge rst_n) begin
        if (!rst_n) begin
            s         <= {SW{1'b0}};
            xp        <= {W{1'b0}};
            out_valid <= 1'b0;
            out_data  <= {W{1'b0}};
        end else begin
            out_valid <= in_valid;
            if (in_valid) begin
                if (en) begin
                    s        <= sn;
                    xp       <= in_data;
                    out_data <= satv;
                end else begin
                    s        <= {SW{1'b0}};
                    xp       <= {W{1'b0}};
                    out_data <= in_data;
                end
            end
        end
    end
endmodule
