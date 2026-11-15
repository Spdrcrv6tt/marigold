// PDM clock generator and bit sampler.
// DIV = 1: the system clock is the PDM clock (lowest power: nothing runs faster than the
// microphone). pdm_clk_o is clk gated by en; the mic drives its bit while pdm_clk is high, so
// the bit is captured on the falling edge and handed to the rising-edge logic.
// DIV >= 4:
// pdm_clk_o = clk / DIV, 50% duty (DIV even, >= 4). The mic drives its data after the rising
// edge of pdm_clk; the bit is taken through a two-flop synchroniser and accepted once per
// PDM period at count SAMPLE_AT (default just after the falling edge, which leaves the mic
// about half a PDM period of data-valid delay).
module pdm_clkgen #(
    parameter DIV       = 16,
    parameter SAMPLE_AT = DIV / 2 + 1
) (
    input  wire clk,
    input  wire rst_n,
    input  wire en,
    input  wire pdm_dat_i,
    output reg  pdm_clk_o,
    output reg  bit_valid,
    output reg  bit_data
);
  generate
  if (DIV == 1) begin : g_div1
    wire gclk;
    reg  sn, run;
    clk_gate u_cg (.clk(clk), .en(en), .gclk(gclk));
    always @* pdm_clk_o = gclk;
    always @(negedge clk or negedge rst_n) begin
        if (!rst_n) sn <= 1'b0;
        else        sn <= pdm_dat_i;
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            run       <= 1'b0;
            bit_valid <= 1'b0;
            bit_data  <= 1'b0;
        end else begin
            run       <= en;
            bit_valid <= run & en;
            bit_data  <= sn;
        end
    end
  end else begin : g_div
    localparam CW = $clog2(DIV);
    reg [CW-1:0] cnt;
    reg s1, s2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt       <= {CW{1'b0}};
            pdm_clk_o <= 1'b0;
            bit_valid <= 1'b0;
            bit_data  <= 1'b0;
            s1        <= 1'b0;
            s2        <= 1'b0;
        end else begin
            s1        <= pdm_dat_i;
            s2        <= s1;
            bit_valid <= 1'b0;
            if (!en) begin
                cnt       <= {CW{1'b0}};
                pdm_clk_o <= 1'b0;
            end else begin
                cnt       <= (cnt == DIV - 1) ? {CW{1'b0}} : cnt + 1'b1;
                pdm_clk_o <= (cnt < DIV / 2);
                if (cnt == SAMPLE_AT) begin
                    bit_valid <= 1'b1;
                    bit_data  <= s2;
                end
            end
        end
    end
  end
  endgenerate
endmodule
