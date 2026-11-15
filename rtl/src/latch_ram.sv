// Small standard-cell memory: an array of latches (FLOPS = 0) or flip-flops (FLOPS = 1),
// one write port, one registered read port. For sizes where the PDK's SRAM macros (smallest
// is 64 x 8 at 0.10 mm2) are larger than the storage they hold.
//
// Latch version: write data is registered, then the addressed word's gated clock is high for
// the first half of the following cycle and the latches take it. Only the written word
// switches. A word written in cycle t can be read (re with its address) from cycle t+1 on;
// rdata appears the cycle after re.
// No reset: contents are undefined until written.
module latch_ram #(
    parameter WORDS = 64,
    parameter WIDTH = 8,
    parameter FLOPS = 0
) (
    input  wire                     clk,
    input  wire                     we,
    input  wire [$clog2(WORDS)-1:0] waddr,
    input  wire [WIDTH-1:0]         wdata,
    input  wire                     re,
    input  wire [$clog2(WORDS)-1:0] raddr,
    output reg  [WIDTH-1:0]         rdata
);
    reg [WIDTH-1:0] mem [0:WORDS-1];

    generate
        if (FLOPS != 0) begin : g_flop
            always @(posedge clk) begin
                if (we) mem[waddr] <= wdata;
            end
        end else begin : g_latch
            reg  [WIDTH-1:0] wd;
            wire [WORDS-1:0] wclk;
            wire             gclk_w;
            clk_gate u_cgw (.clk(clk), .en(we), .gclk(gclk_w));
            always @(posedge gclk_w) wd <= wdata;
            genvar g;
            for (g = 0; g < WORDS; g = g + 1) begin : g_word
                clk_gate u_wg (.clk(clk), .en(we && (waddr == g)), .gclk(wclk[g]));
                /* verilator lint_off LATCH */
                always @* begin
                    if (wclk[g]) mem[g] = wd;
                end
                /* verilator lint_on LATCH */
            end
        end
    endgenerate

    wire gclk_r;
    clk_gate u_cgr (.clk(clk), .en(re), .gclk(gclk_r));
    always @(posedge gclk_r) rdata <= mem[raddr];
endmodule
