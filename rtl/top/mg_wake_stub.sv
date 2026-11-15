// Stand-in for the wake spotter (docs/architecture.md section 3.1) until the wake agent's RTL lands in rtl/wake/:
// same ports, plus a test port. It fires `wake` with the sample whose index (counted from the first sample with
// en = 1, as the spotter's n) equals tst_at, when tst_en is set. It reads nothing from config SRAM.
module mg_wake_stub (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               en,
    input  wire               in_valid,
    input  wire signed [15:0] in_data,
    input  wire               in_vad,
    input  wire               in_raw,
    input  wire [14:0]        tpl_addr,
    input  wire [7:0]         tpl_len,
    input  wire [15:0]        sens,
    output wire               cfg_req,
    output wire [14:0]        cfg_addr,
    input  wire               cfg_gnt,
    input  wire [31:0]        cfg_rdata,
    output wire               wake,
    output wire               busy,
    input  wire               tst_en,
    input  wire [31:0]        tst_at
);
    reg [31:0] n;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) n <= 32'd0;
        else if (!en) n <= 32'd0;
        else if (in_valid) n <= n + 32'd1;
    assign wake = en && in_valid && tst_en && (n == tst_at);
    assign cfg_req = 1'b0;
    assign cfg_addr = 15'd0;
    assign busy = 1'b0;
endmodule
