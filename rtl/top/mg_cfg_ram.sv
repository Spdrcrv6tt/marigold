// Config SRAM (docs/architecture.md section 3): 32 KB of 32-bit words, one port, in the island (clk_fe), kept
// powered. The loader writes bytes with top priority; NR readers share the port by fixed priority (reader 0
// highest). A reader holds req (and addr) until gnt; the word at addr arrives the next clock with rvalid.
module mg_cfg_ram #(
    parameter AW = 15,                    // byte address bits (32 KB)
    parameter NR = 4
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               we,
    input  wire [AW-1:0]      waddr,
    input  wire [7:0]         wdata,
    input  wire [NR-1:0]      req,
    input  wire [NR*AW-1:0]   addr,
    output wire [NR-1:0]      gnt,
    output reg  [NR-1:0]      rvalid,
    output reg  [31:0]        rdata
);
    reg [31:0] mem [0:(1 << (AW - 2)) - 1];

    // fixed priority: the lowest requesting index, and nobody while the loader writes
    wire [NR-1:0] lower;                   // lower[i]: some reader below i requests
    genvar g;
    generate for (g = 0; g < NR; g = g + 1) begin : g_arb
        if (g == 0) begin : g0
            assign lower[g] = 1'b0;
        end else begin : gn
            assign lower[g] = lower[g-1] | req[g-1];
        end
        assign gnt[g] = req[g] & ~lower[g] & ~we;
    end endgenerate

    reg [AW-1:0] ra;
    integer i;
    always @(*) begin
        ra = {AW{1'b0}};
        for (i = 0; i < NR; i = i + 1) if (gnt[i]) ra = addr[i*AW +: AW];
    end

    wire en = we | (|req);
    wire gclk;
    clk_gate u_cg (.clk(clk), .en(en), .gclk(gclk));

    always @(posedge gclk) begin
        if (we) case (waddr[1:0])
            2'd0: mem[waddr[AW-1:2]][7:0]   <= wdata;
            2'd1: mem[waddr[AW-1:2]][15:8]  <= wdata;
            2'd2: mem[waddr[AW-1:2]][23:16] <= wdata;
            default: mem[waddr[AW-1:2]][31:24] <= wdata;
        endcase
        else if (|gnt) rdata <= mem[ra[AW-1:2]];
    end
    always @(posedge clk or negedge rst_n)
        if (!rst_n) rvalid <= {NR{1'b0}};
        else rvalid <= gnt;
endmodule
