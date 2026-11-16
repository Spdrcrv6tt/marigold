// Testbench wrapper for mg_wake: config SRAM model (random grant delays from an LFSR, rdata on the clock after
// the grant), sample counter, and a trace file of every debug event and wake (compared with the golden model by
// test_wake.py). The trace file name comes from the plusarg +TRACE=<path>, the template from +CFG=<mem file>.
module wake_tb #(
    parameter MEMDIR    = "",
    parameter ROM_STALL = 0,
    parameter CFG_STALL = 0,
    parameter LANES     = 4,
    parameter C         = 96,
    parameter NOUT      = 40,
    parameter GATE      = 0,
    parameter PREROLL   = 0
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               en,
    input  wire               in_valid,
    input  wire signed [15:0] in_data,
    input  wire               in_vad,
    input  wire [15:0]        sens,
    output wire               wake,
    output wire               busy
);
    localparam [14:0] TPL = 15'h0123;          // unaligned on purpose
    reg  [31:0] cfgm [0:8191];
    wire        cfg_req, cfg_gnt;
    wire [14:0] cfg_addr;
    reg  [31:0] cfg_rdata;
    reg  [15:0] lf;
    integer f, n;
    reg ovr_seen = 1'b0;
    string trace, cfgf;
    initial begin
        if (!$value$plusargs("TRACE=%s", trace)) trace = "trace.txt";
        if (!$value$plusargs("CFG=%s", cfgf)) cfgf = "cfg.mem";
        $readmemh(cfgf, cfgm);
        f = $fopen(trace, "w");
    end
    always @(posedge clk or negedge rst_n)
        if (!rst_n) lf <= 16'h1D0F; else lf <= {lf[14:0], lf[15] ^ lf[13] ^ lf[12] ^ lf[10]};
    assign cfg_gnt = cfg_req & ((CFG_STALL == 0) | (lf[1:0] == 2'd0));
    always @(posedge clk) begin
        cfg_rdata <= cfg_gnt ? cfgm[cfg_addr[14:2]] : 32'hDEADBEEF;   // valid only on the clock after the grant
    end
    wire        dv;  wire [1:0] dk;  wire [31:0] dd;
    mg_wake #(.MEMDIR(MEMDIR), .ROM_STALL(ROM_STALL), .LANES(LANES), .C(C), .NOUT(NOUT), .GATE(GATE), .PREROLL(PREROLL)) dut (.clk(clk), .rst_n(rst_n), .en(en), .in_valid(in_valid), .in_data(in_data),
        .in_vad(in_vad), .in_raw(1'b0), .tpl_addr(TPL), .tpl_len(8'd0), .sens(sens), .cfg_req(cfg_req), .cfg_addr(cfg_addr),
        .cfg_gnt(cfg_gnt), .cfg_rdata(cfg_rdata), .wake(wake), .busy(busy), .dbg_v(dv), .dbg_k(dk), .dbg_d(dd));
    longint bcyc = 0, nfr = 0;
    always @(posedge clk) begin
        if (dut.busy) bcyc <= bcyc + 1;
        if (dut.fe_start) nfr <= nfr + 1;
    end
    final $display("WAKE_STATS frames=%0d busy_clocks=%0d per_frame=%0d", nfr, bcyc, (nfr > 0) ? bcyc / nfr : 0);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) n <= 0;
        else begin
            if (!en) n <= 0;
            else if (in_valid) n <= n + 1;
            if (dv) begin $fwrite(f, "%0d %0d\n", dk, $signed(dd)); $fflush(f); end
            if (wake) begin $fwrite(f, "W %0d\n", n); $fflush(f); end
            if (dut.ovr && !ovr_seen) begin $fwrite(f, "OVERRUN\n"); $fflush(f); end
            ovr_seen <= dut.ovr;
`ifdef WAKE_DEBUG
            if (dut.u_nn.h_we) $fwrite(f, "H %0d %08x\n", dut.u_nn.h_wa, dut.u_nn.w_wd);
            if (dut.u_nn.v_we) $fwrite(f, "V %0d %08x\n", dut.u_nn.v_wa, dut.u_nn.w_wd);
            if (dut.u_nn.o_we) $fwrite(f, "O %0d %08x\n", dut.u_nn.o_wa, dut.u_nn.w_wd);
`endif
        end
    end
endmodule
