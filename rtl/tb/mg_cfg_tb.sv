// Unit test wrapper: config loader + config SRAM + matcher, with a testbench-driven reader of higher priority than
// the matcher (stall source). test_cfg.py.
module mg_cfg_tb #(
    parameter [15:0] MODEL_D = 16'd0,
    parameter [7:0]  MODEL_ID = 8'd0,
    parameter [7:0]  TEACHER_ID = 8'd0
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        begin_i,
    input  wire        commit_i,
    input  wire        in_valid,
    input  wire [7:0]  in_data,
    input  wire        noise_req,         // reader 0 (beats the matcher)
    input  wire        go,
    input  wire        q_we,
    input  wire [5:0]  q_waddr,
    input  wire [31:0] q_wdata,
    input  wire        v_we,
    input  wire [2:0]  v_waddr,
    input  wire [31:0] v_wdata,
    output wire        valid,
    output wire [3:0]  err,
    output wire [15:0] cnt,
    output wire        busy,
    output wire        done,
    output wire        accept,
    output wire [7:0]  action,
    output wire [15:0] best,
    output wire [31:0] sb,
    output wire [31:0] s2,
    output wire        has_s2,
    output wire [7:0]  conf,
    output wire [7:0]  slot_mask,
    output wire [255:0] vals,
    output wire [14:0] act_base,
    output wire [14:0] gpio_base
);
    wire        we;
    wire [14:0] waddr, vec_base, slot_base, wake_base, m_addr;
    wire [7:0]  wdata, n_actions, n_slots, n_gpio, wake_len, host, i2c_addr;
    wire [15:0] D, N, wake_sens, flags;
    wire [7:0]  conf_sh;
    wire [31:0] reject_thr, margin, rdata;
    wire [1:0]  gnt, rvalid;
    wire        m_req;
    mg_cfg_load #(.MODEL_D(MODEL_D), .MODEL_ID(MODEL_ID), .TEACHER_ID(TEACHER_ID)) u_ld (.clk(clk), .rst_n(rst_n), .begin_i(begin_i), .commit_i(commit_i), .in_valid(in_valid),
        .in_data(in_data), .we(we), .waddr(waddr), .wdata(wdata), .valid(valid), .err(err), .cnt(cnt), .D(D), .N(N),
        .n_actions(n_actions), .n_slots(n_slots), .n_gpio(n_gpio), .wake_len(wake_len), .host(host),
        .i2c_addr(i2c_addr), .reject_thr(reject_thr), .margin(margin), .wake_sens(wake_sens), .flags(flags), .conf_sh(conf_sh),
        .vec_base(vec_base), .act_base(act_base), .slot_base(slot_base), .gpio_base(gpio_base), .wake_base(wake_base));
    mg_cfg_ram #(.NR(2)) u_ram (.clk(clk), .rst_n(rst_n), .we(we), .waddr(waddr), .wdata(wdata),
        .req({m_req, noise_req}), .addr({m_addr, 15'd0}), .gnt(gnt), .rvalid(rvalid), .rdata(rdata));
    mg_match u_m (.clk(clk), .rst_n(rst_n), .go(go), .abort(begin_i), .D(D), .N(N), .n_actions(n_actions), .n_slots(n_slots),
        .vec_base(vec_base), .act_base(act_base), .slot_base(slot_base), .reject_thr(reject_thr), .margin(margin),
        .conf_sh(conf_sh), .q_we(q_we), .q_waddr(q_waddr), .q_wdata(q_wdata), .v_we(v_we), .v_waddr(v_waddr), .v_wdata(v_wdata),
        .req(m_req), .addr(m_addr), .gnt(gnt[1]), .rvalid(rvalid[1]), .rdata(rdata), .busy(busy), .done(done),
        .accept(accept), .action(action), .best(best), .best_action(), .sb(sb), .s2(s2), .has_s2(has_s2), .conf(conf),
        .slot_mask(slot_mask), .vals(vals));
endmodule
