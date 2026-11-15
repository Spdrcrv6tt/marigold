// Island control (docs/architecture.md sections 3, 5, 6, 8): I2C target and its register file, config loader and
// config SRAM, matcher, event sequencing, event frames (I2C / UART / irq) and the GPIO rule engine, all on clk_fe.
// The core reaches it through the island side of the bridge (br_*: one access per br_req, answered by br_ack).
// The wake spotter has the highest-priority config read port (spot_*).
module mg_ctl #(
    parameter [6:0]  I2C_DEF = 7'h2A,
    parameter [31:0] SEG_PRE = 32'd4800,
    parameter [31:0] SEG_MIN = 32'd8000,
    parameter [31:0] SEG_CAP = 32'd96000,
    parameter [7:0]  SEG_HANG = 8'd8,
    parameter [7:0]  SEG_ONSET = 8'd3,
    parameter [31:0] SEG_LISTEN = 32'd64000,
    parameter        UDIV = 26,
    parameter        MS = 3072
) (
    input  wire        clk,
    input  wire        rst_n,
    // pins
    input  wire        scl_i,
    input  wire        sda_i,
    output wire        sda_oe,
    output wire        uart_tx_o,
    output wire        irq_o,
    output wire [7:0]  gpio_o,
    // events from the capture ring
    input  wire        ev_wake,
    input  wire        ev_tmo,
    // state for the rest of the island
    output wire        cfg_valid,
    output wire        listening,
    output wire [14:0] wake_base,
    output wire [7:0]  wake_len,
    output wire [15:0] wake_sens,
    // wake spotter config read port
    input  wire        spot_req,
    input  wire [14:0] spot_addr,
    output wire        spot_gnt,
    output wire        spot_rvalid,
    output wire [31:0] cfg_rdata,
    // core bridge, island side
    input  wire        br_req,
    input  wire        br_we,
    input  wire [15:0] br_addr,
    input  wire [31:0] br_wdata,
    output reg         br_ack,
    output reg  [31:0] br_rdata
);
    // ---------------------------------------------------------------- config
    wire        ld_begin, ld_commit, ld_valid, cfg_we;
    wire [7:0]  ld_byte;
    wire        ld_in;
    wire [3:0]  ld_err;
    wire [15:0] ld_cnt, D, N, flags;
    wire [7:0]  conf_sh;
    wire [14:0] cfg_waddr, vec_base, act_base, slot_base, gpio_base;
    wire [7:0]  cfg_wdata, n_actions, n_slots, n_gpio, host, i2c_addr;
    wire [31:0] reject_thr, margin;
    mg_cfg_load u_ld (.clk(clk), .rst_n(rst_n), .begin_i(ld_begin), .commit_i(ld_commit), .in_valid(ld_in),
        .in_data(ld_byte), .we(cfg_we), .waddr(cfg_waddr), .wdata(cfg_wdata), .valid(ld_valid), .err(ld_err),
        .cnt(ld_cnt), .D(D), .N(N), .n_actions(n_actions), .n_slots(n_slots), .n_gpio(n_gpio), .wake_len(wake_len),
        .host(host), .i2c_addr(i2c_addr), .reject_thr(reject_thr), .margin(margin), .wake_sens(wake_sens),
        .flags(flags), .conf_sh(conf_sh), .vec_base(vec_base), .act_base(act_base), .slot_base(slot_base), .gpio_base(gpio_base),
        .wake_base(wake_base));
    assign cfg_valid = ld_valid;

    wire        g_req, m_req, b_req;
    wire [14:0] g_addr, m_addr;
    reg  [14:0] b_addr;
    wire [3:0]  gnt, rvalid;
    mg_cfg_ram #(.NR(4)) u_ram (.clk(clk), .rst_n(rst_n), .we(cfg_we), .waddr(cfg_waddr), .wdata(cfg_wdata),
        .req({b_req, m_req, g_req, spot_req}), .addr({b_addr, m_addr, g_addr, spot_addr}), .gnt(gnt),
        .rvalid(rvalid), .rdata(cfg_rdata));
    assign spot_gnt = gnt[0];
    assign spot_rvalid = rvalid[0];

    // ---------------------------------------------------------------- matcher
    wire        m_go, m_emit, m_busy, m_wait, m_done, m_acc, m_has;
    wire [7:0]  m_action, m_conf, m_mask, m_a1;
    wire [15:0] m_best;
    wire [31:0] m_sb, m_s2;
    wire [255:0] m_vals;
    wire        q_we = br_req && br_we && (br_addr[15:8] == 8'h81);
    wire        v_we = br_req && br_we && (br_addr[15:5] == 11'b1000_0000_010);     // 0x8040 .. 0x805C
    mg_match u_m (.clk(clk), .rst_n(rst_n), .go(m_go), .emit(m_emit), .D(D), .N(N), .n_actions(n_actions), .n_slots(n_slots),
        .vec_base(vec_base), .act_base(act_base), .slot_base(slot_base), .reject_thr(reject_thr), .margin(margin),
        .conf_sh(conf_sh), .q_we(q_we), .q_waddr(br_addr[7:2]), .q_wdata(br_wdata), .v_we(v_we), .v_waddr(br_addr[4:2]),
        .v_wdata(br_wdata), .req(m_req), .addr(m_addr), .gnt(gnt[2]), .rvalid(rvalid[2]), .rdata(cfg_rdata),
        .busy(m_busy), .waiting(m_wait), .done(m_done), .accept(m_acc), .action(m_action), .best(m_best), .best_action(m_a1), .sb(m_sb), .s2(m_s2),
        .has_s2(m_has), .conf(m_conf), .slot_mask(m_mask), .vals(m_vals));
    assign m_go   = br_req && br_we && (br_addr == 16'h8008) && br_wdata[0] && ld_valid && !m_busy;
    assign m_emit = br_req && br_we && (br_addr == 16'h8008) && br_wdata[1] && !br_wdata[0] && m_wait;

    // ---------------------------------------------------------------- events
    reg         pw, pt, pm, m_done_q;
    reg  [1:0]  e_ev;
    reg  [7:0]  e_id;
    wire        o_busy;
    wire        e_go = !o_busy && (pw || pt || pm);
    always @(*) begin
        if (pw)      begin e_ev = 2'd1; e_id = 8'hF0; end
        else if (pt) begin e_ev = 2'd3; e_id = 8'hF2; end
        else if (m_acc) begin e_ev = 2'd0; e_id = m_action; end
        else         begin e_ev = 2'd2; e_id = 8'hF1; end
    end
    wire        to_host = (host != 8'd0) && ((e_ev == 2'd0) || flags[e_ev == 2'd1 ? 1 : e_ev == 2'd2 ? 2 : 3]);
    wire        msg_busy, gpio_busy;
    assign o_busy = msg_busy || gpio_busy;

    // ---------------------------------------------------------------- frames and GPIO
    wire [5:0]  head_len;
    wire [7:0]  head_byte;
    wire        dropped, pop, drop_all, clr_dropped;
    mg_msg #(.UDIV(UDIV)) u_msg (.clk(clk), .rst_n(rst_n), .go(e_go && to_host), .ev(e_ev), .action(m_action),
        .slot_mask(m_mask), .vals(m_vals), .conf(m_conf), .busy(msg_busy), .host(host), .irq_en(ld_valid && flags[0]),
        .head_len(head_len), .head_byte(head_byte), .pop(pop), .drop_all(drop_all), .dropped(dropped),
        .clr_dropped(clr_dropped), .tx_o(uart_tx_o), .irq_o(irq_o));
    mg_gpio #(.MS(MS)) u_gpio (.clk(clk), .rst_n(rst_n), .go(e_go), .id(e_id), .slot_mask(e_ev == 2'd0 ? m_mask : 8'd0),
        .vals(m_vals), .n_actions(n_actions), .n_gpio(n_gpio), .act_base(act_base), .gpio_base(gpio_base),
        .req(g_req), .addr(g_addr), .gnt(gnt[1]), .rvalid(rvalid[1]), .rdata(cfg_rdata), .busy(gpio_busy),
        .gpio_o(gpio_o));

    // ---------------------------------------------------------------- I2C and its registers
    wire [7:0]  ptr, wr_reg, wr_data, rd_reg;
    wire        wr_stb, rd_stb;
    wire [6:0]  dev = (ld_valid && i2c_addr[6:0] != 7'd0) ? i2c_addr[6:0] : I2C_DEF;
    reg  [7:0]  rd_data;
    reg         loading, listen_en;
    mg_i2c u_i2c (.clk(clk), .rst_n(rst_n), .scl_i(scl_i), .sda_i(sda_i), .sda_oe(sda_oe), .addr(dev), .ptr(ptr),
        .wr_stb(wr_stb), .wr_reg(wr_reg), .wr_data(wr_data), .rd_stb(rd_stb), .rd_reg(rd_reg), .rd_data(rd_data));
    wire [7:0]  status = {ld_err, listening, (head_len != 6'd0), loading, ld_valid};
    always @(*) case (ptr)
        8'h00: rd_data = 8'h4D;
        8'h01: rd_data = status;
        8'h04: rd_data = {2'd0, head_len};
        8'h05: rd_data = head_byte;
        8'h06: rd_data = ld_cnt[7:0];
        8'h07: rd_data = ld_cnt[15:8];
        8'h08: rd_data = {7'd0, dropped};
        default: rd_data = 8'h00;
    endcase
    wire ctrl = wr_stb && (wr_reg == 8'h02);
    assign ld_begin    = ctrl && wr_data[0];
    assign ld_commit   = ctrl && wr_data[1];
    assign drop_all    = ctrl && wr_data[3];
    assign ld_in       = wr_stb && (wr_reg == 8'h03);
    assign ld_byte     = wr_data;
    assign pop         = rd_stb && (rd_reg == 8'h05);
    assign clr_dropped = rd_stb && (rd_reg == 8'h08);
    assign listening   = ld_valid && listen_en;

    // ---------------------------------------------------------------- bridge: registers and config reads
    reg  b_rd;                                     // a config read is waiting for the port
    assign b_req = b_rd;
    wire [31:0] st_reg = {24'd0, ld_err, (head_len != 6'd0), m_busy, listening, ld_valid};
    reg  [31:0] reg_rd;
    always @(*) case (br_addr[7:0])
        8'h00: reg_rd = 32'h4D474931;
        8'h04: reg_rd = st_reg;
        8'h08: reg_rd = {28'd0, m_wait, m_done && !m_acc, m_done, m_busy};
        8'h0C: reg_rd = {m_mask, m_a1, m_best};
        8'h10: reg_rd = m_sb;
        8'h14: reg_rd = m_has ? m_s2 : 32'h80000000;
        8'h20: reg_rd = SEG_PRE;
        8'h24: reg_rd = SEG_MIN;
        8'h28: reg_rd = SEG_CAP;
        8'h2C: reg_rd = {16'd0, SEG_ONSET, SEG_HANG};
        8'h30: reg_rd = SEG_LISTEN;
        default: reg_rd = 32'd0;
    endcase

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pw <= 1'b0; pt <= 1'b0; pm <= 1'b0; m_done_q <= 1'b0; loading <= 1'b0; listen_en <= 1'b0;
            b_rd <= 1'b0; b_addr <= 15'd0; br_ack <= 1'b0; br_rdata <= 32'd0;
        end else begin
            // events: WAKE, TIMEOUT and match results queue here and go out one at a time
            m_done_q <= m_done;
            if (ev_wake && listening) pw <= 1'b1; else if (e_go && pw) pw <= 1'b0;
            if (ev_tmo && listening) pt <= 1'b1; else if (e_go && !pw && pt) pt <= 1'b0;
            if (m_done && !m_done_q) pm <= 1'b1; else if (e_go && !pw && !pt) pm <= 1'b0;
            // I2C control
            if (ld_begin) begin loading <= 1'b1; listen_en <= 1'b0; end
            if (ld_commit) begin loading <= 1'b0; listen_en <= 1'b1; end
            if (ctrl && !wr_data[0] && !wr_data[1]) begin
                if (wr_data[2]) listen_en <= 1'b1;
                if (wr_data[4]) listen_en <= 1'b0;
            end
            // bridge
            br_ack <= 1'b0;
            if (br_req) begin
                if (!br_addr[15]) begin b_rd <= 1'b1; b_addr <= br_addr[14:0]; end    // config: via the port
                else begin br_ack <= 1'b1; br_rdata <= reg_rd; end
            end
            if (b_rd && gnt[3]) b_rd <= 1'b0;
            if (rvalid[3]) begin br_ack <= 1'b1; br_rdata <= cfg_rdata; end
        end
    end
endmodule
