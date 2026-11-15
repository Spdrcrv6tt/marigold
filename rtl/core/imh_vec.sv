// Per-element unit: the operations that happen once per activation instead of once per weight, as hardwired loops
// over the activation RAM. C reference: vec_go() in fw/hal_host.c; arithmetic: docs/golden-format.md. All loops are
// two-dimensional:
//     for r in 0..rows-1, i in 0..n-1:   source element src + r*srow + i*sstr,  destination dst + r*drow + i*dstr
//
//   MUL      dst = sat16(rshift_round(a * b, p0)), a at src, b at src + p1       (SiLU gate product)
//   COPY     dst = src (16 bit): convolution window gather, tensor re-layout, scatter. With B8 (go bit 7) the
//            destination is int8, addressed in bytes: dst = sat(rshift_round(src, p0), 127) (cross-attention K, V).
//   LUT      dst = lut(src): tanh / GELU / SiLU / asinh ROM with linear interpolation; optionally saturate;
//            optionally add the results into S and their squares into Q (GroupNorm statistics).
//   STAT     S += x, Q += x*x   (x 16 bit, or 32 bit in two planes)
//   NSTAT    from S, Q and the count p0: mean, variance (+ eps p1), integer square root, reciprocal:
//            mean / rec / sh1 of norm() in the golden model, kept for the next NORM with the AUTO flag.
//   NORM     dst = sat16(rshift_round(rshift_round((x - mean) * rec, sh1) * G, sh2) + B)
//            G (int16) and B (int32) come from the weight source, 6 bytes per channel in order
//            (one record per element, or per row for GroupNorm).
//   SOFTMAX  per row of n raw 48-bit scores (three planes): max, e = exp2 ROM of (max - s) * C >> sh,
//            e written as int32, S = sum(e), and floor(2^44 / S) written to a table (the attention reciprocal).
//   ROPE     rotary position: pairs (x at src + r*srow + i*sstr, partner at + p1) rotated by the Q14 cos/sin
//            row read from the weight source at faddr (16 cos, 16 sin).
//
// Pipelined (Poinsetta; Imhotep's unit took 2 to 14 clocks per element): one element enters a five-stage pipeline
// per clock. Source words come from three one-word caches (the operands at src, src + p1 and src + 2 p1), so a
// contiguous row costs one RAM read per 16 elements per operand; every result is written as it leaves its stage
// (one RAM access per element, byte enables), and reads take the port in the clocks without a write. NORM records
// stream from the weight source through a 32-byte buffer. The order of the C reference holds for any overlap of
// source and destination: a write updates the cached copies of its word, and an element does not start while an
// element ahead of it in the pipeline still has to write one of its operands. SOFTMAX: the maximum pass, then the
// exponent pass, then the divider (45 clocks per row). ROPE: one pair every second clock (two writes). The
// controller is stalled while an operation runs.
module imh_vec #(
    parameter LANES = 16
) (
    clk, rst_n, reg_we, reg_idx, reg_wdata, go, busy, done, s_acc, q_acc, o_mean, o_rec, o_sh1,
    m_en, m_we, m_addr, m_wdata, m_rdata,
    seek_valid, seek_addr, ws_data, ws_avail, ws_take
);
    localparam W  = LANES * 16;
    localparam NB = LANES * 2;
    localparam LB = $clog2(LANES);
    input  wire          clk;
    input  wire          rst_n;
    input  wire          reg_we;       // 0 src 1 sstr 2 srow 3 dst 4 dstr 5 drow 6 n 7 rows 8 p0 9 p1 10 p2 11 p3 12 faddr
    input  wire [4:0]    reg_idx;
    input  wire [31:0]   reg_wdata;
    input  wire          go;           // reg_wdata: [2:0] op, [3] clear S/Q, [4] wide source, [5] G/B per row, [6] auto,
                                       // [7] COPY to int8
    output wire          busy;
    output wire          done;
    output reg  signed [63:0] s_acc;
    output reg  [63:0]   q_acc;
    output wire [31:0]   o_mean;
    output wire [31:0]   o_rec;
    output wire [31:0]   o_sh1;
    output reg           m_en;
    output reg  [NB-1:0] m_we;
    output reg  [31:0]   m_addr;
    output reg  [W-1:0]  m_wdata;
    input  wire [W-1:0]  m_rdata;
    output wire          seek_valid;
    output wire [31:0]   seek_addr;
    input  wire [8*LANES-1:0] ws_data;
    input  wire [5:0]    ws_avail;
    output reg  [5:0]    ws_take;

    // Unit clock gate (power fix 1, reports/p1-power.md section 7): clocked only while busy (S_DONE included),
    // started (go) or written (reg_we). Idle with neither, no register here changes, so the gate acts like an enable.
    wire gclk;
    clk_gate u_cg (.clk(clk), .en(busy | go | reg_we), .gclk(gclk));

`include "imh_rom.svh"

    localparam [2:0] OP_MUL = 3'd0, OP_COPY = 3'd1, OP_LUT = 3'd2, OP_STAT = 3'd3, OP_NORM = 3'd4, OP_SOFTMAX = 3'd5,
                     OP_NSTAT = 3'd6, OP_ROPE = 3'd7;
    reg [31:0] r_src, r_sstr, r_srow, r_dst, r_dstr, r_drow, r_n, r_rows, r_p0, r_p1, r_p2, r_p3, r_faddr;
    always @(posedge gclk or negedge rst_n) begin
        if (!rst_n) begin
            r_src <= 0; r_sstr <= 0; r_srow <= 0; r_dst <= 0; r_dstr <= 0; r_drow <= 0; r_n <= 0; r_rows <= 0;
            r_p0 <= 0; r_p1 <= 0; r_p2 <= 0; r_p3 <= 0; r_faddr <= 0;
        end else if (reg_we) begin
            case (reg_idx)
                5'd0: r_src <= reg_wdata;
                5'd1: r_sstr <= reg_wdata;
                5'd2: r_srow <= reg_wdata;
                5'd3: r_dst <= reg_wdata;
                5'd4: r_dstr <= reg_wdata;
                5'd5: r_drow <= reg_wdata;
                5'd6: r_n <= reg_wdata;
                5'd7: r_rows <= reg_wdata;
                5'd8: r_p0 <= reg_wdata;
                5'd9: r_p1 <= reg_wdata;
                5'd10: r_p2 <= reg_wdata;
                5'd11: r_p3 <= reg_wdata;
                5'd12: r_faddr <= reg_wdata;
                default: ;
            endcase
        end
    end

    localparam [3:0] V_IDLE = 4'd0, V_SEEK = 4'd1, V_REC = 4'd2, V_RUN = 4'd3, V_DRAIN = 4'd4, V_DIV = 4'd5,
                     V_RCW = 4'd6, V_DONE = 4'd7, V_NS = 4'd8;
    reg [3:0]  st;
    reg [2:0]  op;
    reg        wide, gbrow, auto, pass, b8, issued_all;
    reg [31:0] r, i, sbase, dbase, sa, da;
    reg [511:0] crec;                  // ROPE: 16 cos, 16 sin from the weight source
    reg [6:0]  gcnt;
    reg signed [47:0] mx;
    reg [32:0] ssum;
    reg [5:0]  dcnt;
    reg [32:0] drem;
    reg [44:0] dquo;
    // statistics results (NSTAT), used by NORM with AUTO
    reg signed [31:0] n_mean;
    reg [32:0] n_rec;
    reg [5:0]  n_sh1;
    assign o_mean = n_mean;
    assign o_rec  = n_rec[31:0];
    assign o_sh1  = {26'd0, n_sh1};

    assign busy = (st != V_IDLE);
    assign done = (st == V_DONE);
    assign seek_valid = (st == V_SEEK);
    assign seek_addr  = r_faddr;

    // ---- operand caches: slot k holds one RAM word for the operand at sa + k*p1
    reg            c_v0, c_v1, c_v2;
    reg  [31:0]    c_a0, c_a1, c_a2;
    reg  [W-1:0]   c_d0, c_d1, c_d2;
    reg            rd_p;                             // a read was issued last clock
    reg  [1:0]     rd_k;
    reg  [31:0]    rd_a;
    wire [31:0]    ea0 = sa, ea1 = sa + r_p1, ea2 = sa + {r_p1[30:0], 1'b0};
    wire [31:0]    wa0 = ea0 >> LB, wa1 = ea1 >> LB, wa2 = ea2 >> LB;
    wire           need1 = (op == OP_MUL) || (op == OP_ROPE) || (op == OP_SOFTMAX) || ((op == OP_STAT || op == OP_NORM) && wide);
    wire           need2 = (op == OP_SOFTMAX);
    wire           hit0 = c_v0 && c_a0 == wa0, hit1 = c_v1 && c_a1 == wa1, hit2 = c_v2 && c_a2 == wa2;
    wire           coming0 = rd_p && rd_k == 2'd0 && rd_a == wa0;
    wire           coming1 = rd_p && rd_k == 2'd1 && rd_a == wa1;
    wire           coming2 = rd_p && rd_k == 2'd2 && rd_a == wa2;
    wire           miss0 = !hit0 && !coming0, miss1 = need1 && !hit1 && !coming1, miss2 = need2 && !hit2 && !coming2;
    wire [15:0]    e_x0 = c_d0[16 * ea0[LB-1:0] +: 16];
    wire [15:0]    e_x1 = c_d1[16 * ea1[LB-1:0] +: 16];
    wire [15:0]    e_x2 = c_d2[16 * ea2[LB-1:0] +: 16];
    wire signed [47:0] e_s48 = {e_x2, e_x1, e_x0};

    // ---- NORM records: a 32-byte buffer filled from the weight source
    reg  [255:0]   rb;
    reg  [5:0]     rb_cnt;
    reg  [31:0]    fetch_left;
    reg  [47:0]    grec;                             // G/B per row: this row's record

    // ---- pipeline stages 1..5: valid, destination (elements; bytes for COPY to int8), ROPE partner, NORM record
    reg            v1, v2, v3, v4, v5, v6;
    reg  [31:0]    d1, d2, d3, d4, d5, d6, q1, q2, q3;
    reg  [15:0]    x0_1, x1_1, x2_1;
    reg  [3:0]     ix1;
    reg  [47:0]    g1, g2, g3, g4, g5;

    // ---- hazards: an element waits while an element in the pipeline still has to write one of its operands
    wire           pairw = (op == OP_SOFTMAX), ropew = (op == OP_ROPE), bytew = (op == OP_COPY) && b8;
    function automatic hz1(input v, input [31:0] d, input [31:0] q, input [31:0] e, input pr, input rp, input by);
        reg [31:0] de;
        begin
            de = by ? {1'b0, d[31:1]} : d;
            hz1 = v && (e == de || (pr && e == de + 32'd1) || (rp && e == q));
        end
    endfunction
    function automatic hz(input [31:0] e);
        hz = hz1(v1, d1, q1, e, pairw, ropew, bytew) || hz1(v2, d2, q2, e, pairw, ropew, bytew) ||
             hz1(v3, d3, q3, e, pairw, ropew, bytew) || hz1(v4, d4, 32'd0, e, pairw, 1'b0, bytew) ||
             hz1(v5, d5, 32'd0, e, pairw, 1'b0, bytew) || hz1(v6, d6, 32'd0, e, pairw, 1'b0, bytew);
    endfunction
    wire hazard = hz(ea0) || (need1 && hz(ea1)) || (need2 && hz(ea2));
    wire norm_rec = (op == OP_NORM) && (!gbrow || i == 32'd0);           // this element takes a record
    wire [47:0] e_rec = (op == OP_NORM && gbrow && i != 32'd0) ? grec : rb[47:0];
    wire can_issue = (st == V_RUN) && !issued_all && hit0 && (!need1 || hit1) && (!need2 || hit2) && !hazard &&
                     (!norm_rec || rb_cnt >= 6'd6) && !(op == OP_ROPE && v1);
    wire to_pipe = can_issue && !(op == OP_SOFTMAX && !pass);            // the maximum pass ends at the issue
    wire el_last  = (i == r_n - 32'd1);
    wire row_last = (r == r_rows - 32'd1);

    // ---- LUT datapath (stage 1 -> 2 -> 3)
    wire [3:0]  l_f = r_p0[3:0], l_fo = r_p0[7:4];
    wire [1:0]  l_tab = r_p0[9:8];
    wire        l_tails = r_p0[10], l_sat = r_p0[11], l_acc = r_p0[12], l_r4 = r_p0[13];   // r4: range [-16, 16)
    wire [3:0]  l_fmin = l_r4 ? 4'd4 : 4'd5;
    wire [3:0]  l_fw = (l_f > l_fmin) ? l_f : l_fmin;
    wire signed [23:0] l_x0  = {{8{x0_1[15]}}, x0_1};
    wire signed [23:0] l_xs  = l_x0 <<< (l_fw - l_f);
    wire signed [23:0] l_lim = (l_r4 ? 24'sd16 : 24'sd8) <<< l_fw;
    wire signed [23:0] l_cl  = (l_xs < -l_lim) ? -l_lim : (l_xs > l_lim - 24'sd1) ? l_lim - 24'sd1 : l_xs;
    wire [23:0] l_u   = l_cl + l_lim;
    wire [3:0]  l_sh  = l_fw - l_fmin;
    wire [23:0] l_ix  = l_u >> l_sh;
    wire [9:0]  l_idx = l_ix[9:0];
    wire [23:0] l_fr  = l_u & ((24'd1 << l_sh) - 24'd1);
    reg  signed [23:0] a_xs;
    reg  signed [19:0] a_t0, a_dt;
    reg  [10:0] a_fr;
    reg         a_hi, a_lo;
    reg  signed [39:0] y_r;
    wire signed [31:0] l_ip  = a_dt * $signed({1'b0, a_fr});
    wire signed [39:0] l_ipx = {{8{l_ip[31]}}, l_ip};
    wire signed [39:0] l_t0x = {{20{a_t0[19]}}, a_t0};
    wire signed [39:0] l_y14 = l_t0x + (l_ipx >>> l_sh);
    wire signed [63:0] y_rx  = {{24{y_r[39]}}, y_r};
    function automatic signed [39:0] rrs(input signed [39:0] v, input signed [5:0] s);
        begin
            if (s <= 0) rrs = v <<< (-s);
            else        rrs = (v + (40'sd1 <<< (s - 1))) >>> s;
        end
    endfunction
    wire signed [5:0]  l_s1 = 6'sd14 - $signed({2'b00, l_fo});
    wire signed [5:0]  l_s2 = $signed({2'b00, l_fw}) - $signed({2'b00, l_fo});
    wire signed [39:0] l_yi = rrs(l_y14, l_s1);
    wire signed [39:0] l_yt = rrs({{16{a_xs[23]}}, a_xs}, l_s2);
    wire signed [39:0] l_y0 = !l_tails ? l_yi : a_hi ? l_yt : a_lo ? 40'sd0 : l_yi;
    wire signed [39:0] l_y  = !l_sat ? l_y0 : (l_y0 > 40'sd32767) ? 40'sd32767 : (l_y0 < -40'sd32767) ? -40'sd32767 : l_y0;

    // ---- STAT / NORM datapath (stage 1 -> 2 -> 3 -> 4 -> 5)
    wire signed [31:0] x32  = wide ? {x1_1, x0_1} : {{16{x0_1[15]}}, x0_1};
    wire signed [63:0] x64  = {{32{x32[31]}}, x32};
    wire signed [31:0] mean_v = auto ? n_mean : r_p0;
    wire [32:0]        rec_v  = auto ? n_rec : {1'b0, r_p2};
    wire [5:0]         n_sh1v = auto ? n_sh1 : r_p3[5:0];
    wire [5:0]         n_sh2  = r_p3[13:8];
    wire signed [32:0] n_d  = {x32[31], x32} - {mean_v[31], mean_v};
    reg  signed [32:0] d_r;
    reg  signed [63:0] p1_r;
    reg  signed [31:0] yn_r;
    reg  signed [47:0] p2_r;
    wire signed [66:0] n_p1  = d_r * $signed({1'b0, rec_v});
    wire signed [63:0] n_y   = (p1_r + (64'sd1 <<< (n_sh1v - 6'd1))) >>> n_sh1v;
    wire signed [15:0] g_g   = g4[15:0];
    wire signed [31:0] g_b   = g5[47:16];
    wire signed [47:0] n_p2  = yn_r * g_g;
    wire signed [47:0] g_bx  = {{16{g_b[31]}}, g_b};
    // stage 5 rounds and shifts (n_r5, registered in n_q6 with the bias, power fix 4), stage 6 adds the bias,
    // saturates and writes
    wire signed [47:0] n_r5  = (p2_r + (48'sd1 <<< (n_sh2 - 6'd1))) >>> n_sh2;
    reg  signed [47:0] n_q6, n_b6;
    wire signed [47:0] n_o0  = n_q6 + n_b6;
    wire signed [47:0] n_o   = (n_o0 > 48'sd32767) ? 48'sd32767 : (n_o0 < -48'sd32767) ? -48'sd32767 : n_o0;

    // ---- SOFTMAX datapath (exponent pass: stage 1 -> 2 -> 3)
    wire signed [47:0] s48 = {x2_1, x1_1, x0_1};
    wire [48:0] sm_d  = mx - s48;
    reg  [63:0] sp_r, tt_r;
    wire [5:0]  sm_sh = r_p0[21:16];
    wire [63:0] sm_tt = (sp_r + (64'd1 << (sm_sh - 6'd1))) >> sm_sh;
    wire [51:0] sm_n  = tt_r[63:12];
    wire [16:0] sm_e  = (sm_n > 52'd16) ? 17'd0 : (exp2_rom(tt_r[11:4]) >> sm_n[4:0]);
    wire [33:0] dv_sh = {drem, (dcnt == 6'd44)};
    wire        dv_ge = (dv_sh >= {1'b0, ssum});

    // ---- ROPE / MUL / COPY datapath (stage 1 -> 2; ROPE's partner written from stage 3)
    wire signed [15:0] xa = x0_1, xb = x1_1;
    wire signed [15:0] r_c = crec[16 * ix1 +: 16];
    wire signed [15:0] r_s = crec[256 + 16 * ix1 +: 16];
    wire signed [32:0] r_y1 = xa * r_c - xb * r_s;
    wire signed [32:0] r_y2 = xb * r_c + xa * r_s;
    wire signed [32:0] r_q1 = (r_y1 + 33'sd8192) >>> 14;
    wire signed [32:0] r_q2 = (r_y2 + 33'sd8192) >>> 14;
    wire signed [32:0] r_z1 = (r_q1 > 33'sd32767) ? 33'sd32767 : (r_q1 < -33'sd32767) ? -33'sd32767 : r_q1;
    wire signed [32:0] r_z2 = (r_q2 > 33'sd32767) ? 33'sd32767 : (r_q2 < -33'sd32767) ? -33'sd32767 : r_q2;
    reg  signed [15:0] ro1, ro2, ro2_3;
    wire signed [39:0] m_p  = xa * xb;
    wire signed [39:0] m_q  = rrs(m_p, {1'b0, r_p0[4:0]});
    wire signed [39:0] m_z  = (m_q > 40'sd32767) ? 40'sd32767 : (m_q < -40'sd32767) ? -40'sd32767 : m_q;
    wire signed [16:0] q8r = ($signed({x0_1[15], x0_1}) + ((r_p0[4:0] == 5'd0) ? 17'sd0 : (17'sd1 <<< (r_p0[4:0] - 5'd1))))
                             >>> r_p0[4:0];
    wire        [7:0]  q8  = (q8r > 17'sd127) ? 8'sd127 : (q8r < -17'sd127) ? -8'sd127 : q8r[7:0];

    // ---- NSTAT: sequential divider, integer square root
    reg  [3:0]  ns;
    reg  [63:0] dn, dd, dq;
    reg  [64:0] dr;
    reg  [6:0]  dc;
    reg         drun;
    wire [64:0] dr_sh = {dr[63:0], dn[63]};
    wire        dr_ge = (dr_sh >= {1'b0, dd});
    wire signed [63:0] ns_num = (s_acc <<< 1) + $signed({32'd0, r_p0});
    wire        ns_neg = ns_num[63];
    reg         neg_r;
    reg  signed [63:0] t1_r, t2_r;
    reg  [63:0] ssv, var_r, iq_n, iq_res, iq_bit;
    reg  [5:0]  iq_c;
    reg  signed [7:0] s_r;
    wire signed [63:0] meanx = {{32{n_mean[31]}}, n_mean};
    integer bi;
    reg  [6:0]  bl;
    always @* begin
        bl = 7'd0;
        for (bi = 0; bi < 64; bi = bi + 1) if (var_r[bi]) bl = bi[6:0] + 7'd1;
    end
    wire signed [7:0] s_c  = ((8'sd61 - $signed({1'b0, bl})) >>> 1) <<< 1;
    wire signed [7:0] s_ng = -s_r;
    wire [63:0] iq_t = iq_res + iq_bit;

    // ---- weight source: the ROPE record (V_REC), NORM records (V_RUN)
    wire [6:0] rleft = 7'd64 - gcnt;
    wire [6:0] rb_space = 7'd32 - {1'b0, rb_cnt};
    always @* begin
        ws_take = 6'd0;
        if (st == V_REC && ws_avail != 6'd0) ws_take = ({1'b0, ws_avail} < rleft) ? ws_avail : rleft[5:0];
        else if (st == V_RUN && op == OP_NORM && fetch_left != 32'd0) begin
            ws_take = (ws_avail > LANES) ? 6'(LANES) : ws_avail;
            if ({1'b0, ws_take} > rb_space) ws_take = rb_space[5:0];
            if ({26'd0, ws_take} > fetch_left) ws_take = fetch_left[5:0];
        end
    end

    // ---- the RAM port: the write of the stage that writes this clock, else a cache fill
    wire w_copy = (op == OP_COPY) && v1;
    wire w_mul  = (op == OP_MUL) && v2;
    wire w_lut  = (op == OP_LUT) && v3;
    wire w_norm = (op == OP_NORM) && v6;
    wire w_sm   = (op == OP_SOFTMAX) && v3;
    wire w_rp1  = (op == OP_ROPE) && v2;
    wire w_rp2  = (op == OP_ROPE) && v3;
    wire w_rcw  = (st == V_RCW);
    reg         wr16, wr32, wr8, rd_go;
    reg  [1:0]  rd_kn;
    reg  [31:0] wr_a, wr_d;
    always @* begin
        wr16 = 1'b0; wr32 = 1'b0; wr8 = 1'b0; wr_a = 32'd0; wr_d = 32'd0;
        if (w_copy) begin
            wr_a = d1;
            if (b8) begin wr8 = 1'b1; wr_d = {24'd0, q8}; end
            else    begin wr16 = 1'b1; wr_d = {16'd0, x0_1}; end
        end
        else if (w_mul)  begin wr16 = 1'b1; wr_a = d2; wr_d = {16'd0, ro1}; end
        else if (w_lut)  begin wr16 = 1'b1; wr_a = d3; wr_d = {16'd0, y_r[15:0]}; end
        else if (w_norm) begin wr16 = 1'b1; wr_a = d6; wr_d = {16'd0, n_o[15:0]}; end
        else if (w_sm)   begin wr32 = 1'b1; wr_a = d3; wr_d = {15'd0, sm_e}; end
        else if (w_rp1)  begin wr16 = 1'b1; wr_a = d2; wr_d = {16'd0, ro1}; end
        else if (w_rp2)  begin wr16 = 1'b1; wr_a = q3; wr_d = {16'd0, ro2_3}; end
        else if (w_rcw)  begin wr32 = 1'b1; wr_a = r_p2 + {r[30:0], 1'b0}; wr_d = dquo[31:0]; end
        m_en = 1'b0; m_we = {NB{1'b0}}; m_addr = 32'd0; m_wdata = {W{1'b0}};
        rd_go = 1'b0; rd_kn = 2'd0;
        if (wr16) begin
            m_en = 1'b1; m_addr = wr_a >> LB; m_wdata = {LANES{wr_d[15:0]}};
            m_we = {{(NB-2){1'b0}}, 2'b11} << {wr_a[LB-1:0], 1'b0};
        end else if (wr8) begin
            m_en = 1'b1; m_addr = wr_a >> (LB + 1); m_wdata = {NB{wr_d[7:0]}};
            m_we = {{(NB-1){1'b0}}, 1'b1} << wr_a[LB:0];
        end else if (wr32) begin
            m_en = 1'b1; m_addr = wr_a >> LB; m_wdata = {(LANES/2){wr_d}};
            m_we = {{(NB-4){1'b0}}, 4'b1111} << {wr_a[LB-1:1], 2'b00};
        end else if (st == V_RUN && !issued_all && (miss0 || miss1 || miss2)) begin
            rd_go = 1'b1; rd_kn = miss0 ? 2'd0 : miss1 ? 2'd1 : 2'd2;
            m_en = 1'b1; m_addr = miss0 ? wa0 : miss1 ? wa1 : wa2;
        end
    end
    // a write updates the cached copies of its word, and the word that arrives this clock
    reg [W-1:0] wmask;
    integer tm;
    always @* for (tm = 0; tm < NB; tm = tm + 1) wmask[8 * tm +: 8] = {8{m_we[tm]}};
    wire wr_any = wr16 | wr8 | wr32;
    wire [W-1:0] c_m0 = (wr_any && m_addr == c_a0) ? ((c_d0 & ~wmask) | (m_wdata & wmask)) : c_d0;
    wire [W-1:0] c_m1 = (wr_any && m_addr == c_a1) ? ((c_d1 & ~wmask) | (m_wdata & wmask)) : c_d1;
    wire [W-1:0] c_m2 = (wr_any && m_addr == c_a2) ? ((c_d2 & ~wmask) | (m_wdata & wmask)) : c_d2;
    wire [W-1:0] c_in = (wr_any && m_addr == rd_a) ? ((m_rdata & ~wmask) | (m_wdata & wmask)) : m_rdata;

    integer t;
    reg [255:0] rb_n;
    reg [5:0]   rc_n;
    always @* begin                                    // NORM record buffer: take a record, append new bytes
        rb_n = (can_issue && norm_rec) ? (rb >> 48) : rb;
        rc_n = (can_issue && norm_rec) ? rb_cnt - 6'd6 : rb_cnt;
        for (t = 0; t < LANES; t = t + 1)
            if (st == V_RUN && t < ws_take) rb_n[8 * (rc_n + t) +: 8] = ws_data[8 * t +: 8];
        if (st == V_RUN) rc_n = rc_n + ws_take;
    end

    integer e;
    always @(posedge gclk or negedge rst_n) begin
        if (!rst_n) begin
            st <= V_IDLE; op <= 3'd0; wide <= 1'b0; gbrow <= 1'b0; auto <= 1'b0; b8 <= 1'b0; pass <= 1'b0;
            issued_all <= 1'b0; r <= 0; i <= 0; sbase <= 0; dbase <= 0; sa <= 0; da <= 0;
            crec <= 512'd0; gcnt <= 7'd0; mx <= 48'sd0; ssum <= 33'd0;
            dcnt <= 6'd0; drem <= 33'd0; dquo <= 45'd0; s_acc <= 64'sd0; q_acc <= 64'd0;
            c_v0 <= 1'b0; c_v1 <= 1'b0; c_v2 <= 1'b0; c_a0 <= 0; c_a1 <= 0; c_a2 <= 0;
            c_d0 <= {W{1'b0}}; c_d1 <= {W{1'b0}}; c_d2 <= {W{1'b0}};
            rd_p <= 1'b0; rd_k <= 2'd0; rd_a <= 32'd0;
            rb <= 256'd0; rb_cnt <= 6'd0; fetch_left <= 32'd0; grec <= 48'd0;
            v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; v4 <= 1'b0; v5 <= 1'b0; v6 <= 1'b0; d6 <= 0; n_q6 <= 0; n_b6 <= 0;
            d1 <= 0; d2 <= 0; d3 <= 0; d4 <= 0; d5 <= 0; q1 <= 0; q2 <= 0; q3 <= 0;
            x0_1 <= 0; x1_1 <= 0; x2_1 <= 0; ix1 <= 0; g1 <= 0; g2 <= 0; g3 <= 0; g4 <= 0; g5 <= 0;
            a_xs <= 0; a_t0 <= 0; a_dt <= 0; a_fr <= 0; a_hi <= 0; a_lo <= 0; y_r <= 0;
            d_r <= 0; p1_r <= 0; yn_r <= 0; p2_r <= 0; sp_r <= 0; tt_r <= 0; ro1 <= 0; ro2 <= 0; ro2_3 <= 0;
            n_mean <= 0; n_rec <= 0; n_sh1 <= 0; ns <= 0; dn <= 0; dd <= 0; dq <= 0; dr <= 0; dc <= 0; drun <= 0;
            neg_r <= 0; t1_r <= 0; t2_r <= 0; ssv <= 0; var_r <= 0; iq_n <= 0; iq_res <= 0; iq_bit <= 0; iq_c <= 0; s_r <= 0;
        end else begin
            // ---- caches: this clock's write, then the word read last clock
            c_d0 <= c_m0; c_d1 <= c_m1; c_d2 <= c_m2;
            if (rd_p) begin
                case (rd_k)
                    2'd0: begin c_d0 <= c_in; c_a0 <= rd_a; c_v0 <= 1'b1; end
                    2'd1: begin c_d1 <= c_in; c_a1 <= rd_a; c_v1 <= 1'b1; end
                    default: begin c_d2 <= c_in; c_a2 <= rd_a; c_v2 <= 1'b1; end
                endcase
            end
            rd_p <= rd_go; rd_k <= rd_kn; rd_a <= m_addr;

            // ---- NORM record buffer
            if (st == V_RUN && op == OP_NORM) begin
                rb <= rb_n; rb_cnt <= rc_n; fetch_left <= fetch_left - {26'd0, ws_take};
            end

            // ---- pipeline
            v1 <= to_pipe; v2 <= v1; v3 <= v2; v4 <= v3; v5 <= v4; v6 <= v5 && (op == OP_NORM);
            if (v5) begin d6 <= d5; n_q6 <= n_r5; n_b6 <= g_bx; end                   // stage 5 -> 6 (NORM)
            d2 <= d1; d3 <= d2; d4 <= d3; d5 <= d4; q2 <= q1; q3 <= q2;
            g2 <= g1; g3 <= g2; g4 <= g3; g5 <= g4;
            if (can_issue) begin
                x0_1 <= e_x0; x1_1 <= e_x1; x2_1 <= e_x2; ix1 <= i[3:0]; q1 <= ea1; g1 <= e_rec;
                d1 <= (op == OP_ROPE) ? ea0 : da;          // ROPE writes in place
                if (op == OP_NORM && gbrow && i == 32'd0) grec <= rb[47:0];
                if (op == OP_SOFTMAX && !pass) begin
                    if (i == 32'd0 || e_s48 > mx) mx <= e_s48;
                    if (i == 32'd0) ssum <= 33'd0;
                end
            end
            if (v1) begin                                  // stage 1
                if (op == OP_STAT) begin s_acc <= s_acc + x64; q_acc <= q_acc + x64 * x64; end
                a_xs <= l_xs; a_t0 <= lut_rom(l_tab, l_idx); a_dt <= lut_rom(l_tab, l_idx + 10'd1) - lut_rom(l_tab, l_idx);
                a_fr <= l_fr[10:0]; a_hi <= (l_xs >= l_lim); a_lo <= (l_xs < -l_lim);
                d_r <= n_d;
                ro1 <= (op == OP_MUL) ? m_z[15:0] : r_z1[15:0];
                ro2 <= r_z2[15:0];
                sp_r <= sm_d * r_p0[15:0];
            end
            if (v2) begin                                  // stage 2
                y_r <= l_y; p1_r <= n_p1[63:0]; tt_r <= sm_tt; ro2_3 <= ro2;
            end
            if (v3) begin                                  // stage 3
                yn_r <= n_y[31:0];
                if (op == OP_LUT && l_acc) begin s_acc <= s_acc + y_rx; q_acc <= q_acc + y_rx * y_rx; end
                if (op == OP_SOFTMAX) ssum <= ssum + {16'd0, sm_e};
            end
            if (v4) p2_r <= n_p2;                          // stage 4

            // ---- order of the elements
            if (can_issue) begin
                if (!el_last) begin
                    i <= i + 32'd1; sa <= sa + r_sstr; da <= da + r_dstr;
                end else if (op == OP_SOFTMAX && !pass) begin
                    pass <= 1'b1; i <= 32'd0; sa <= sbase; da <= dbase;
                end else if (op == OP_SOFTMAX || row_last) begin
                    issued_all <= 1'b1;
                end else begin
                    r <= r + 32'd1; i <= 32'd0;
                    sbase <= sbase + r_srow; dbase <= dbase + r_drow; sa <= sbase + r_srow; da <= dbase + r_drow;
                end
            end

            case (st)
                V_IDLE: if (go) begin
                    op <= reg_wdata[2:0]; wide <= reg_wdata[4]; gbrow <= reg_wdata[5]; auto <= reg_wdata[6]; b8 <= reg_wdata[7];
                    if (reg_wdata[3]) begin s_acc <= 64'sd0; q_acc <= 64'd0; end
                    r <= 32'd0; i <= 32'd0; sbase <= r_src; dbase <= r_dst; sa <= r_src; da <= r_dst; pass <= 1'b0;
                    issued_all <= (r_n == 32'd0 || r_rows == 32'd0); ns <= 4'd0; drun <= 1'b0; gcnt <= 7'd0;
                    c_v0 <= 1'b0; c_v1 <= 1'b0; c_v2 <= 1'b0;
                    rb_cnt <= 6'd0;
                    fetch_left <= (reg_wdata[2:0] != OP_NORM) ? 32'd0 : reg_wdata[5] ? 32'd6 * r_rows : 32'd6 * r_n * r_rows;
                    st <= (reg_wdata[2:0] == OP_NSTAT) ? V_NS :
                          (reg_wdata[2:0] == OP_NORM || reg_wdata[2:0] == OP_ROPE) ? V_SEEK : V_RUN;
                end
                V_SEEK: st <= (op == OP_ROPE) ? V_REC : V_RUN;
                V_REC: begin                                   // ROPE: the 64-byte cos/sin record
                    for (t = 0; t < LANES; t = t + 1)
                        if (t < ws_take) crec[8 * (gcnt[5:0] + t[5:0]) +: 8] <= ws_data[8 * t +: 8];
                    gcnt <= gcnt + {1'b0, ws_take};
                    if (ws_avail != 6'd0 && {1'b0, ws_take} == rleft) st <= V_RUN;
                end
                V_RUN: if (issued_all || (can_issue && el_last && (op == OP_SOFTMAX ? pass : row_last))) st <= V_DRAIN;
                V_DRAIN: if (!v1 && !v2 && !v3 && !v4 && !v5 && !v6) begin
                    if (op == OP_SOFTMAX) begin dcnt <= 6'd44; drem <= 33'd0; dquo <= 45'd0; st <= V_DIV; end
                    else st <= V_DONE;
                end
                V_DIV: begin
                    drem <= dv_ge ? dv_sh[32:0] - ssum : dv_sh[32:0];
                    dquo <= {dquo[43:0], dv_ge};
                    dcnt <= dcnt - 6'd1;
                    if (dcnt == 6'd0) st <= V_RCW;
                end
                V_RCW: begin
                    if (row_last) st <= V_DONE;
                    else begin
                        r <= r + 32'd1; i <= 32'd0; pass <= 1'b0; issued_all <= 1'b0;
                        sbase <= sbase + r_srow; dbase <= dbase + r_drow; sa <= sbase + r_srow; da <= dbase + r_drow;
                        st <= V_RUN;
                    end
                end
                V_NS: begin
                    if (drun) begin
                        // one bit of the 64 / 64 restoring divider per clock
                        dr <= dr_ge ? dr_sh - {1'b0, dd} : dr_sh;
                        dq <= {dq[62:0], dr_ge};
                        dn <= {dn[62:0], 1'b0};
                        dc <= dc - 7'd1;
                        if (dc == 7'd0) drun <= 1'b0;
                    end else begin
                        ns <= ns + 4'd1;
                        case (ns)
                            4'd0: begin      // mean = floor((2S + N) / 2N)
                                neg_r <= ns_neg;
                                dn <= ns_neg ? (-ns_num) + {31'd0, r_p0, 1'b0} - 64'd1 : ns_num;
                                dd <= {31'd0, r_p0, 1'b0}; dq <= 64'd0; dr <= 65'd0; dc <= 7'd63; drun <= 1'b1;
                            end
                            4'd1: n_mean <= neg_r ? -dq[31:0] : dq[31:0];
                            4'd2: begin t1_r <= meanx * s_acc; t2_r <= meanx * meanx; end
                            4'd3: ssv <= q_acc - {t1_r[62:0], 1'b0} + {32'd0, r_p0} * t2_r;
                            4'd4: begin dn <= ssv; dd <= {32'd0, r_p0}; dq <= 64'd0; dr <= 65'd0; dc <= 7'd63; drun <= 1'b1; end
                            4'd5: var_r <= (dq + {32'd0, r_p1} == 64'd0) ? 64'd1 : dq + {32'd0, r_p1};
                            4'd6: s_r <= s_c;
                            4'd7: begin
                                iq_n <= s_r[7] ? (var_r >> s_ng[5:0]) : (var_r << s_r[5:0]);
                                iq_res <= 64'd0; iq_bit <= 64'd1 << 60; iq_c <= 6'd31;
                            end
                            4'd8: begin      // bitwise integer square root, 31 steps
                                if (iq_c != 6'd0) begin
                                    ns <= ns;
                                    iq_c <= iq_c - 6'd1;
                                    if (iq_n >= iq_t) begin iq_n <= iq_n - iq_t; iq_res <= (iq_res >> 1) + iq_bit; end
                                    else iq_res <= iq_res >> 1;
                                    iq_bit <= iq_bit >> 2;
                                end
                            end
                            4'd9: begin dn <= 64'd1 << 61; dd <= iq_res; dq <= 64'd0; dr <= 65'd0; dc <= 7'd63; drun <= 1'b1; end
                            default: begin
                                n_rec <= dq[32:0];
                                n_sh1 <= 6'd49 - s_r[6:1];
                                st <= V_DONE;
                            end
                        endcase
                    end
                end
                V_DONE: st <= V_IDLE;
                default: st <= V_IDLE;
            endcase
        end
    end
endmodule
