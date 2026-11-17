// Matcher (docs/config-blob.md "Matching rule", docs/architecture.md sections 8.2, 8.3, 8.7), island clock domain.
// q (D int8, written by the core into a 64 x 32 latch RAM) and the 8 raw slot values (registers) in; reads the
// vector table and the vectors from config SRAM, one int8 MAC per granted read:
//   s_i = bias_i + sum_d q[d] v_i[d] (int32); b = argmax (lowest index on ties). The best score of a different
//   action, s_2, is kept on the fly: when the leader changes to another action the old leader becomes s_2, and a
//   vector of another action that does not lead raises s_2 (that is max{s_j : action[j] != action[b]} at the end).
// Reject (NOMATCH, done at once) if action[b] == 0, s_b < reject_thr, s_b - s_2 < margin (only if some other action
// exists), or action[b] has no action-table entry. Accept (COMMAND): each slot of the action takes its raw value
// (written by the core before go, one per config slot): level codes -1 low, -2 medium, -3 high, -4 max, -5 min are
// mapped with the slot entry's level bytes 1..3 (0 = default: low = min, medium = floor((min + max) / 2), high = max;
// max and min map to the slot's max and min), then clamped as min(max(v, min), max); a slot at or above n_slots gets
// 0; the other slots are zeroed. A raw value of 0x80000000 (INT32_MIN: nothing heard) is never clamped: the slot
// leaves slot_mask and reads 0, so the frame and the GPIO binary rules leave it out (R16, John's ruling of Nov 17,
// 2026). confidence = clamp(s_b >>> conf_sh, 0, 255), conf_sh from the blob.
module mg_match (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        go,
    input  wire        abort,             // CTRL begin (config reload): stop at once, no result (review R6)
    input  wire [15:0] D,
    input  wire [15:0] N,
    input  wire [7:0]  n_actions,
    input  wire [7:0]  n_slots,
    input  wire [14:0] vec_base,
    input  wire [14:0] act_base,
    input  wire [14:0] slot_base,
    input  wire [31:0] reject_thr,
    input  wire [31:0] margin,
    input  wire [7:0]  conf_sh,
    input  wire        q_we,
    input  wire [5:0]  q_waddr,
    input  wire [31:0] q_wdata,
    input  wire        v_we,
    input  wire [2:0]  v_waddr,
    input  wire [31:0] v_wdata,
    output reg         req,
    output reg  [14:0] addr,
    input  wire        gnt,
    input  wire        rvalid,
    input  wire [31:0] rdata,
    output wire        busy,
    output reg         done,              // the result below is valid (until the next go)
    output reg         accept,            // 1 COMMAND, 0 NOMATCH
    output reg  [7:0]  action,            // action[b] when accepted, else 0
    output reg  [15:0] best,              // b
    output wire [7:0]  best_action,       // action[b] (valid from the decision on)
    output reg  [31:0] sb,
    output reg  [31:0] s2,
    output reg         has_s2,
    output wire [7:0]  conf,
    output reg  [7:0]  slot_mask,         // the accepted action's slots that carry a value, else 0
    output wire [255:0] vals              // clamped values, slot k at bits 32k
);
    localparam S_IDLE = 3'd0, S_SCORE = 3'd1, S_DEC = 3'd2, S_ACT = 3'd3, S_SLOT = 3'd4;
    localparam [31:0] NO_VALUE = 32'h80000000;                     // raw value: nothing heard for this slot (R16)
    reg  [2:0]  st;
    assign busy = (st != S_IDLE);
    assign best_action = a1;

    reg  [31:0] val [0:7];
    genvar gv;
    generate for (gv = 0; gv < 8; gv = gv + 1) begin : g_val
        assign vals[32*gv +: 32] = val[gv];
    end endgenerate

    // ---- scoring: address generator (issue) and accumulator (return, one clock after each grant)
    reg  [15:0] ii;            // vector being issued
    reg  [8:0]  di;            // 0 = its table entry, 1..D = byte di - 1
    reg  [14:0] vrow;          // vec_base + ii * D
    reg         iss_done;
    reg         t_ent, t_last; // tag of the read in flight
    reg  [1:0]  t_lane, t_ql;  // byte lane of v in the config word, of q[d] in the q word
    reg  [15:0] t_idx;
    reg  [7:0]  ca;            // action of the vector being accumulated
    reg  signed [31:0] acc;
    reg  [7:0]  a1;
    reg  [15:0] ri;

    wire [31:0] qw;
    wire [7:0]  dsel = di[7:0] - 8'd1;  // byte index d of the issued byte read
    latch_ram #(.WORDS(64), .WIDTH(32), .FLOPS(0)) u_q (
        .clk(clk), .we(q_we), .waddr(q_waddr), .wdata(q_wdata),
        .re(st == S_SCORE && gnt && di != 9'd0), .raddr(dsel[7:2]), .rdata(qw));

    wire [14:0] ent_a  = 15'h20 + {ii[12:0], 2'b00};
    wire [7:0]  vb  = t_lane == 2'd0 ? rdata[7:0] : t_lane == 2'd1 ? rdata[15:8] : t_lane == 2'd2 ? rdata[23:16] : rdata[31:24];
    wire [7:0]  qb  = t_ql == 2'd0 ? qw[7:0] : t_ql == 2'd1 ? qw[15:8] : t_ql == 2'd2 ? qw[23:16] : qw[31:24];
    wire signed [15:0] prod = $signed(qb) * $signed(vb);
    wire signed [31:0] acc_n = acc + {{16{prod[15]}}, prod};
    wire        lead = (acc_n > $signed(sb)) || (t_idx == 16'd0);    // strictly greater: lowest index wins ties

    // ---- decision, action lookup, clamping
    wire signed [31:0] dif = $signed(sb) - $signed(s2);
    wire        rej = (a1 == 8'd0) || ($signed(sb) < $signed(reject_thr)) || (has_s2 && (dif < $signed(margin)));
    reg  [7:0]  j;             // action-table entry / slot index
    reg  [2:0]  ph;            // step within the entry or slot
    reg  signed [31:0] smin;
    reg  [23:0] slv;                       // level bytes {high, medium, low} of the slot entry
    // byte fetch: fn bytes (1..4) little endian from byte address addr into fv; the tables after the vectors need not
    // be word aligned (N x D bytes of vectors), so they are read a byte at a time
    reg         fbusy;
    reg  [2:0]  fn, fk;
    reg  [1:0]  fl;
    reg  [31:0] fv;
    wire [7:0]  fbyte = fl == 2'd0 ? rdata[7:0] : fl == 2'd1 ? rdata[15:8] : fl == 2'd2 ? rdata[23:16] : rdata[31:24];
    wire [31:0] fv_n = fv | ({24'd0, fbyte} << {fk, 3'b000});
    wire        fdone = fbusy && rvalid && (fk + 3'd1 == fn);
    wire signed [31:0] sv = $signed(val[j[2:0]]);
    wire signed [32:0] smid = ($signed({smin[31], smin}) + $signed({fv_n[31], fv_n})) >>> 1;   // floor((min + max) / 2)
    wire signed [31:0] lvm = (sv == -32'sd1) ? ((slv[7:0] != 8'd0) ? {24'd0, slv[7:0]} : smin) :
                             (sv == -32'sd2) ? ((slv[15:8] != 8'd0) ? {24'd0, slv[15:8]} : smid[31:0]) :
                             (sv == -32'sd3) ? ((slv[23:16] != 8'd0) ? {24'd0, slv[23:16]} : $signed(fv_n)) :
                             (sv == -32'sd4) ? $signed(fv_n) : (sv == -32'sd5) ? smin : sv;
    wire signed [31:0] vlo = (lvm < smin) ? smin : lvm;                      // min(max(v, min), max), as the model

    wire signed [31:0] csh = $signed(sb) >>> conf_sh;
    assign conf = (csh < 0) ? 8'd0 : (csh > 255) ? 8'd255 : csh[7:0];

    wire gen = go || busy || v_we || abort;
    wire gclk;
    clk_gate u_cg (.clk(clk), .en(gen), .gclk(gclk));

    integer k;
    always @(posedge gclk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; req <= 1'b0; addr <= 15'd0; done <= 1'b0; accept <= 1'b0; action <= 8'd0; best <= 16'd0;
            sb <= 32'd0; s2 <= 32'd0; has_s2 <= 1'b0; slot_mask <= 8'd0; ii <= 16'd0; di <= 9'd0; vrow <= 15'd0;
            iss_done <= 1'b0; t_ent <= 1'b0; t_last <= 1'b0; t_lane <= 2'd0; t_ql <= 2'd0; t_idx <= 16'd0; ca <= 8'd0; acc <= 32'd0;
            a1 <= 8'd0; ri <= 16'd0; j <= 8'd0; ph <= 3'd0; smin <= 32'd0;
            slv <= 24'd0; fbusy <= 1'b0; fn <= 3'd0; fk <= 3'd0; fl <= 2'd0; fv <= 32'd0;
            for (k = 0; k < 8; k = k + 1) val[k] <= 32'd0;
        end else begin
            if (v_we && !busy) val[v_waddr] <= v_wdata;
            if (abort) begin st <= S_IDLE; req <= 1'b0; fbusy <= 1'b0; done <= 1'b0; end
            else begin
                // byte fetch engine (S_ACT, S_SLOT); before the state machine so a new fetch started there wins
                if (fbusy && st != S_SCORE) begin
                    if (gnt) begin req <= 1'b0; fl <= addr[1:0]; end
                    if (rvalid) begin
                        fv <= fv_n; fk <= fk + 3'd1;
                        if (fk + 3'd1 == fn) fbusy <= 1'b0;
                        else begin req <= 1'b1; addr <= addr + 15'd1; end
                    end
                end
                case (st)
                S_IDLE: if (go) begin
                    st <= S_SCORE; done <= 1'b0; ii <= 16'd0; di <= 9'd0; vrow <= vec_base; iss_done <= 1'b0;
                    has_s2 <= 1'b0; ri <= 16'd0;
                    req <= 1'b1; addr <= 15'h20;
                end
                S_SCORE: begin
                    // issue
                    if (gnt) begin
                        t_ent <= (di == 9'd0); t_lane <= addr[1:0]; t_ql <= dsel[1:0]; t_idx <= ii; t_last <= (di == D[8:0]);
                        if (di == D[8:0]) begin
                            di <= 9'd0; vrow <= vrow + D[14:0];
                            if (ii == N - 16'd1) begin iss_done <= 1'b1; req <= 1'b0; end
                            else begin ii <= ii + 16'd1; addr <= ent_a + 15'd4; end
                        end else begin
                            di <= di + 9'd1;
                            addr <= vrow + {6'd0, di[7:0]};      // byte di of the row (next issue: byte index di)
                        end
                    end
                    // return
                    if (rvalid) begin
                        if (t_ent) begin
                            ca <= rdata[7:0];
                            acc <= {{16{rdata[31]}}, rdata[31:16]};
                        end else begin
                            acc <= acc_n;
                            if (t_last) begin
                                if (lead) begin
                                    if (t_idx != 16'd0 && ca != a1) begin s2 <= sb; has_s2 <= 1'b1; end
                                    sb <= acc_n; a1 <= ca; best <= t_idx;
                                end else if (ca != a1 && (!has_s2 || acc_n > $signed(s2))) begin
                                    s2 <= acc_n; has_s2 <= 1'b1;
                                end
                                if (t_idx == N - 16'd1) st <= S_DEC;
                            end
                        end
                    end
                end
                S_DEC: begin
                    if (rej) begin
                        st <= S_IDLE; done <= 1'b1; accept <= 1'b0; action <= 8'd0; slot_mask <= 8'd0;
                    end else begin
                        st <= S_ACT; j <= 8'd0; ph <= 3'd0;
                    end
                end
                S_ACT: begin                                            // find the entry with id == a1
                    if (ph == 3'd0) begin
                        if (j == n_actions) begin                       // none: NOMATCH
                            st <= S_IDLE; done <= 1'b1; accept <= 1'b0; action <= 8'd0; slot_mask <= 8'd0;
                        end else begin
                            fbusy <= 1'b1; fn <= 3'd2; fk <= 3'd0; fv <= 32'd0;
                            req <= 1'b1; addr <= act_base + {5'd0, j, 2'b00}; ph <= 3'd1;
                        end
                    end else if (fdone) begin
                        if (fv_n[7:0] == a1) begin slot_mask <= fv_n[15:8]; st <= S_SLOT; j <= 8'd0; end
                        else j <= j + 8'd1;
                        ph <= 3'd0;
                    end
                end
                S_SLOT: begin                                           // clamp each slot of the action
                    if (j == 8'd8) begin
                        st <= S_IDLE; done <= 1'b1; accept <= 1'b1; action <= a1;
                    end else if (!slot_mask[j[2:0]] || j >= n_slots) begin
                        val[j[2:0]] <= 32'd0; j <= j + 8'd1; ph <= 3'd0;
                    end else if (val[j[2:0]] == NO_VALUE) begin          // not said: out of the mask, not clamped
                        slot_mask[j[2:0]] <= 1'b0; val[j[2:0]] <= 32'd0; j <= j + 8'd1; ph <= 3'd0;
                    end else case (ph)
                        3'd0: begin                                     // level bytes 1..3
                            fbusy <= 1'b1; fn <= 3'd3; fk <= 3'd0; fv <= 32'd0;
                            req <= 1'b1; addr <= slot_base + j * 12 + 15'd1; ph <= 3'd1;
                        end
                        3'd1: if (fdone) begin                          // min
                            slv <= fv_n[23:0];
                            fbusy <= 1'b1; fn <= 3'd4; fk <= 3'd0; fv <= 32'd0;
                            req <= 1'b1; addr <= slot_base + j * 12 + 15'd4; ph <= 3'd2;
                        end
                        3'd2: if (fdone) begin                          // max
                            smin <= $signed(fv_n);
                            fbusy <= 1'b1; fn <= 3'd4; fk <= 3'd0; fv <= 32'd0;
                            req <= 1'b1; addr <= slot_base + j * 12 + 15'd8; ph <= 3'd3;
                        end
                        default: if (fdone) begin
                            val[j[2:0]] <= (vlo > $signed(fv_n)) ? fv_n : vlo;
                            j <= j + 8'd1; ph <= 3'd0;
                        end
                    endcase
                end
                default: st <= S_IDLE;
                endcase
            end
        end
    end
endmodule
