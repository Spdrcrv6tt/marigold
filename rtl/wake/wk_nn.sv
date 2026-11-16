// Wake spotter phone network (docs/golden-wake.md, section NN; golden sw/wake/golden/golden.py net()).
// Parametric: LANES MAC lanes (one output channel each), C channels, NOUT output channels padded to a multiple of
// LANES. Twelve layers: pointwise 64->C, five depthwise k=5 + pointwise C->C blocks (dilations 1,1,2,2,2), output
// C->NOUT. 4-bit weights; the ROM word is 4*LANES bits, read one per clock in the image's order: per layer a shift
// word, per group of LANES output channels 8 bias words (lane j's int32 at bit 32j of the 8-word concatenation),
// one weight word per input (or tap, lane j's weight at bit 4j), 4 multiplier words (lane j's M at bit 16j of the
// 4-word concatenation). Two-stage pipeline: stage 1 pops a ROM word and issues the activation read, stage 2 does
// MAC / requantization (LANES/4 requantization multipliers, so the 4 multiplier words finish a group). The ROM port
// has request/grant and a 4-word FIFO, so grant stalls only add bubbles. Depthwise histories live in one RAM: taps
// before the run's first frame read as zero (`fresh` with start begins a run). Padded output channels are dropped.
module wk_nn #(
    parameter LANES = 16,
    parameter C     = 240,
    parameter NOUT  = 48,
    parameter XW    = 10            // feature buffer address bits (slots x 64 bytes)
) (
    input  wire               clk,
    input  wire               clk_x,         // free-running clock: feature buffer and logit trace
    input  wire               rst_n,
    input  wire               clr,
    input  wire               start,
    input  wire               fresh,         // with start: first frame of a run (zero history)
    input  wire [XW-1:0]      x_base,        // with start: feature buffer address of this frame's 64 bytes
    output reg                busy,
    output reg                done,
    input  wire               x_we,
    input  wire [XW-1:0]      x_addr,
    input  wire [7:0]         x_data,
    output wire               rom_req,
    output reg  [16:0]        rom_addr,
    input  wire               rom_gnt,
    input  wire               rom_rvalid,
    input  wire [4*LANES-1:0] rom_rdata,
    input  wire [5:0]         z_idx,
    output wire signed [15:0] z_val,
    output reg  signed [15:0] zmax,
    output wire               dbg_v,
    output wire [21:0]        dbg_d
);
    localparam integer W    = 4 * LANES;
    localparam integer NG   = C / LANES;
    localparam integer NGO  = NOUT / LANES;
    localparam integer LL   = $clog2(LANES);
    localparam integer NQ   = LANES / 4;
    localparam integer HW   = NG * 37;
    localparam integer HA   = $clog2(HW);
    localparam integer VA   = (NG > 1) ? $clog2(NG) : 1;
    localparam integer TOTAL = (1 + NG * (12 + 64)) + 5 * (1 + NG * (12 + 5)) + 5 * (1 + NG * (12 + C)) + (1 + NGO * (12 + C));

    function automatic [1:0] kind_of(input [3:0] l);      // 0 pw from x, 1 dw, 2 pw, 3 out
        kind_of = (l == 0) ? 2'd0 : (l == 11) ? 2'd3 : l[0] ? 2'd1 : 2'd2;
    endfunction
    function automatic [9:0] nw_of(input [3:0] l);
        nw_of = (l == 0) ? 10'd64 : (l == 11) ? 10'(C) : l[0] ? 10'd5 : 10'(C);
    endfunction
    function automatic [7:0] ng_of(input [3:0] l);
        ng_of = (l == 11) ? 8'(NGO) : 8'(NG);
    endfunction
    function automatic [1:0] dil_of(input [2:0] jj);  dil_of = (jj < 2) ? 2'd1 : 2'd2; endfunction
    function automatic [3:0] rs_of(input [2:0] jj);   rs_of = (jj < 2) ? 4'd5 : 4'd9; endfunction
    function automatic [HA-1:0] hb_of(input [2:0] jj);
        case (jj) 3'd0: hb_of = 0; 3'd1: hb_of = HA'(NG * 5); 3'd2: hb_of = HA'(NG * 10); 3'd3: hb_of = HA'(NG * 19); default: hb_of = HA'(NG * 28); endcase
    endfunction

    // ---- memories
    wire              xr_re;  wire [XW-1:0] xr_a;  wire [7:0] xr_d;
    wire              hr_re;  wire [HA-1:0] hr_a;  wire [8*LANES-1:0] hr_d;
    wire              vr_re;  wire [VA-1:0] vr_a;  wire [8*LANES-1:0] vr_d;
    wire              or_re;  wire [VA-1:0] or_a;  wire [8*LANES-1:0] or_d;
    reg               h_we, v_we, o_we;
    reg  [HA-1:0]     h_wa;   reg [VA-1:0] v_wa, o_wa;
    reg  [8*LANES-1:0] w_wd;
    wk_ram #(.WORDS(1 << XW), .WIDTH(8))        u_x (.clk(clk_x), .we(x_we), .waddr(x_addr), .wdata(x_data), .re(xr_re), .raddr(xr_a), .rdata(xr_d));
    wk_ram #(.WORDS(1 << HA), .WIDTH(8*LANES))  u_h (.clk(clk), .we(h_we), .waddr(h_wa), .wdata(w_wd), .re(hr_re), .raddr(hr_a), .rdata(hr_d));
    wk_ram #(.WORDS(1 << VA), .WIDTH(8*LANES))  u_v (.clk(clk), .we(v_we), .waddr(v_wa), .wdata(w_wd), .re(vr_re), .raddr(vr_a), .rdata(vr_d));
    wk_ram #(.WORDS(1 << VA), .WIDTH(8*LANES))  u_o (.clk(clk), .we(o_we), .waddr(o_wa), .wdata(w_wd), .re(or_re), .raddr(or_a), .rdata(or_d));
    reg signed [15:0] zb [0:39];
    assign z_val = zb[z_idx];

    // ---- ROM FIFO
    reg [W-1:0] fq [0:3];
    reg [1:0]   fh, ft;
    reg [2:0]   fc;
    reg         run;
    reg [16:0]  issued;
    assign rom_req = run && (issued < TOTAL) && (fc <= 3'd1);
    wire pop;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fh <= 0; ft <= 0; fc <= 0; issued <= 0; rom_addr <= 0;
        end else if (clr || start) begin
            fh <= 0; ft <= 0; fc <= 0; issued <= 0; rom_addr <= 0;
        end else begin
            if (rom_gnt) begin issued <= issued + 1; rom_addr <= rom_addr + 1; end
            if (rom_rvalid) begin fq[ft] <= rom_rdata; ft <= ft + 1; end
            fc <= fc + (rom_rvalid ? 3'd1 : 3'd0) - (pop ? 3'd1 : 3'd0);
            if (pop) fh <= fh + 1;
        end
    end
    wire [W-1:0] word = fq[fh];

    // ---- stage 1: sequencing
    reg [3:0]  l;
    reg [1:0]  ph;          // 0 shift, 1 bias, 2 weights, 3 multipliers
    reg [9:0]  cnt;
    reg [7:0]  g;
    reg        s1_on;
    reg [3:0]  wp [0:4];
    reg [3:0]  tcnt;        // frames since the run's first frame, saturating at 8
    reg [XW-1:0] xb;
    wire [1:0] kd = kind_of(l);
    wire [2:0] jd = 3'((l - 4'd1) >> 1);
    assign pop = s1_on && (fc != 0);
    wire [4:0] back = 5'((5'd4 - 5'(cnt[2:0])) * 5'(dil_of(jd)));
    wire [4:0] sl0  = {1'b0, wp[jd]} + {1'b0, rs_of(jd)} - back;
    wire [3:0] slot = (sl0 >= {1'b0, rs_of(jd)}) ? 4'(sl0 - {1'b0, rs_of(jd)}) : sl0[3:0];
    wire       tz   = back > {1'b0, tcnt};
    wire is_w = (ph == 2'd2);
    assign xr_re = pop && is_w && (kd == 2'd0);
    assign xr_a  = xb + XW'(cnt[5:0]);
    assign hr_re = pop && is_w && (kd == 2'd1);
    assign hr_a  = hb_of(jd) + HA'(slot) * HA'(NG) + HA'(g);
    assign vr_re = pop && is_w && (kd == 2'd2);
    assign vr_a  = VA'(cnt >> LL);
    assign or_re = pop && is_w && (kd == 2'd3);
    assign or_a  = VA'(cnt >> LL);

    // ---- stage 2
    reg        s2_v;
    reg [1:0]  s2_ph, s2_kd;
    reg [9:0]  s2_cnt;
    reg [7:0]  s2_g;
    reg [3:0]  s2_l;
    reg        s2_tz, s2_last;
    reg [W-1:0] s2_w;
    reg [5:0]  shf;
    reg signed [31:0] acc [0:LANES-1];
    reg [7*W-1:0]  breg;               // bias words 0..6
    reg [8*LANES-1:0] ybuf;

    function automatic signed [8:0] act(input [1:0] k, input [9:0] c, input integer lane, input z,
                                        input [7:0] xd, input [8*LANES-1:0] hd, input [8*LANES-1:0] vd, input [8*LANES-1:0] od);
        reg [7:0] b;
        begin
            case (k)
                2'd0: act = $signed({xd[7], xd});
                2'd1: begin b = hd[8*lane +: 8]; act = z ? 9'sd0 : $signed({1'b0, b}); end
                2'd2: begin b = vd[8*(c % LANES) +: 8]; act = $signed({1'b0, b}); end
                default: begin b = od[8*(c % LANES) +: 8]; act = $signed({1'b0, b}); end
            endcase
        end
    endfunction

    wire signed [12:0] pr [0:LANES-1];
    wire [8*W-1:0] bcat = {s2_w, breg};
    genvar gq;
    generate for (gq = 0; gq < LANES; gq = gq + 1) begin : g_mac
        wire signed [8:0]  aq   = act(s2_kd, s2_cnt, gq, s2_tz, xr_d, hr_d, vr_d, or_d);
        wire signed [12:0] wq13 = {{9{s2_w[4*gq+3]}}, s2_w[4*gq +: 4]};
        wire signed [12:0] aq13 = {{4{aq[8]}}, aq};
        assign pr[gq] = wq13 * aq13;
    end endgenerate
    // requantization: multiplier word m (= s2_cnt[1:0]) carries lanes m*NQ .. m*NQ + NQ - 1
    wire signed [48:0] rq [0:NQ-1];
    wire [7:0]         yb [0:NQ-1];
    wire signed [15:0] yz [0:NQ-1];
    generate for (gq = 0; gq < NQ; gq = gq + 1) begin : g_rq
        wire [7:0] ln = 8'(s2_cnt[1:0]) * 8'(NQ) + 8'(gq);
        wire signed [31:0] al = acc[ln];
        wire signed [48:0] prod = al * $signed({1'b0, s2_w[16*gq +: 16]});
        assign rq[gq] = (prod + (49'sd1 <<< (shf - 6'd1))) >>> shf;
        assign yb[gq] = (rq[gq] < 0) ? 8'd0 : (rq[gq] > 255) ? 8'd255 : rq[gq][7:0];
        assign yz[gq] = (rq[gq] < -49'sd32768) ? -16'sd32768 : (rq[gq] > 49'sd32767) ? 16'sd32767 : rq[gq][15:0];
    end endgenerate
    reg [8*LANES-1:0] yfull;
    integer q, qq;
    always @* begin
        yfull = ybuf;
        for (qq = 0; qq < NQ; qq = qq + 1) yfull[8*(3*NQ + qq) +: 8] = yb[qq];
    end
    reg signed [15:0] zm;
    always @* begin
        zm = zmax;
        for (qq = 0; qq < NQ; qq = qq + 1)
            if (LANES * s2_g + NQ * s2_cnt[1:0] + qq < 40 && yz[qq] > zm) zm = yz[qq];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            l <= 0; ph <= 0; cnt <= 0; g <= 0; s1_on <= 0; run <= 0; busy <= 0; done <= 0; tcnt <= 0; xb <= 0;
            for (q = 0; q < 5; q = q + 1) wp[q] <= 0;
            s2_v <= 0; s2_ph <= 0; s2_kd <= 0; s2_cnt <= 0; s2_g <= 0; s2_l <= 0; s2_tz <= 0; s2_w <= 0; s2_last <= 0;
            shf <= 1; breg <= 0; ybuf <= 0; h_we <= 0; v_we <= 0; o_we <= 0; h_wa <= 0; v_wa <= 0; o_wa <= 0; w_wd <= 0;
            zmax <= -16'sd32768;
            for (q = 0; q < LANES; q = q + 1) acc[q] <= 0;
        end else begin
            done <= 0; h_we <= 0; v_we <= 0; o_we <= 0;
            if (clr) begin
                l <= 0; ph <= 0; cnt <= 0; g <= 0; s1_on <= 0; run <= 0; busy <= 0; s2_v <= 0; tcnt <= 0;
                for (q = 0; q < 5; q = q + 1) wp[q] <= 0;
            end else begin
                if (start) begin
                    l <= 0; ph <= 0; cnt <= 0; g <= 0; s1_on <= 1; run <= 1; busy <= 1; zmax <= -16'sd32768; xb <= x_base;
                    if (fresh) tcnt <= 0;
                end
                // ---- stage 1
                s2_v <= pop;
                if (pop) begin
                    s2_ph <= ph; s2_kd <= kd; s2_cnt <= cnt; s2_g <= g; s2_l <= l; s2_tz <= tz; s2_w <= word;
                    s2_last <= 1'b0;
                    case (ph)
                        2'd0: begin ph <= 2'd1; cnt <= 0; end
                        2'd1: if (cnt == 10'd7) begin ph <= 2'd2; cnt <= 0; end else cnt <= cnt + 1;
                        2'd2: if (cnt == nw_of(l) - 10'd1) begin ph <= 2'd3; cnt <= 0; end else cnt <= cnt + 1;
                        default: if (cnt == 10'd3) begin
                            cnt <= 0;
                            if (g == ng_of(l) - 8'd1) begin
                                g <= 0; ph <= 2'd0;
                                if (l == 4'd11) begin s1_on <= 0; s2_last <= 1'b1; end
                                else l <= l + 1;
                            end else begin g <= g + 1; ph <= 2'd1; end
                        end else cnt <= cnt + 1;
                    endcase
                end
                // ---- stage 2
                if (s2_v) begin
                    case (s2_ph)
                        2'd0: shf <= s2_w[5:0];
                        2'd1: if (s2_cnt[2:0] != 3'd7) breg[W*s2_cnt[2:0] +: W] <= s2_w;
                              else for (q = 0; q < LANES; q = q + 1) acc[q] <= bcat[32*q +: 32];
                        2'd2: for (q = 0; q < LANES; q = q + 1)
                                  acc[q] <= acc[q] + {{19{pr[q][12]}}, pr[q]};
                        default: begin
                            if (s2_kd == 2'd3) begin
                                for (qq = 0; qq < NQ; qq = qq + 1)
                                    if (LANES * s2_g + NQ * s2_cnt[1:0] + qq < 40)
                                        zb[6'(LANES * s2_g + NQ * s2_cnt[1:0] + qq)] <= yz[qq];
                                zmax <= zm;
                            end else if (s2_cnt[1:0] != 2'd3) begin
                                for (qq = 0; qq < NQ; qq = qq + 1) ybuf[8*(NQ * s2_cnt[1:0] + qq) +: 8] <= yb[qq];
                            end else begin
                                w_wd <= yfull;
                                if (s2_kd == 2'd1)       begin v_we <= 1; v_wa <= VA'(s2_g); end
                                else if (s2_l == 4'd10)  begin o_we <= 1; o_wa <= VA'(s2_g); end
                                else begin
                                    h_we <= 1;
                                    h_wa <= hb_of(3'(s2_l >> 1)) + HA'(wp[3'(s2_l >> 1)]) * HA'(NG) + HA'(s2_g);
                                end
                            end
                            if (s2_last && s2_cnt[1:0] == 2'd3) begin
                                run <= 0; busy <= 0; done <= 1;
                                tcnt <= (tcnt == 4'd8) ? tcnt : tcnt + 1;
                                for (q = 0; q < 5; q = q + 1) wp[q] <= (wp[q] == rs_of(q[2:0]) - 4'd1) ? 4'd0 : wp[q] + 1;
                            end
                        end
                    endcase
                end
            end
        end
    end

    // ---- logit trace (testbench only): after done, logits 0..39 one per clk_x clock
    reg [5:0] tr_i;
    reg       tr_on;
    always @(posedge clk_x or negedge rst_n) begin
        if (!rst_n) begin tr_i <= 0; tr_on <= 0; end
        else if (done) begin tr_on <= 1; tr_i <= 0; end
        else if (tr_on) begin
            if (tr_i == 6'd39) tr_on <= 0;
            tr_i <= tr_i + 1;
        end
    end
    assign dbg_v = tr_on;
    assign dbg_d = {tr_i, zb[tr_i]};
endmodule
