// Wake spotter phone network (docs/golden-wake.md, section NN; golden sw/wake/golden/golden.py net()).
// Twelve layers (pointwise 64->96, five depthwise k=5 + pointwise 96->96 blocks with dilations 1,1,2,2,2, output
// 96->40), 4-bit weights, 4 MAC lanes (one output channel each), one 16-bit ROM word per clock in the order the
// ROM image stores them: per layer a shift word, per group of 4 output channels 8 bias words, the weight words,
// 4 multiplier words. Two-stage pipeline: stage 1 pops a ROM word and issues the activation read, stage 2 does
// the MAC / requantization. The ROM port has request/grant and a 4-word FIFO, so grant stalls only add bubbles.
// Depthwise histories live in one RAM (888 x 32b): taps before the session's first frame read as zero.
module wk_nn (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               clr,
    input  wire               start,
    output reg                busy,
    output reg                done,
    input  wire               x_we,
    input  wire [5:0]         x_addr,
    input  wire [7:0]         x_data,
    output wire               rom_req,
    output reg  [14:0]        rom_addr,
    input  wire               rom_gnt,
    input  wire               rom_rvalid,
    input  wire [15:0]        rom_rdata,
    input  wire [5:0]         z_idx,
    output wire signed [15:0] z_val,
    output reg  signed [15:0] zmax,
    output reg                dbg_v,
    output reg  [21:0]        dbg_d
);
    localparam integer NL = 12;
    localparam integer TOTAL = 17916;
    // per layer: kind (0 pw from x, 1 dw, 2 pw, 3 out), groups, weight words per group
    function automatic [1:0] kind_of(input [3:0] l);
        kind_of = (l == 0) ? 2'd0 : (l == 11) ? 2'd3 : l[0] ? 2'd1 : 2'd2;
    endfunction
    function automatic [6:0] nw_of(input [3:0] l);
        nw_of = (l == 0) ? 7'd64 : (l == 11) ? 7'd96 : l[0] ? 7'd5 : 7'd96;
    endfunction
    function automatic [4:0] ng_of(input [3:0] l);
        ng_of = (l == 11) ? 5'd10 : 5'd24;
    endfunction
    // depthwise history j = 0..4: dilation, ring size, base word
    function automatic [1:0] dil_of(input [2:0] jj);  dil_of = (jj < 2) ? 2'd1 : 2'd2; endfunction
    function automatic [3:0] rs_of(input [2:0] jj);   rs_of = (jj < 2) ? 4'd5 : 4'd9; endfunction
    function automatic [9:0] hb_of(input [2:0] jj);
        case (jj) 3'd0: hb_of = 0; 3'd1: hb_of = 120; 3'd2: hb_of = 240; 3'd3: hb_of = 456; default: hb_of = 672; endcase
    endfunction

    // ---- memories
    reg  x_re_r;
    wire        xr_re;  wire [5:0] xr_a;  wire [7:0]  xr_d;
    wire        hr_re;  wire [9:0] hr_a;  wire [31:0] hr_d;
    wire        vr_re;  wire [4:0] vr_a;  wire [31:0] vr_d;
    wire        or_re;  wire [4:0] or_a;  wire [31:0] or_d;
    reg         h_we, v_we, o_we;
    reg  [9:0]  h_wa;   reg [4:0] v_wa, o_wa;
    reg  [31:0] w_wd;
    wk_ram #(.WORDS(64),  .WIDTH(8))  u_x (.clk(clk), .we(x_we), .waddr(x_addr), .wdata(x_data), .re(xr_re), .raddr(xr_a), .rdata(xr_d));
    wk_ram #(.WORDS(888), .WIDTH(32)) u_h (.clk(clk), .we(h_we), .waddr(h_wa), .wdata(w_wd), .re(hr_re), .raddr(hr_a), .rdata(hr_d));
    wk_ram #(.WORDS(24),  .WIDTH(32)) u_v (.clk(clk), .we(v_we), .waddr(v_wa), .wdata(w_wd), .re(vr_re), .raddr(vr_a), .rdata(vr_d));
    wk_ram #(.WORDS(24),  .WIDTH(32)) u_o (.clk(clk), .we(o_we), .waddr(o_wa), .wdata(w_wd), .re(or_re), .raddr(or_a), .rdata(or_d));
    reg signed [15:0] zb [0:39];
    assign z_val = zb[z_idx];

    // ---- ROM FIFO
    reg [15:0] fq [0:3];
    reg [1:0]  fh, ft;
    reg [2:0]  fc;
    reg        run;           // frame in progress (requests allowed)
    reg [14:0] issued;
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
    wire [15:0] word = fq[fh];

    // ---- stage 1: sequencing
    reg [3:0] l;          // layer
    reg [1:0] ph;         // 0 shift, 1 bias, 2 weights, 3 multipliers
    reg [6:0] cnt;
    reg [4:0] g;
    reg       s1_on;      // stage 1 active
    reg [3:0] wp [0:4];   // history write slot per depthwise layer
    reg [3:0] tcnt;       // frames since the session start, saturating at 8
    wire [1:0] kd = kind_of(l);
    wire [2:0] jd = 3'((l - 4'd1) >> 1);       // depthwise layer index for l odd
    wire [2:0] jw = 3'(l >> 1);                // history written by pointwise layer l (even, < 10)
    assign pop = s1_on && (fc != 0);
    // depthwise tap k = cnt: slot = wp - (4 - k) d mod R; zero when (4 - k) d > tcnt
    wire [4:0] back = 5'((3'd4 - cnt[2:0]) * dil_of(jd));
    wire [4:0] sl0  = {1'b0, wp[jd]} + {1'b0, rs_of(jd)} - back;
    wire [3:0] slot = (sl0 >= {1'b0, rs_of(jd)}) ? 4'(sl0 - {1'b0, rs_of(jd)}) : sl0[3:0];
    wire       tz   = back > {1'b0, tcnt};
    wire is_w = (ph == 2'd2);
    assign xr_re = pop && is_w && (kd == 2'd0);
    assign xr_a  = cnt[5:0];
    assign hr_re = pop && is_w && (kd == 2'd1);
    assign hr_a  = hb_of(jd) + 10'(slot) * 10'd24 + 10'(g);
    assign vr_re = pop && is_w && (kd == 2'd2);
    assign vr_a  = cnt[6:2];
    assign or_re = pop && is_w && (kd == 2'd3);
    assign or_a  = cnt[6:2];

    // ---- stage 2 registers
    reg        s2_v;
    reg [1:0]  s2_ph, s2_kd;
    reg [6:0]  s2_cnt;
    reg [4:0]  s2_g;
    reg [3:0]  s2_l;
    reg        s2_tz;
    reg [15:0] s2_w;
    reg        s2_last;      // last word of the frame
    reg [5:0]  shf;
    reg signed [31:0] acc [0:3];
    reg [23:0] ybuf;

    // activation per lane for the word in stage 2
    function automatic signed [8:0] act(input [1:0] k, input [6:0] c, input [1:0] lane, input z,
                                        input [7:0] xd, input [31:0] hd, input [31:0] vd, input [31:0] od);
        reg [7:0] b;
        begin
            case (k)
                2'd0: act = $signed({xd[7], xd});
                2'd1: begin b = hd[8*lane +: 8]; act = z ? 9'sd0 : $signed({1'b0, b}); end
                2'd2: begin b = vd[8*c[1:0] +: 8]; act = $signed({1'b0, b}); end
                default: begin b = od[8*c[1:0] +: 8]; act = $signed({1'b0, b}); end
            endcase
        end
    endfunction
    // MAC products, explicitly sign-extended to 13 bits (4b x 9b)
    wire signed [12:0] pr [0:3];
    genvar gq;
    generate for (gq = 0; gq < 4; gq = gq + 1) begin : g_mac
        wire signed [8:0]  aq   = act(s2_kd, s2_cnt, gq[1:0], s2_tz, xr_d, hr_d, vr_d, or_d);
        wire signed [12:0] wq13 = {{9{s2_w[4*gq+3]}}, s2_w[4*gq +: 4]};
        wire signed [12:0] aq13 = {{4{aq[8]}}, aq};
        assign pr[gq] = wq13 * aq13;
    end endgenerate
    // requantization of lane s2_cnt[1:0]
    wire signed [31:0] acc_l = acc[s2_cnt[1:0]];
    wire signed [48:0] prod  = acc_l * $signed({1'b0, s2_w});              // 32 x 17 signed
    wire signed [48:0] rq    = (prod + (49'sd1 <<< (shf - 6'd1))) >>> shf;
    wire [7:0]  yb  = (rq < 0) ? 8'd0 : (rq > 255) ? 8'd255 : rq[7:0];
    wire signed [15:0] yz = (rq < -49'sd32768) ? -16'sd32768 : (rq > 49'sd32767) ? 16'sd32767 : rq[15:0];

    integer q;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            l <= 0; ph <= 0; cnt <= 0; g <= 0; s1_on <= 0; run <= 0; busy <= 0; done <= 0; tcnt <= 0;
            for (q = 0; q < 5; q = q + 1) wp[q] <= 0;
            s2_v <= 0; s2_ph <= 0; s2_kd <= 0; s2_cnt <= 0; s2_g <= 0; s2_l <= 0; s2_tz <= 0; s2_w <= 0; s2_last <= 0;
            shf <= 1; ybuf <= 0; h_we <= 0; v_we <= 0; o_we <= 0; h_wa <= 0; v_wa <= 0; o_wa <= 0; w_wd <= 0;
            zmax <= -16'sd32768; dbg_v <= 0; dbg_d <= 0;
            for (q = 0; q < 4; q = q + 1) acc[q] <= 0;
        end else begin
            done <= 0; h_we <= 0; v_we <= 0; o_we <= 0; dbg_v <= 0;
            if (clr) begin
                l <= 0; ph <= 0; cnt <= 0; g <= 0; s1_on <= 0; run <= 0; busy <= 0; s2_v <= 0; tcnt <= 0;
                for (q = 0; q < 5; q = q + 1) wp[q] <= 0;
            end else begin
                if (start) begin
                    l <= 0; ph <= 0; cnt <= 0; g <= 0; s1_on <= 1; run <= 1; busy <= 1; zmax <= -16'sd32768;
                end
                // ---- stage 1: pop and advance
                s2_v <= pop;
                if (pop) begin
                    s2_ph <= ph; s2_kd <= kd; s2_cnt <= cnt; s2_g <= g; s2_l <= l; s2_tz <= tz; s2_w <= word;
                    s2_last <= 1'b0;
                    case (ph)
                        2'd0: begin ph <= 2'd1; cnt <= 0; end
                        2'd1: if (cnt == 7'd7) begin ph <= 2'd2; cnt <= 0; end else cnt <= cnt + 1;
                        2'd2: if (cnt == nw_of(l) - 7'd1) begin ph <= 2'd3; cnt <= 0; end else cnt <= cnt + 1;
                        default: if (cnt == 7'd3) begin
                            cnt <= 0;
                            if (g == ng_of(l) - 5'd1) begin
                                g <= 0; ph <= 2'd0;
                                if (l == 4'(NL - 1)) begin s1_on <= 0; s2_last <= 1'b1; end
                                else l <= l + 1;
                            end else begin g <= g + 1; ph <= 2'd1; end
                        end else cnt <= cnt + 1;
                    endcase
                end
                // ---- stage 2: execute
                if (s2_v) begin
                    case (s2_ph)
                        2'd0: shf <= s2_w[5:0];
                        2'd1: if (s2_cnt[0]) acc[s2_cnt[2:1]][31:16] <= s2_w;
                              else           acc[s2_cnt[2:1]][15:0]  <= s2_w;
                        2'd2: for (q = 0; q < 4; q = q + 1)
                                  acc[q] <= acc[q] + {{19{pr[q][12]}}, pr[q]};
                        default: begin
                            if (s2_kd == 2'd3) begin
                                zb[{s2_g, 2'b00} + 6'(s2_cnt[1:0])] <= yz;
                                if (yz > zmax) zmax <= yz;
                                dbg_v <= 1; dbg_d <= {6'({s2_g, 2'b00} + 6'(s2_cnt[1:0])), yz};
                            end else if (s2_cnt[1:0] != 2'd3) begin
                                ybuf[8*s2_cnt[1:0] +: 8] <= yb;
                            end else begin
                                w_wd <= {yb, ybuf};
                                if (s2_kd == 2'd1)       begin v_we <= 1; v_wa <= s2_g; end
                                else if (s2_l == 4'd10)  begin o_we <= 1; o_wa <= s2_g; end
                                else begin
                                    h_we <= 1;
                                    h_wa <= hb_of(3'(s2_l >> 1)) + 10'(wp[3'(s2_l >> 1)]) * 10'd24 + 10'(s2_g);
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
endmodule
