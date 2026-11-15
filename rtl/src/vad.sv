// Voice activity detector: band-passed frame level (two one-pole filters, about 320 Hz to
// 1.8 kHz, then sum of |b|) and zero-crossing count against an adaptive noise floor, with
// attack and hangover. The threshold also has to clear the mean of the non-speech frames
// plus a multiple of their mean deviation, which keeps the detector off in fluctuating
// noise such as a crowd. The noise statistics never learn speech: they take only frames
// under K_G/4 times the floor, and a frame over K_CAP/4 times the floor is speech whatever
// they hold, so the detector cannot lock itself out. No multiplier (only constant factors).
// Reference and full description: rtl/model/vad_model.py.
module vad #(
    parameter FRAME_LOG2 = 8,     // 256 samples = 16 ms at 16 kHz
    parameter K_ON       = 8,     // on threshold  = floor * K_ON / 4
    parameter K_OFF      = 6,     // threshold while active
    parameter K_Z        = 5,     // threshold when the zero-crossing count is unlike the noise
    parameter Z_DIFF     = 24,
    parameter E_MIN      = 1024,
    parameter ATTACK     = 2,
    parameter HANG       = 20,
    parameter UP_IDLE    = 5,
    parameter UP_ACT     = 7,
    parameter DOWN       = 2,
    parameter HP_SH      = 3,     // high-pass corner = fs * 2^-HP_SH / (2 pi)
    parameter LP_SH      = 1,
    parameter K_D        = 16,    // threshold >= mean + deviation * K_D / 4 (0 = off)
    parameter K_DOFF     = 12,    // same while active
    parameter NM_SH      = 4,
    parameter ND_SH      = 4,
    parameter K_G        = 16,    // noise statistics learn only frames below floor * K_G / 4 (0 = all; < 64)
    parameter K_CAP      = 48     // a frame over floor * K_CAP / 4 (and E_MIN) is speech (0 = off; < 64)
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               in_valid,
    input  wire signed [15:0] in_data,
    output reg                vad_o,       // speech active (attack + hangover applied)
    output reg                frame_stb,   // one clock, after each frame update
    output reg                raw_o,       // this frame alone looked like speech
    output reg  [23:0]        e_o,         // frame level
    output reg  [7:0]         z_o,         // frame zero crossings
    output reg  [23:0]        nf_o,        // noise floor
    output reg  [11:0]        zn_o,        // noise zero-crossing average, 4 fraction bits
    output reg  [23:0]        nm_o,        // mean level of the non-speech frames
    output reg  [23:0]        nd_o         // their mean absolute deviation from nm_o
);
    localparam [23:0] EMAX = 24'hFFFFFF;

    reg [FRAME_LOG2-1:0] cnt;
    reg [24:0]           e_acc;
    reg [8:0]            z_acc;
    reg                  prev_neg;
    reg                  upd;
    reg                  init;
    reg [7:0]            att;
    reg [7:0]            hang;

    // ---- per sample, two pipeline stages
    // stage 1: high-pass.  s1 += x - (s1 >>> HP_SH);  hp = x - (s1 >>> HP_SH)
    reg  signed [19:0] s1;
    reg  signed [17:0] hp_r;
    reg                hp_v;
    wire signed [19:0] xin  = {{4{in_data[15]}}, in_data};
    wire signed [19:0] s1n  = s1 + xin - (s1 >>> HP_SH);
    wire signed [19:0] hpw  = xin - (s1n >>> HP_SH);
    // stage 2: low-pass.   s2 += hp - (s2 >>> LP_SH); b = s2 >>> LP_SH
    reg  signed [19:0] s2;
    wire signed [19:0] hpx  = {{2{hp_r[17]}}, hp_r};
    wire signed [19:0] s2n  = s2 + hpx - (s2 >>> LP_SH);
    wire signed [19:0] bw   = s2n >>> LP_SH;
    wire [19:0]        absb = bw[19] ? (~bw + 20'd1) : bw;
    wire [24:0] e_sum  = e_acc + {5'd0, absb};
    wire [23:0] e_sat  = e_sum[24] ? EMAX : e_sum[23:0];
    wire        zc     = hp_r[17] ^ prev_neg;
    wire [8:0]  z_sum  = z_acc + {8'd0, zc};
    wire [7:0]  z_sat  = z_sum[8] ? 8'hFF : z_sum[7:0];
    wire        last   = &cnt;

    // ---- per frame (from the registered e_o / z_o)
    wire [3:0]  k      = vad_o ? K_OFF : K_ON;
    wire [27:0] thr_f  = nf_o * k;
    wire [3:0]  kz     = K_Z;
    wire [27:0] thrz_f = nf_o * kz;
    // mean + deviation test
    wire [4:0]  kd     = vad_o ? K_DOFF : K_D;
    wire [4:0]  kdz    = K_D;
    wire [28:0] dev_f  = nd_o * kd;
    wire [28:0] devz_f = nd_o * kdz;
    wire [27:0] thr_d  = {4'd0, nm_o} + {1'b0, dev_f[28:2]};
    wire [27:0] thr_dz = {4'd0, nm_o} + {1'b0, devz_f[28:2]};
    wire        use_d  = (K_D != 0);
    wire [27:0] thr_a  = {2'b00, thr_f[27:2]};
    wire [27:0] thr_az = {2'b00, thrz_f[27:2]};
    wire [27:0] thr    = (use_d && (thr_d  > thr_a))  ? thr_d  : thr_a;
    wire [27:0] thr_z  = (use_d && (thr_dz > thr_az)) ? thr_dz : thr_az;
    wire [8:0]  zd_s   = {1'b0, z_o} - {1'b0, zn_o[11:4]};
    wire [7:0]  zdiff  = zd_s[8] ? (~zd_s[7:0] + 8'd1) : zd_s[7:0];
    wire        above  = e_o > E_MIN;
    wire        loud   = ({4'd0, e_o} > thr) && above;
    wire        odd    = ({4'd0, e_o} > thr_z) && above && (zdiff > Z_DIFF);
    // far above the floor is speech whatever nm and nd say: the detector cannot lock itself out
    wire [5:0]  kc     = K_CAP;
    wire [29:0] cap_f  = nf_o * kc;
    wire        huge   = (K_CAP != 0) && ({4'd0, e_o} > cap_f[29:2]) && above;
    wire        raw    = loud | odd | huge;

    wire [23:0] nf_dn  = nf_o - ((nf_o - e_o) >> DOWN);
    wire [24:0] nf_up  = {1'b0, nf_o} + {1'b0, ((vad_o | raw) ? (nf_o >> UP_ACT) : (nf_o >> UP_IDLE))} + 25'd1;
    wire [23:0] nf_n   = (e_o < nf_o) ? nf_dn : (nf_up[24] ? EMAX : nf_up[23:0]);

    wire signed [12:0] zn_d = $signed({1'b0, z_o, 4'd0}) - $signed({1'b0, zn_o});
    wire signed [12:0] zn_s = zn_d >>> 3;
    wire [12:0]        zn_n = {1'b0, zn_o} + zn_s;

    // nd += (|e - nm| - nd) >>> ND_SH;  nm += (e - nm) >>> NM_SH   (non-speech frames under floor * K_G / 4 only)
    wire [5:0]  kg     = K_G;
    wire [29:0] gate_f = nf_o * kg;
    wire        quiet  = (K_G == 0) || ({4'd0, e_o} < gate_f[29:2]);
    wire signed [24:0] em_d   = $signed({1'b0, e_o}) - $signed({1'b0, nm_o});
    wire [23:0]        em_abs = em_d[24] ? (~em_d[23:0] + 24'd1) : em_d[23:0];
    wire signed [24:0] nd_d   = $signed({1'b0, em_abs}) - $signed({1'b0, nd_o});
    wire signed [24:0] nd_s   = nd_d >>> ND_SH;
    wire signed [24:0] nd_n   = $signed({1'b0, nd_o}) + nd_s;
    wire signed [24:0] nm_s   = em_d >>> NM_SH;
    wire signed [24:0] nm_n   = $signed({1'b0, nm_o}) + nm_s;

    wire [7:0]  att_n  = raw ? ((att >= ATTACK) ? att : att + 8'd1) : 8'd0;

    // everything here only changes around a sample or a frame update: one gated clock
    wire gclk;
    clk_gate u_cg (.clk(clk), .en(in_valid | hp_v | upd | frame_stb), .gclk(gclk));

    always @(posedge gclk or negedge rst_n) begin
        if (!rst_n) begin
            cnt       <= {FRAME_LOG2{1'b0}};
            e_acc     <= 25'd0;
            z_acc     <= 9'd0;
            prev_neg  <= 1'b0;
            s1        <= 20'd0;
            s2        <= 20'd0;
            hp_r      <= 18'd0;
            hp_v      <= 1'b0;
            upd       <= 1'b0;
            init      <= 1'b1;
            att       <= 8'd0;
            hang      <= 8'd0;
            vad_o     <= 1'b0;
            frame_stb <= 1'b0;
            raw_o     <= 1'b0;
            e_o       <= 24'd0;
            z_o       <= 8'd0;
            nf_o      <= 24'd0;
            zn_o      <= 12'd0;
            nm_o      <= 24'd0;
            nd_o      <= 24'd0;
        end else begin
            upd       <= 1'b0;
            frame_stb <= upd;

            hp_v <= in_valid;
            if (in_valid) begin
                s1   <= s1n;
                hp_r <= hpw[17:0];
            end
            if (hp_v) begin
                s2       <= s2n;
                prev_neg <= hp_r[17];
                cnt      <= cnt + 1'b1;
                if (last) begin
                    e_o   <= e_sat;
                    z_o   <= z_sat;
                    e_acc <= 25'd0;
                    z_acc <= 9'd0;
                    upd   <= 1'b1;
                end else begin
                    e_acc <= {1'b0, e_sat};
                    z_acc <= {1'b0, z_sat};
                end
            end

            if (upd) begin
                if (init) begin
                    init  <= 1'b0;
                    nf_o  <= e_o;
                    nm_o  <= e_o;
                    nd_o  <= 24'd0;
                    zn_o  <= {z_o, 4'd0};
                    raw_o <= 1'b0;
                end else begin
                    raw_o <= raw;
                    nf_o  <= nf_n;
                    if (!raw && !vad_o && quiet) begin
                        zn_o <= zn_n[11:0];
                        nm_o <= nm_n[23:0];
                        nd_o <= nd_n[23:0];
                    end
                    att   <= att_n;
                    if (raw && (att_n >= ATTACK)) begin
                        vad_o <= 1'b1;
                        hang  <= HANG;
                    end else if (vad_o) begin
                        if (raw)               hang  <= HANG;
                        else if (hang == 8'd0) vad_o <= 1'b0;
                        else                   hang  <= hang - 8'd1;
                    end
                end
            end
        end
    end
endmodule
