// Wake spotter front end (docs/golden-wake.md, section FE; golden sw/wake/fe.py).
// One run per network frame t: frames a = x[320t .. +400) and b = x[320t+160 .. +400) are read from the sample
// ring, Hann-windowed (Q15), packed as re/im of one 512-point radix-2 DIT FFT (28-bit fields, two multipliers,
// 7 clocks per butterfly), split into the two real spectra, max + 3/8 min magnitude, 32 triangular mel bands
// streamed with two accumulators per frame, log2 code, offset and clamp. Writes 64 int8 features (v - 128):
// x[0..31] frame a, x[32..63] frame b. About 20k clocks per run.
module wk_fe #(
    parameter MEMDIR = ""
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               clr,
    input  wire               start,
    input  wire [9:0]         base,
    output reg                busy,
    output reg                done,
    output reg                sr_re,
    output reg  [9:0]         sr_addr,
    input  wire signed [15:0] sr_rdata,
    output reg                x_we,
    output reg  [5:0]         x_addr,
    output reg  [7:0]         x_data,
    output reg                dbg_v,
    output reg  [15:0]        dbg_d          // {band 6b + frame b flag in bit 6, code v 8b}
);
    localparam K_MIN = 2, K_MAX = 243, FOFF = 96;

    // ---- tables (ROM)
    reg [15:0] win_t [0:199];
    reg [15:0] sin_t [0:128];
    reg [15:0] mel_t [0:K_MAX-K_MIN];
    initial begin
        $readmemh({MEMDIR, "wake_fe_win.mem"}, win_t);
        $readmemh({MEMDIR, "wake_fe_sin.mem"}, sin_t);
        $readmemh({MEMDIR, "wake_fe_mel.mem"}, mel_t);
    end
    function automatic [4:0] lut(input [3:0] f);
        case (f)
            4'd0: lut = 0;  4'd1: lut = 1;  4'd2: lut = 3;  4'd3: lut = 4;  4'd4: lut = 5;  4'd5: lut = 6;
            4'd6: lut = 7;  4'd7: lut = 8;  4'd8: lut = 9;  4'd9: lut = 10; 4'd10: lut = 11; 4'd11: lut = 12;
            4'd12: lut = 13; 4'd13: lut = 14; 4'd14: lut = 15; default: lut = 15;
        endcase
    endfunction
    function automatic [7:0] code(input [47:0] e);       // uint8 feature v
        integer p, q;
        reg [3:0] f;
        reg [11:0] L;
        reg signed [12:0] v;
        begin
            p = -1;
            for (q = 0; q < 48; q = q + 1) if (e[q]) p = q;
            if (p < 0) L = 0;
            else begin
                if (p >= 4) f = e[p-1 -: 4];
                else        f = 4'((e << (4 - p)) & 48'hF);
                L = 12'(16 * p) + {7'd0, lut(f)};
            end
            v = $signed({2'b0, L[11:1]}) - 13'sd96;
            code = (v < 0) ? 8'd0 : (v > 255) ? 8'd255 : v[7:0];
        end
    endfunction
    function automatic [8:0] bitrev9(input [8:0] i);
        integer q;
        for (q = 0; q < 9; q = q + 1) bitrev9[q] = i[8-q];
    endfunction

    // ---- FFT RAM: {re 28b, im 28b}
    reg         f_we, f_re;
    reg  [8:0]  f_wa, f_ra;
    reg  [55:0] f_wd;
    wire [55:0] f_rd;
    wk_ram #(.WORDS(512), .WIDTH(56)) u_fft (.clk(clk), .we(f_we), .waddr(f_wa), .wdata(f_wd), .re(f_re), .raddr(f_ra), .rdata(f_rd));
    wire signed [27:0] rd_re = f_rd[55:28], rd_im = f_rd[27:0];

    // ---- two multipliers: 32b x 17b signed
    reg  signed [31:0] m1a, m2a;
    reg  signed [16:0] m1b, m2b;
    wire signed [48:0] m1 = m1a * m1b;
    wire signed [48:0] m2 = m2a * m2b;

    localparam S_IDLE = 0, S_LD = 1, S_FFT = 2, S_MEL = 3, S_EMIT = 4, S_FLUSH = 5;
    reg [2:0]  st;
    reg [3:0]  ph;          // phase within a step
    reg [9:0]  bse;
    reg [9:0]  i;           // load index / bin
    reg [3:0]  stg;         // FFT stage 1..9
    reg [7:0]  j;           // butterfly index
    reg signed [27:0] ar, ai, br_, bi_, tr, ti;
    reg signed [15:0] ya;
    reg signed [27:0] zkr, zki;
    reg [47:0] alA, ahA, alB, ahB;   // accumulators: band i-1 (lo), band i (hi), frames a and b
    reg [5:0]  iprev;
    reg        first;
    reg [28:0] mA, mB;
    reg        emit_b, flush;
    reg [5:0]  eb;          // band being emitted

    // window coefficient for load index
    wire [7:0]  wi   = (i < 10'd200) ? i[7:0] : 8'(10'd399 - i);
    wire [15:0] wco  = win_t[wi];
    // butterfly addressing
    wire [8:0]  half = 9'd1 << (stg - 4'd1);
    wire [8:0]  pos  = {1'b0, j} & (half - 9'd1);
    wire [8:0]  top  = (({1'b0, j} >> (stg - 4'd1)) << stg) | pos;
    wire [8:0]  bot  = top | half;
    wire [8:0]  tk   = pos << (4'd9 - stg);                  // twiddle index 0..255
    wire signed [16:0] tw_c = (tk <= 9'd128) ? $signed({1'b0, sin_t[8'(9'd128 - tk)]}) : -$signed({1'b0, sin_t[8'(tk - 9'd128)]});
    wire signed [16:0] tw_s = (tk <= 9'd128) ? $signed({1'b0, sin_t[tk[7:0]]}) : $signed({1'b0, sin_t[8'(9'd256 - tk)]});
    // mel table for bin i
    wire [15:0] melw = mel_t[8'(i - K_MIN)];
    wire [5:0]  mi   = melw[14:9];
    wire [8:0]  mwu  = melw[8:0];
    // magnitudes from zk (registered) and Z[N-k] (f_rd)
    wire signed [28:0] Ar = zkr + rd_re, Ai = zki - rd_im, Br = zki + rd_im, Bi = rd_re - zkr;
    function automatic [28:0] mag(input signed [28:0] r, input signed [28:0] im);
        reg [28:0] a, b, mx, mn;
        begin
            a = r[28] ? 29'(-r) : r;
            b = im[28] ? 29'(-im) : im;
            mx = (a > b) ? a : b;
            mn = (a > b) ? b : a;
            mag = mx + 29'((30'(mn) * 30'd3) >> 3);
        end
    endfunction
    wire signed [48:0] rnd14 = 49'sd16384;
    wire [7:0] cv = code(emit_b ? alB : alA);           // one shared log encoder

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; ph <= 0; busy <= 0; done <= 0; sr_re <= 0; sr_addr <= 0; x_we <= 0; x_addr <= 0; x_data <= 0;
            f_we <= 0; f_re <= 0; f_wa <= 0; f_ra <= 0; f_wd <= 0; i <= 0; stg <= 1; j <= 0; bse <= 0;
            ar <= 0; ai <= 0; br_ <= 0; bi_ <= 0; tr <= 0; ti <= 0; ya <= 0; zkr <= 0; zki <= 0;
            alA <= 0; ahA <= 0; alB <= 0; ahB <= 0; iprev <= 0; first <= 1; mA <= 0; mB <= 0; emit_b <= 0; flush <= 0; eb <= 0;
            m1a <= 0; m1b <= 0; m2a <= 0; m2b <= 0; dbg_v <= 0; dbg_d <= 0;
        end else begin
            done <= 0; sr_re <= 0; x_we <= 0; f_we <= 0; f_re <= 0; dbg_v <= 0;
            if (clr) begin
                st <= S_IDLE; busy <= 0; ph <= 0;
            end else case (st)
            S_IDLE: if (start) begin
                st <= S_LD; busy <= 1; ph <= 0; i <= 0; bse <= base;
            end
            // ---- load: 4 clocks per index below 400, 1 clock for the zero padding
            S_LD: begin
                if (i >= 10'd400) begin
                    f_we <= 1; f_wa <= bitrev9(i[8:0]); f_wd <= 56'd0;
                    if (i == 10'd511) begin st <= S_FFT; stg <= 1; j <= 0; ph <= 0; end
                    i <= i + 1;
                end else case (ph)
                    0: begin sr_re <= 1; sr_addr <= bse + i; ph <= 1; end
                    1: begin sr_re <= 1; sr_addr <= bse + 10'd160 + i; ph <= 2; end
                    2: begin                         // frame a sample arrived: m1 = x w
                        m1a <= {{16{sr_rdata[15]}}, sr_rdata}; m1b <= {1'b0, wco}; ph <= 3;
                    end
                    3: begin                         // frame b sample arrived: m2 = x w
                        m2a <= {{16{sr_rdata[15]}}, sr_rdata}; m2b <= {1'b0, wco};
                        ya <= 16'((m1 + rnd14) >>> 15); ph <= 4;
                    end
                    4: begin
                        f_we <= 1; f_wa <= bitrev9(i[8:0]);
                        f_wd <= {{12{ya[15]}}, ya, 28'((m2 + rnd14) >>> 15)};
                        i <= i + 1; ph <= 0;
                    end
                    default: ph <= 0;
                endcase
            end
            // ---- FFT: 7 clocks per butterfly
            S_FFT: case (ph)
                0: begin f_re <= 1; f_ra <= top; ph <= 1; end
                1: begin f_re <= 1; f_ra <= bot; ph <= 2; end
                2: begin ar <= rd_re; ai <= rd_im; ph <= 3; end
                3: begin                                     // b arrived: tr = (br c + bi s + 2^14) >> 15
                    br_ <= rd_re; bi_ <= rd_im;
                    m1a <= 32'(rd_re); m1b <= tw_c; m2a <= 32'(rd_im); m2b <= tw_s; ph <= 4;
                end
                4: begin
                    tr <= 28'((m1 + m2 + rnd14) >>> 15);
                    m1a <= 32'(bi_); m1b <= tw_c; m2a <= 32'(br_); m2b <= tw_s; ph <= 5;
                end
                5: begin
                    ti <= 28'((m1 - m2 + rnd14) >>> 15);
                    ph <= 6;
                end
                6: begin f_we <= 1; f_wa <= top; f_wd <= {28'(ar + tr), 28'(ai + ti)}; ph <= 7; end
                7: begin
                    f_we <= 1; f_wa <= bot; f_wd <= {28'(ar - tr), 28'(ai - ti)}; ph <= 0;
                    if (j == 8'd255) begin
                        j <= 0;
                        if (stg == 4'd9) begin st <= S_MEL; i <= K_MIN; first <= 1; iprev <= 0; alA <= 0; ahA <= 0; alB <= 0; ahB <= 0; end
                        else stg <= stg + 1;
                    end else j <= j + 1;
                end
                default: ph <= 0;
            endcase
            // ---- mel: per bin read Z[k], Z[512-k]; magnitudes; band change emits band iprev-1
            S_MEL: case (ph)
                0: begin f_re <= 1; f_ra <= i[8:0]; ph <= 1; end
                1: begin f_re <= 1; f_ra <= 9'(10'd512 - i); ph <= 2; end
                2: begin zkr <= rd_re; zki <= rd_im; ph <= 3; end
                3: begin
                    mA <= mag(Ar, Ai); mB <= mag(Br, Bi);
                    if (!first && mi != iprev) begin
                        if (iprev >= 6'd1) begin st <= S_EMIT; emit_b <= 0; flush <= 0; eb <= iprev - 6'd1; end
                        else begin alA <= ahA; alB <= ahB; ahA <= 0; ahB <= 0; ph <= 4; end
                    end else ph <= 4;
                    first <= 0; iprev <= mi;
                end
                4: begin                                     // m1 = m wu (band i), m2 = m (256 - wu) (band i-1)
                    m1a <= {3'd0, mA}; m1b <= {8'd0, mwu}; m2a <= {3'd0, mA}; m2b <= {8'd0, 9'(9'd256 - mwu)};
                    ph <= 5;
                end
                5: begin                                     // frame a
                    if (mi < 6'd32) ahA <= ahA + m1[47:0];
                    if (mi >= 6'd1) alA <= alA + m2[47:0];
                    m1a <= {3'd0, mB}; m2a <= {3'd0, mB};
                    ph <= 6;
                end
                6: begin                                     // frame b
                    if (mi < 6'd32) ahB <= ahB + m1[47:0];
                    if (mi >= 6'd1) alB <= alB + m2[47:0];
                    ph <= 0;
                    if (i == K_MAX) begin st <= S_EMIT; emit_b <= 0; flush <= 1; eb <= iprev - 6'd1; end
                    else i <= i + 1;
                end
                default: ph <= 0;
            endcase
            // ---- emit band iprev-1 (or band 31 at the end) for frames a and b
            S_EMIT: begin
                x_we <= 1;
                dbg_v <= 1;
                if (!emit_b) begin
                    x_addr <= eb; x_data <= cv ^ 8'h80;
                    dbg_d <= {2'b00, eb, cv};
                    emit_b <= 1;
                end else begin
                    x_addr <= 6'd32 + eb; x_data <= cv ^ 8'h80;
                    dbg_d <= {2'b01, eb, cv};
                    if (flush) begin
                        st <= S_IDLE; busy <= 0; done <= 1;
                    end else begin
                        alA <= ahA; alB <= ahB; ahA <= 0; ahB <= 0;
                        st <= S_MEL; ph <= 4;
                    end
                end
            end
            default: st <= S_IDLE;
            endcase
        end
    end
endmodule
