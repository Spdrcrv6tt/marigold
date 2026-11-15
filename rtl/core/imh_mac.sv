// Imhotep MAC array with its hardwired row sequencer, requantiser and argmax.
// One operation (docs/architecture.md section 2; C reference: mac_go() in fw/hal_host.c):
//
//   for o in 0..nout-1:
//       stream sources: read 8 header bytes: bias[39:0], M[15:0], sh[7:0]
//       RAM sources:    M = 1, sh = 0, or (HDRRAM) M = the 32-bit RAM word at h + 4*o and sh = hsh
//       acc = bias (0 for RAM weights)
//       for i in 0..nin-1:  acc += x * w
//       output: sat16 / sat25 in two planes (optionally added to what the planes hold: RESACC) /
//               raw 48 bit in three planes / running argmax of the unsaturated requantised value
//
// How x and w meet the lanes, by mode:
//   batch  (default) x = RAM word at element x + o*xo + i*xi: LANES values, one per lane; one weight per
//                    clock shared by all lanes (frames in parallel). Output: a word of LANES results.
//   scalar           x = that one element, lane 0 only; one weight per clock. Output: one element.
//   fan-in           x is a contiguous int16 vector (x aligned to a word, xi = 1): each clock a RAM word gives
//                    LANES consecutive x and the weight source gives up to LANES consecutive weights, one pair
//                    per lane; the row result is the sum of the lane accumulators. Output: one element.
//                    This is how a batch of one (the decoder) uses every multiplier when the weights are local.
// X8 (ctrl bit 8, batch and scalar): x is int8, addressed in bytes (batch: LANES consecutive bytes, i.e. half a RAM
// word, x a multiple of LANES; scalar: one byte), sign-extended into the lanes (cross-attention K and V as int8).
// Ring (xw != 0, batch and scalar modes): the x offset of input i inside row o is (xs0 + i*xi) mod xw, i.e. x is
// read from a ring of xw elements starting at x + o*xo (xi < xw, xs0 < xw). Local attention reads a window of a
// K/V ring buffer in place.
// Weight source port: ws_data carries ws_avail valid bytes (the next bytes of the stream, low byte first);
// ws_take says how many were consumed this clock. An on-die ROM offers LANES bytes per clock, the flash one.
// RAM weights take 2 clocks per MAC (two reads of a single-port RAM).
// Requantise is serial: per lane, the product acc * M is built from 16-bit limbs on one 17 x 17 multiplier.
// Lanes are the front-end agent's mac_lane (16 x 18 multiply, 48-bit accumulator, registered product).
module imh_mac #(
    parameter LANES = 16
) (
    clk, rst_n,
    reg_we, reg_idx, reg_wdata, go, busy, done,
    argidx, argval, satflag,
    m_en, m_we, m_addr, m_wdata, m_rdata,
    seek_valid, seek_addr, ws_data, ws_avail, ws_take
);
    localparam W  = LANES * 16;
    localparam NB = LANES * 2;
    localparam LB = $clog2(LANES);

    input  wire          clk;
    input  wire          rst_n;
    input  wire          reg_we;       // 0..11: faddr x xi xo w wi wo y ys yp nin nout; 16: h; 17: hsh; 18: xw; 19: xs0
    input  wire [4:0]    reg_idx;
    input  wire [31:0]   reg_wdata;
    input  wire          go;           // start, reg_wdata[8:0] = ctrl
    output wire          busy;
    output wire          done;
    output reg  [31:0]   argidx;
    output reg  [31:0]   argval;
    output reg           satflag;
    output reg           m_en;
    output reg  [NB-1:0] m_we;
    output reg  [31:0]   m_addr;       // word address
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

    reg [31:0] r_faddr, r_x, r_xi, r_xo, r_w, r_wi, r_wo, r_y, r_ys, r_yp, r_nin, r_nout, r_h;
    reg [5:0]  r_hsh;
    reg [31:0] r_xw, r_xs0;
    always @(posedge gclk or negedge rst_n) begin
        if (!rst_n) begin
            r_faddr <= 0; r_x <= 0; r_xi <= 0; r_xo <= 0; r_w <= 0; r_wi <= 0; r_wo <= 0;
            r_y <= 0; r_ys <= 0; r_yp <= 0; r_nin <= 0; r_nout <= 0; r_h <= 0; r_hsh <= 0; r_xw <= 0; r_xs0 <= 0;
        end else if (reg_we) begin
            case (reg_idx)
                5'd0: r_faddr <= reg_wdata;
                5'd1: r_x <= reg_wdata;
                5'd2: r_xi <= reg_wdata;
                5'd3: r_xo <= reg_wdata;
                5'd4: r_w <= reg_wdata;
                5'd5: r_wi <= reg_wdata;
                5'd6: r_wo <= reg_wdata;
                5'd7: r_y <= reg_wdata;
                5'd8: r_ys <= reg_wdata;
                5'd9: r_yp <= reg_wdata;
                5'd10: r_nin <= reg_wdata;
                5'd11: r_nout <= reg_wdata;
                5'd16: r_h <= reg_wdata;
                5'd17: r_hsh <= reg_wdata[5:0];
                5'd18: r_xw <= reg_wdata;
                5'd19: r_xs0 <= reg_wdata;
                default: ;
            endcase
        end
    end

    localparam [3:0] S_IDLE = 4'd0, S_SEEK = 4'd1, S_HDR = 4'd2, S_HR0 = 4'd3, S_ROW = 4'd4,
                     S_RUN = 4'd5, S_DRAIN = 4'd6, S_SUM = 4'd7, S_OLD0 = 4'd8, S_OLD1 = 4'd9, S_OLD2 = 4'd10,
                     S_RQ = 4'd11, S_WR = 4'd12, S_NEXT = 4'd13, S_DONE = 4'd14, S_DRAIN2 = 4'd15;
    reg [3:0]  st;
    reg [1:0]  wsrc, om;
    reg        scalar, fanin, resacc, hdrram, x8, xhp, xh;
    wire       stream = !wsrc[1];
    wire       single = scalar || fanin;
    reg [31:0] o, i, xrow, xcur, wrow, wcur, yrow, ycur, hcur;
    reg [63:0] hdr;
    reg [3:0]  hcnt;
    reg        nib, ph;
    reg [W-1:0] xreg, oldlo, oldhi;
    reg [LB:0] xsel, wsel;
    reg [31:0] Mreg;
    reg [5:0]  shreg;

    assign busy = (st != S_IDLE);
    assign done = (st == S_DONE);
    assign seek_valid = (st == S_SEEK);
    assign seek_addr  = r_faddr;

    // ---- shared weight (batch / scalar)
    wire        last  = (i == r_nin - 32'd1);
    wire [3:0]  nibv  = nib ? ws_data[7:4] : ws_data[3:0];
    wire [LB-1:0] wl32 = {wsel[LB-1:1], 1'b0};               // 32-bit weights are 4-byte aligned
    wire [LB-1:0] hl32 = {hcur[LB:2], 1'b0};
    wire [31:0] wword = m_rdata[16 * wl32 +: 32];
    reg signed [17:0] wv;
    always @* begin
        case (wsrc)
            2'd0: wv = {{10{ws_data[7]}}, ws_data[7:0]};
            2'd1: wv = {{14{nibv[3]}}, nibv};
            2'd2: wv = {{2{m_rdata[16 * wsel[LB-1:0] + 15]}}, m_rdata[16 * wsel[LB-1:0] +: 16]};
            default: wv = wword[17:0];
        endcase
    end

    // ---- fan-in: how many elements this clock, and which lanes
    wire [LB:0]  f_lo    = {1'b0, i[LB-1:0]};
    wire [31:0]  f_left  = r_nin - i;
    wire [6:0]   f_have  = wsrc[0] ? {ws_avail, 1'b0} : {1'b0, ws_avail};          // elements on offer
    wire [6:0]   f_room  = LANES - f_lo;
    wire [6:0]   f_m0    = (f_have < f_room) ? f_have : f_room;
    wire [6:0]   f_m     = (f_left < {25'd0, f_m0}) ? f_left[6:0] : f_m0;          // elements taken
    wire [6:0]   f_mb    = (f_m + 7'd1) >> 1;
    wire [5:0]   f_bytes = wsrc[0] ? f_mb[5:0] : f_m[5:0];
    wire         f_last  = ({25'd0, f_m} == f_left);
    wire         f_wrap  = (f_lo + f_m == LANES);

    wire fire_b = (st == S_RUN) && !fanin && (stream ? (ws_avail != 6'd0) : ph);
    wire fire_f = (st == S_RUN) && fanin && (f_m != 7'd0);
    wire [5:0] h_need = 6'd8 - {2'd0, hcnt};

    always @* begin
        ws_take = 6'd0;
        if (st == S_HDR && ws_avail != 6'd0) ws_take = (ws_avail < h_need) ? ws_avail : h_need;
        else if (fire_f) ws_take = f_bytes;
        else if (fire_b && stream && (wsrc == 2'd0 || nib || last)) ws_take = 6'd1;
    end

    // ---- lanes
    wire [W-1:0]        xsrc = stream ? m_rdata : xreg;
    wire [31:0]         xinc  = xcur + r_xi;
    wire [31:0]         xnext = (r_xw != 32'd0 && xinc >= xrow + r_xw) ? xinc - r_xw : xinc;   // ring wrap
    wire [31:0]         xfirst = (r_xw != 32'd0 && !fanin) ? xrow + r_xs0 : xrow;
    wire signed [47:0]  bias = stream ? {{8{hdr[39]}}, hdr[39:0]} : 48'sd0;
    wire signed [47:0]  acc [0:LANES-1];
    wire [LANES-1:0]    lsat;
    genvar g;
    generate
        for (g = 0; g < LANES; g = g + 1) begin : lane
            wire [LB:0]        fidx = g - f_lo;                                     // position inside this clock's weights
            wire               fact = (g >= f_lo) && ({1'b0, fidx} < f_m);
            wire signed [7:0]  fw8  = ws_data[8 * fidx[LB-1:0] +: 8];
            wire signed [3:0]  fw4  = ws_data[4 * fidx[LB-1:0] +: 4];
            wire signed [17:0] wl   = !fanin ? wv : wsrc[0] ? {{14{fw4[3]}}, fw4} : {{10{fw8[7]}}, fw8};
            wire signed [7:0]  xb   = scalar ? xsrc[8 * xsel +: 8] : xsrc[8 * (LANES * (stream ? xhp : xh) + g) +: 8];
            wire signed [15:0] xw16 = scalar ? xsrc[16 * xsel[LB-1:0] +: 16] : xsrc[16 * g +: 16];
            wire signed [15:0] a    = (scalar && g != 0) ? 16'sd0 : x8 ? {{8{xb[7]}}, xb} : xw16;
            wire               en   = fanin ? (fire_f && fact) : (fire_b && (!scalar || g == 0));
            wire signed [47:0] lb   = (fanin && g != 0) ? 48'sd0 : bias;
            mac_lane #(.AW(16), .WW(18), .ACCW(48), .OW(16), .SW(5), .PIPE(2)) u (
                .clk(gclk), .rst_n(rst_n), .load(st == S_ROW), .en(en),
                .a(a), .w(wl), .bias(lb), .shift(5'd0), .acc(acc[g]), .q(), .sat_flag(lsat[g]));
        end
    endgenerate
    // sum of the lane accumulators (fan-in)
    reg signed [47:0] fsum;
    integer k;
    always @* begin
        fsum = 48'sd0;
        for (k = 0; k < LANES; k = k + 1) fsum = fsum + acc[k];
    end
    reg signed [47:0] fsum_r;

    // ---- requantise: one lane at a time; acc * M from 16-bit limbs on one 17 x 17 multiplier
    wire [LB:0]  nl = (single || om == 2'd3) ? 1 : LANES;
    reg  [LB:0]  rc;
    reg  [2:0]   rk;                   // 0..5 product steps, 6 round/shift, 7 store
    reg  signed [63:0] p_r, r_r;
    reg  [W-1:0] outw [0:2];
    reg  signed [63:0] best;
    wire signed [47:0] acc_sel = fanin ? fsum_r : acc[rc[LB-1:0]];
    wire               mh = (rk >= 3'd3);                                          // M limb
    wire [1:0]         la = mh ? rk[1:0] + 2'd1 : rk[1:0];                         // acc limb (rk 3,4,5 -> 0,1,2)
    wire signed [16:0] limb = (la == 2'd0) ? {1'b0, acc_sel[15:0]} : (la == 2'd1) ? {1'b0, acc_sel[31:16]} : {acc_sel[47], acc_sel[47:32]};
    wire signed [16:0] mlimb = mh ? {1'b0, Mreg[31:16]} : {1'b0, Mreg[15:0]};
    wire signed [33:0] lprod = limb * mlimb;
    wire signed [63:0] lext  = {{30{lprod[33]}}, lprod};
    wire [2:0]         lpos  = {1'b0, la} + {2'b00, mh};
    wire signed [63:0] lsh   = lext <<< {lpos, 4'b0000};
    wire signed [63:0] rnd  = (shreg == 6'd0) ? 64'sd0 : (64'sd1 <<< (shreg - 6'd1));
    wire signed [63:0] rq   = (p_r + rnd) >>> shreg;
    wire signed [63:0] s16  = (r_r > 64'sd32767) ? 64'sd32767 : (r_r < -64'sd32767) ? -64'sd32767 : r_r;
    wire signed [63:0] s25  = (r_r > 64'sd16777215) ? 64'sd16777215 : (r_r < -64'sd16777215) ? -64'sd16777215 : r_r;
    wire [LB-1:0]      osel = single ? ycur[LB-1:0] : rc[LB-1:0];                  // lane of the old value (RESACC)
    wire [31:0]        yhi  = yrow + r_yp;
    wire [LB-1:0]      oselh = single ? yhi[LB-1:0] : rc[LB-1:0];
    wire signed [31:0] oldv = {oldhi[16 * oselh +: 16], oldlo[16 * osel +: 16]};
    wire signed [63:0] oldx = {{32{oldv[31]}}, oldv};
    wire signed [63:0] accs = s25 + oldx;
    wire signed [63:0] s25a = (accs > 64'sd16777215) ? 64'sd16777215 : (accs < -64'sd16777215) ? -64'sd16777215 : accs;
    wire signed [63:0] v25  = resacc ? s25a : s25;
    wire [1:0]   np = (om == 2'd0) ? 2'd1 : (om == 2'd1) ? 2'd2 : (om == 2'd2) ? 2'd3 : 2'd0;
    reg  [1:0]   pl;
    wire [LB-1:0] ol = single ? {LB{1'b0}} : rc[LB-1:0];                           // slot of the result in outw

    always @* begin
        m_en = 1'b0; m_we = {NB{1'b0}}; m_addr = 32'd0; m_wdata = {W{1'b0}};
        case (st)
            S_HR0: begin m_en = 1'b1; m_addr = hcur >> (LB + 1); end
            S_ROW: begin m_en = 1'b1; m_addr = xfirst >> (x8 ? LB + 1 : LB); end
            S_RUN: begin
                if (fanin) begin
                    if (fire_f && f_wrap && !f_last) begin m_en = 1'b1; m_addr = (xrow + i + {25'd0, f_m}) >> LB; end
                end else if (stream) begin
                    if (fire_b && !last) begin m_en = 1'b1; m_addr = xnext >> (x8 ? LB + 1 : LB); end
                end else if (!ph) begin
                    m_en = 1'b1; m_addr = wcur >> (LB + 1);
                end else if (!last) begin
                    m_en = 1'b1; m_addr = xnext >> (x8 ? LB + 1 : LB);
                end
            end
            S_OLD0: begin m_en = 1'b1; m_addr = yrow >> LB; end
            S_OLD1: begin m_en = 1'b1; m_addr = (yrow + r_yp) >> LB; end
            S_WR: begin
                m_en    = 1'b1;
                m_addr  = ycur >> LB;
                m_wdata = single ? {LANES{outw[pl][15:0]}} : outw[pl];
                m_we    = single ? ({{(NB-2){1'b0}}, 2'b11} << {ycur[LB-1:0], 1'b0}) : {NB{1'b1}};
            end
            default: ;
        endcase
    end

    integer t;
    always @(posedge gclk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; wsrc <= 2'd0; om <= 2'd0; scalar <= 1'b0; fanin <= 1'b0; resacc <= 1'b0; hdrram <= 1'b0;
            o <= 0; i <= 0; xrow <= 0; xcur <= 0; wrow <= 0; wcur <= 0; yrow <= 0; ycur <= 0; hcur <= 0;
            hdr <= 64'd0; hcnt <= 4'd0; nib <= 1'b0; ph <= 1'b0; xreg <= {W{1'b0}}; xsel <= 0; wsel <= 0;
            x8 <= 1'b0; xhp <= 1'b0; xh <= 1'b0;
            oldlo <= {W{1'b0}}; oldhi <= {W{1'b0}}; fsum_r <= 48'sd0;
            Mreg <= 32'd1; shreg <= 6'd0; rc <= 0; rk <= 3'd0; p_r <= 64'sd0; r_r <= 64'sd0;
            best <= 64'sd0; argidx <= 32'd0; argval <= 32'd0; satflag <= 1'b0; pl <= 2'd0;
        end else begin
            case (st)
                S_IDLE: if (go) begin
                    wsrc <= reg_wdata[1:0]; om <= reg_wdata[3:2]; scalar <= reg_wdata[4]; fanin <= reg_wdata[5];
                    resacc <= reg_wdata[6]; hdrram <= reg_wdata[7]; x8 <= reg_wdata[8];
                    o <= 32'd0; xrow <= r_x; wrow <= r_w; yrow <= r_y; hcur <= r_h; hcnt <= 4'd0;
                    st <= !reg_wdata[1] ? S_SEEK : reg_wdata[7] ? S_HR0 : S_ROW;
                end
                S_SEEK: st <= S_HDR;
                S_HDR: if (ws_avail != 6'd0) begin
                    for (t = 0; t < 8; t = t + 1)
                        if (t < ws_take) hdr[8 * (hcnt[2:0] + t[2:0]) +: 8] <= ws_data[8 * t +: 8];
                    hcnt <= hcnt + ws_take[3:0];
                    if (ws_take == h_need) st <= S_ROW;
                end
                S_HR0: st <= S_ROW;
                S_ROW: begin
                    i <= 32'd0; xcur <= xfirst; wcur <= wrow; nib <= 1'b0; ph <= 1'b0;
                    xsel <= xfirst[LB:0]; xhp <= xfirst[LB];
                    if (stream) begin Mreg <= {16'd0, hdr[55:40]}; shreg <= hdr[61:56]; end
                    else if (hdrram) begin Mreg <= m_rdata[16 * hl32 +: 32]; shreg <= r_hsh; end
                    else begin Mreg <= 32'd1; shreg <= 6'd0; end
                    st <= S_RUN;
                end
                S_RUN: begin
                    if (fanin) begin
                        if (fire_f) begin
                            i <= i + {25'd0, f_m};
                            if (f_last) st <= S_DRAIN;
                        end
                    end else if (stream) begin
                        if (fire_b) begin
                            if (wsrc == 2'd1) nib <= ~nib;
                            if (last) st <= S_DRAIN;
                            else begin i <= i + 32'd1; xcur <= xnext; xsel <= xnext[LB:0]; xhp <= xnext[LB]; end
                        end
                    end else if (!ph) begin
                        xreg <= m_rdata; xh <= xhp; wsel <= wcur[LB+1:1]; ph <= 1'b1;
                    end else begin
                        ph <= 1'b0;
                        if (last) st <= S_DRAIN;
                        else begin
                            i <= i + 32'd1; xcur <= xnext; wcur <= wcur + r_wi; xsel <= xnext[LB:0]; xhp <= xnext[LB];
                        end
                    end
                end
                S_DRAIN: st <= S_DRAIN2;                       // lane pipeline: operands, product, accumulator
                S_DRAIN2: st <= S_SUM;
                S_SUM: begin
                    fsum_r <= fsum;
                    satflag <= satflag | (|lsat);
                    rc <= 0; rk <= (om == 2'd2) ? 3'd6 : 3'd0; ycur <= yrow; pl <= 2'd0;
                    st <= resacc ? S_OLD0 : S_RQ;
                end
                S_OLD0: st <= S_OLD1;
                S_OLD1: begin oldlo <= m_rdata; st <= S_OLD2; end
                S_OLD2: begin oldhi <= m_rdata; st <= S_RQ; end
                S_RQ: begin
                    case (rk)
                        3'd0: begin p_r <= lsh; rk <= 3'd1; end
                        3'd1: begin p_r <= p_r + lsh; rk <= 3'd2; end
                        3'd2: begin p_r <= p_r + lsh; rk <= (Mreg[31:16] != 16'd0) ? 3'd3 : 3'd6; end
                        3'd3: begin p_r <= p_r + lsh; rk <= 3'd4; end
                        3'd4: begin p_r <= p_r + lsh; rk <= 3'd5; end
                        3'd5: begin p_r <= p_r + lsh; rk <= 3'd6; end
                        3'd6: begin
                            if (om == 2'd2) r_r <= $signed(acc_sel); else r_r <= rq;
                            rk <= 3'd7;
                        end
                        default: begin
                            case (om)
                                2'd0: outw[0][16 * ol +: 16] <= s16[15:0];
                                2'd1: begin outw[0][16 * ol +: 16] <= v25[15:0]; outw[1][16 * ol +: 16] <= v25[31:16]; end
                                2'd2: begin
                                    outw[0][16 * ol +: 16] <= r_r[15:0]; outw[1][16 * ol +: 16] <= r_r[31:16];
                                    outw[2][16 * ol +: 16] <= r_r[47:32];
                                end
                                default: if (o == 32'd0 || r_r > best) begin best <= r_r; argidx <= o; argval <= r_r[31:0]; end
                            endcase
                            rc <= rc + 1'b1;
                            rk <= (om == 2'd2) ? 3'd6 : 3'd0;
                            if (rc == nl - 1'b1) st <= (np == 2'd0) ? S_NEXT : S_WR;
                        end
                    endcase
                end
                S_WR: begin
                    pl <= pl + 2'd1; ycur <= ycur + r_yp;
                    if (pl == np - 2'd1) st <= S_NEXT;
                end
                S_NEXT: begin
                    o <= o + 32'd1; xrow <= xrow + r_xo; wrow <= wrow + r_wo; yrow <= yrow + r_ys; hcur <= hcur + 32'd4;
                    hcnt <= 4'd0;
                    if (o == r_nout - 32'd1) st <= S_DONE;
                    else st <= stream ? S_HDR : hdrram ? S_HR0 : S_ROW;
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
