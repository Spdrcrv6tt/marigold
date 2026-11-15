// Fan-out MAC array (Poinsetta, docs/architecture.md section 6). L lanes, each owning one output row of a group of
// L; C frames (a chunk, 1..CMAX; C = 1 for the decoder) share every weight read. C reference: fan_go() in
// fw/hal_host.c. One operation:
//
//   for each group of L output rows o = g*L + l (the last group may be partial: rows o >= nout are not written):
//       weight stream: L header records of 8 bytes {bias[39:0], M[15:0], sh[7:0]}, then for i in 0..nin-1 the L
//       weights of input i (int8: L bytes, lane l = byte l; int4: L/2 bytes, lane 2j low nibble of byte j;
//       int16: 2L bytes, lane l = bytes 2l, 2l + 1, little-endian)
//       acc[c][l] = bias[l]
//       acc[c][l] += x[c][i] * w[i][l]          for i in 0..nin-1, c in 0..C-1: one frame per clock, all lanes
//       per frame c: S16     y[c][o] = sat16(rshift_round(acc * M, sh))           at element y + c*ys + o
//                    RES     sat25 of the same in two planes (low 16 bits at y.., high at y.. + yp); with RESACC
//                            added to the 25-bit value the planes hold, then saturated again
//                    ARGMAX  running argmax of rshift_round(acc * M, sh) over o, lowest index wins, per frame
//                            (argidx/argval: frame 0; argidx_all: every frame's index, frame c at bits 32c)
//   x[c][i]: int16 at RAM element x + c*xs + i. x, xs, y, ys, yp and nin are multiples of 16 (a RAM word).
//
// RAM traffic: x is loaded 16 inputs at a time, one word per frame, so the RAM is read once every 16 MAC clocks per
// frame (Imhotep's batch array read a word on every MAC clock). ROM traffic: one weight vector of L per input,
// reused for the C frames. The requantiser is one pipelined multiplier, one output per clock, after the group's
// MAC loop (not yet overlapped with the next group). Accumulators are 48 bit and do not saturate (the C model
// checks that the arithmetic never needs more).
module imh_fan #(
    parameter L    = 32,
    parameter CMAX = 8,
    parameter RWL  = 16,         // RAM word: RWL elements of 16 bits
    parameter WB   = 16          // weight-source bytes per clock
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               reg_we,       // 0 faddr 1 x 2 xs 3 nin 4 nout 5 nc 6 y 7 ys 8 yp
    input  wire [4:0]         reg_idx,
    input  wire [31:0]        reg_wdata,
    input  wire               go,           // reg_wdata: [0] int4, [1] int16 weights, [3:2] om (0 S16, 1 RES, 3 ARGMAX), [6] RESACC
    output wire               busy,
    output wire               done,
    output reg  [31:0]        argidx,
    output reg  [31:0]        argval,
    output wire [32*CMAX-1:0] argidx_all,
    output reg                m_en,
    output reg  [2*RWL-1:0]   m_we,
    output reg  [31:0]        m_addr,       // word address
    output reg  [16*RWL-1:0]  m_wdata,
    input  wire [16*RWL-1:0]  m_rdata,
    output wire               seek_valid,
    output wire [31:0]        seek_addr,
    input  wire [8*WB-1:0]    ws_data,
    input  wire [5:0]         ws_avail,
    output reg  [5:0]         ws_take
);
    localparam NWG = L / RWL;                        // RAM words per frame per group
    localparam CB  = $clog2(CMAX + 1);
    localparam LB  = $clog2(L);
    localparam RB  = $clog2(RWL);

    // Clocks (power fixes 1 and 2, reports/p1-power.md section 7). Unit gate: the array is clocked only while busy
    // (S_DONE included), started (go) or written (reg_we); idle with neither, no register here changes. Frame gates:
    // frame f's L accumulators are clocked only in the clocks that write them (S_INIT: all frames; S_RUN: frame c).
    wire gclk;
    clk_gate u_cg (.clk(clk), .en(busy | go | reg_we), .gclk(gclk));

    reg [31:0] r_faddr, r_x, r_xs, r_nin, r_nout, r_nc, r_y, r_ys, r_yp;
    always @(posedge gclk or negedge rst_n) begin
        if (!rst_n) begin
            r_faddr <= 0; r_x <= 0; r_xs <= 0; r_nin <= 0; r_nout <= 0; r_nc <= 0; r_y <= 0; r_ys <= 0; r_yp <= 0;
        end else if (reg_we) begin
            case (reg_idx)
                5'd0: r_faddr <= reg_wdata;
                5'd1: r_x <= reg_wdata;
                5'd2: r_xs <= reg_wdata;
                5'd3: r_nin <= reg_wdata;
                5'd4: r_nout <= reg_wdata;
                5'd5: r_nc <= reg_wdata;
                5'd6: r_y <= reg_wdata;
                5'd7: r_ys <= reg_wdata;
                5'd8: r_yp <= reg_wdata;
                default: ;
            endcase
        end
    end

    localparam [3:0] S_IDLE = 4'd0, S_SEEK = 4'd1, S_HDR = 4'd2, S_INIT = 4'd3, S_XL = 4'd4, S_RUN = 4'd5,
                     S_OLD = 4'd6, S_RQ = 4'd7, S_WR = 4'd8, S_NEXT = 4'd9, S_DONE = 4'd10;
    reg [3:0]  st;
    reg        w4, w16, resacc;
    reg [1:0]  om;
    reg [31:0] gbase;                                // first output row of the group
    reg [31:0] i;                                    // current input
    reg [CB-1:0] c, nc;                              // frame in the MAC loop; frames
    assign busy = (st != S_IDLE);
    assign done = (st == S_DONE);
    assign seek_valid = (st == S_SEEK);
    assign seek_addr  = r_faddr;

    // ---- weight stream: header bytes, then weight vectors (wfill assembles the next one, wcur is in use)
    localparam HB = 8 * L;
    reg  [8*HB-1:0] hbuf;
    reg  [15:0]     hn;
    reg  [16*L-1:0] wfb, wcur;
    reg  [7:0]      wfn;
    reg             wv;
    reg  [31:0]     ifetch;                          // inputs of this group whose weights are fetched (or in wfill)
    wire [7:0]      needb = w4 ? (L / 2) : w16 ? 2 * L : L;
    wire            fetching = (st == S_INIT || st == S_XL || st == S_RUN) && (wfn < needb) && (ifetch < r_nin);
    integer t;
    always @* begin
        ws_take = 6'd0;
        if (st == S_HDR) ws_take = ({10'd0, ws_avail} < HB - hn) ? ws_avail : 6'(HB - hn);
        else if (fetching) ws_take = ({2'd0, ws_avail} < needb - wfn) ? ws_avail : 6'(needb - wfn);
    end
    reg  [16*L-1:0] wfb_n;
    always @* begin
        wfb_n = wfb;
        for (t = 0; t < WB; t = t + 1)
            if (fetching && t < ws_take) wfb_n[8 * (wfn + t) +: 8] = ws_data[8 * t +: 8];
    end
    wire [7:0] wfn_n  = wfn + (fetching ? {2'd0, ws_take} : 8'd0);
    wire       wful_n = (wfn_n == needb);

    // header shift-in (power fix 4, clock margin): the bytes taken this clock enter at the top and the rest move down,
    // so after HB bytes byte p is at p, with no write-position decode from hn
    wire [8*HB-1:0] hsh [0:WB];
    assign hsh[0] = hbuf;
    genvar gh;
    generate for (gh = 1; gh <= WB; gh = gh + 1) begin : g_hs
        assign hsh[gh] = {ws_data[8 * gh - 1:0], hbuf[8 * HB - 1:8 * gh]};
    end endgenerate

    // per-lane header fields
    wire signed [47:0] bias [0:L-1];
    wire        [15:0] mval [0:L-1];
    wire        [5:0]  shv  [0:L-1];
    genvar g;
    generate
        for (g = 0; g < L; g = g + 1) begin : hdr
            assign bias[g] = {{8{hbuf[64 * g + 39]}}, hbuf[64 * g +: 40]};
            assign mval[g] = hbuf[64 * g + 40 +: 16];
            assign shv[g]  = hbuf[64 * g + 56 +: 6];
        end
    endgenerate

    // ---- x registers: 16 inputs per frame
    reg  [16*RWL-1:0] xr [0:CMAX-1];
    reg  [CB-1:0]     xc;                            // next frame to load
    reg               xp;                            // a read is in flight for frame xc - 1
    wire [RB-1:0]     k = i[RB-1:0];
    wire signed [15:0] xv = xr[c][16 * k +: 16];

    // ---- lanes: acc[c][l] for the current frame c. Frame f's accumulators sit behind their own clock gate (fclk[f]):
    // S_INIT loads every frame with the bias, S_RUN adds lane l's product to frame c (the frame being written).
    wire signed [47:0] acc [0:CMAX*L-1];
    function automatic signed [15:0] wsel(input [16*L-1:0] wv_, input integer l, input bit four, input bit sixteen);
        if (four)         wsel = {{12{wv_[4 * l + 3]}}, wv_[4 * l +: 4]};
        else if (sixteen) wsel = wv_[16 * l +: 16];
        else              wsel = {{8{wv_[8 * l + 7]}}, wv_[8 * l +: 8]};
    endfunction
    // MAC pipeline (power fix 4, clock margin): the lane products of S_RUN are registered (fpr, frame pc, valid pv)
    // and added to frame pc's accumulators a clock later. The last product of a group lands in S_OLD or S_XL, before
    // the requantiser or the next products read the accumulators.
    reg  [32*L-1:0]    fpr;
    reg  [CB-1:0]      pc;
    reg                pv;
    wire signed [47:0] fs [0:L-1];                   // lane sums of this clock, for frame pc
    wire [CMAX-1:0]    fclk;
    genvar gl, gf;
    generate
        for (gl = 0; gl < L; gl = gl + 1) begin : g_lp
            assign fs[gl] = acc[pc * L + gl] + $signed(fpr[32 * gl +: 32]);
        end
        for (gf = 0; gf < CMAX; gf = gf + 1) begin : g_fr
            wire fen = (st == S_INIT) || (pv && pc == gf);
            clk_gate u_fg (.clk(gclk), .en(fen), .gclk(fclk[gf]));
            reg [48*L-1:0] fa;
            integer ll;
            always @(posedge fclk[gf] or negedge rst_n) begin
                if (!rst_n) fa <= {48*L{1'b0}};
                else if (st == S_INIT) for (ll = 0; ll < L; ll = ll + 1) fa[48 * ll +: 48] <= bias[ll];
                else for (ll = 0; ll < L; ll = ll + 1) fa[48 * ll +: 48] <= fs[ll];
            end
            for (gl = 0; gl < L; gl = gl + 1) begin : g_la
                assign acc[gf * L + gl] = fa[48 * gl +: 48];
            end
        end
    endgenerate

    // ---- requantiser: one element per clock, two stages
    reg  [CB-1:0]  rc;                               // frame being requantised / written
    reg  [LB:0]    rl;                               // lane fed into stage A
    reg            va, vb;
    reg  [LB-1:0]  la;
    reg  signed [64:0] pa;
    reg  [5:0]     sha;
    reg  [16*L-1:0] olo, ohi;                        // results of frame rc: low and high 16 bits
    reg  [32*L-1:0] old;                             // RESACC: the planes' old values of frame rc
    wire signed [47:0] accq = acc[rc * L + rl[LB-1:0]];
    wire signed [64:0] rnd = (sha == 6'd0) ? 65'sd0 : (65'sd1 <<< (sha - 6'd1));
    wire signed [64:0] rq  = (pa + rnd) >>> sha;     // stage B: round and shift (power fix 4: registered in rqr)
    reg  signed [64:0] rqr;                          // stage C: saturate, RESACC, store (lane lb)
    reg  [LB-1:0]      lb;
    wire signed [64:0] s16 = (rqr > 65'sd32767) ? 65'sd32767 : (rqr < -65'sd32767) ? -65'sd32767 : rqr;
    wire signed [64:0] s25 = (rqr > 65'sd16777215) ? 65'sd16777215 : (rqr < -65'sd16777215) ? -65'sd16777215 : rqr;
    wire signed [31:0] oldv = old[32 * lb +: 32];
    wire signed [64:0] sacc = s25 + {{33{oldv[31]}}, oldv};
    wire signed [64:0] s25a = (sacc > 65'sd16777215) ? 65'sd16777215 : (sacc < -65'sd16777215) ? -65'sd16777215 : sacc;
    wire signed [64:0] vres = resacc ? s25a : s25;
    wire [31:0]        oa  = gbase + {{(31-LB){1'b0}}, lb};
    reg  signed [64:0] best [0:CMAX-1];              // ARGMAX: per frame
    reg  [31:0]        bidx [0:CMAX-1];
    genvar gc;
    generate for (gc = 0; gc < CMAX; gc = gc + 1) begin : argo
        assign argidx_all[32 * gc +: 32] = bidx[gc];
    end endgenerate
    reg  [LB:0]        wc;                           // RAM word counter in S_OLD / S_WR
    reg                wp;                           // S_OLD: a read in flight
    wire [31:0]        frow = r_y + rc * r_ys + gbase;   // first element of frame rc's outputs in this group

    // byte enables of output word w: lanes o < nout
    function automatic [2*RWL-1:0] wmask(input [31:0] o0, input [31:0] nout);
        integer e;
        for (e = 0; e < RWL; e = e + 1) wmask[2 * e +: 2] = (o0 + e < nout) ? 2'b11 : 2'b00;
    endfunction

    always @* begin
        m_en = 1'b0; m_we = {2*RWL{1'b0}}; m_addr = 32'd0; m_wdata = {16*RWL{1'b0}};
        case (st)
            S_XL: if (xc < nc) begin m_en = 1'b1; m_addr = (r_x + xc * r_xs + i) >> RB; end
            S_OLD: if (wc < 2 * NWG) begin
                m_en = 1'b1;
                m_addr = (frow + (wc >= NWG ? r_yp : 32'd0) + RWL * (wc % NWG)) >> RB;
            end
            S_WR: begin
                m_en = 1'b1;
                m_addr = (frow + (wc >= NWG ? r_yp : 32'd0) + RWL * (wc % NWG)) >> RB;
                m_wdata = (wc >= NWG) ? ohi[16 * RWL * (wc % NWG) +: 16 * RWL] : olo[16 * RWL * (wc % NWG) +: 16 * RWL];
                m_we = wmask(gbase + RWL * (wc % NWG), r_nout);
            end
            default: ;
        endcase
    end

    integer e, l;
    always @(posedge gclk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; w4 <= 1'b0; resacc <= 1'b0; om <= 2'd0; gbase <= 0; i <= 0; c <= 0; nc <= 0;
            hbuf <= {8*HB{1'b0}}; hn <= 0; wfb <= {16*L{1'b0}}; wcur <= {16*L{1'b0}}; wfn <= 0; wv <= 1'b0; ifetch <= 0; w16 <= 1'b0;
            xc <= 0; xp <= 1'b0; rc <= 0; rl <= 0; va <= 1'b0; vb <= 1'b0; la <= 0; pa <= 0; sha <= 0;
            olo <= {16*L{1'b0}}; ohi <= {16*L{1'b0}}; old <= {32*L{1'b0}}; wc <= 0; wp <= 1'b0;
            for (e = 0; e < CMAX; e = e + 1) begin best[e] <= 65'sd0; bidx[e] <= 32'd0; end
            argidx <= 0; argval <= 0; fpr <= {32*L{1'b0}}; pc <= 0; pv <= 1'b0; rqr <= 65'sd0; lb <= 0;
            for (e = 0; e < CMAX; e = e + 1) xr[e] <= {16*RWL{1'b0}};
        end else begin
            // MAC pipeline: this clock's lane products, for frame c
            pv <= (st == S_RUN) && wv;
            if ((st == S_RUN) && wv) begin
                pc <= c;
                for (l = 0; l < L; l = l + 1) fpr[32 * l +: 32] <= xv * wsel(wcur, l, w4, w16);
            end
            // weight fill (S_INIT, S_XL, S_RUN)
            if (fetching) begin wfb <= wfb_n; wfn <= wfn_n; if (wful_n) ifetch <= ifetch + 32'd1; end
            case (st)
                S_IDLE: if (go) begin
                    w4 <= reg_wdata[0]; w16 <= reg_wdata[1]; om <= reg_wdata[3:2]; resacc <= reg_wdata[6];
                    gbase <= 32'd0; nc <= r_nc[CB-1:0]; st <= S_SEEK;
                end
                S_SEEK: begin hn <= 0; st <= S_HDR; end
                S_HDR: if (ws_avail != 6'd0) begin
                    hbuf <= hsh[ws_take];                     // shift in (g_hs)
                    hn <= hn + {10'd0, ws_take};
                    if (hn + {10'd0, ws_take} == HB) begin
                        st <= S_INIT; i <= 0; c <= 0; wfn <= 0; wv <= 1'b0; ifetch <= 0;
                    end
                end
                S_INIT: begin                                  // accumulators: g_fr
                    xc <= 0; xp <= 1'b0; st <= S_XL;
                end
                S_XL: begin                                    // C reads, one per clock, captured a clock later
                    if (xp) xr[xc - 1'b1] <= m_rdata;
                    if (xc < nc) begin xc <= xc + 1'b1; xp <= 1'b1; end
                    else begin xp <= 1'b0; st <= S_RUN; c <= 0; end
                    if (!wv && fetching && wful_n) begin wcur <= wfb_n; wv <= 1'b1; wfn <= 0; end
                    else if (!wv && wfn == needb) begin wcur <= wfb; wv <= 1'b1; wfn <= 0; end
                end
                S_RUN: begin
                    if (wv) begin                              // frame c's accumulators: g_fr
                        if (c == nc - 1'b1) begin
                            c <= 0; i <= i + 32'd1;
                            // the next input's weights: complete now (this clock's bytes included) or later
                            if (fetching && wful_n) begin wcur <= wfb_n; wfn <= 0; end
                            else if (!fetching && wfn == needb) begin wcur <= wfb; wfn <= 0; end
                            else wv <= 1'b0;
                            if (i + 32'd1 == r_nin) begin st <= S_OLD; rc <= 0; wc <= 0; wp <= 1'b0; end
                            else if (k == RWL - 1) begin st <= S_XL; xc <= 0; xp <= 1'b0; end
                        end else c <= c + 1'b1;
                    end else if (fetching && wful_n) begin wcur <= wfb_n; wv <= 1'b1; wfn <= 0; end
                    else if (!fetching && wfn == needb) begin wcur <= wfb; wv <= 1'b1; wfn <= 0; end
                end
                S_OLD: begin                                   // RESACC: read frame rc's old values (2 x NWG words)
                    if (!(om == 2'd1 && resacc)) begin st <= S_RQ; rl <= 0; va <= 1'b0; vb <= 1'b0; end
                    else begin
                        if (wp) begin
                            for (e = 0; e < RWL; e = e + 1) begin
                                if (wc - 1'b1 >= NWG) old[32 * (RWL * ((wc - 1'b1) % NWG) + e) + 16 +: 16] <= m_rdata[16 * e +: 16];
                                else                  old[32 * (RWL * ((wc - 1'b1) % NWG) + e) +: 16] <= m_rdata[16 * e +: 16];
                            end
                        end
                        if (wc < 2 * NWG) begin wc <= wc + 1'b1; wp <= 1'b1; end
                        else begin wp <= 1'b0; st <= S_RQ; rl <= 0; va <= 1'b0; vb <= 1'b0; end
                    end
                end
                S_RQ: begin
                    // stage A: product
                    va <= (rl < L);
                    if (rl < L) begin
                        pa <= accq * $signed({1'b0, mval[rl[LB-1:0]]}); sha <= shv[rl[LB-1:0]]; la <= rl[LB-1:0];
                        rl <= rl + 1'b1;
                    end
                    // stage B: round and shift
                    vb <= va;
                    if (va) begin rqr <= rq; lb <= la; end
                    // stage C: saturate, store
                    if (vb) begin
                        if (om == 2'd3) begin
                            if (oa < r_nout && (oa == 32'd0 || rqr > best[rc])) begin
                                best[rc] <= rqr; bidx[rc] <= oa;
                                if (rc == 0) begin argidx <= oa; argval <= rqr[31:0]; end
                            end
                        end else if (om == 2'd1) begin
                            olo[16 * lb +: 16] <= vres[15:0]; ohi[16 * lb +: 16] <= vres[31:16];
                        end else olo[16 * lb +: 16] <= s16[15:0];
                    end
                    if (rl == L && !va && !vb) begin wc <= 0; st <= (om == 2'd3) ? S_NEXT : S_WR; end
                end
                S_WR: begin
                    if (wc == ((om == 2'd1) ? 2 * NWG : NWG) - 1) begin
                        wc <= 0;
                        if (rc == nc - 1'b1) st <= S_NEXT;
                        else begin rc <= rc + 1'b1; st <= S_OLD; wp <= 1'b0; end
                    end else wc <= wc + 1'b1;
                end
                S_NEXT: begin
                    if (om == 2'd3 && rc != nc - 1'b1) begin rc <= rc + 1'b1; st <= S_RQ; rl <= 0; va <= 1'b0; end
                    else if (gbase + L >= r_nout) st <= S_DONE;
                    else begin gbase <= gbase + L; hn <= 0; st <= S_HDR; end
                end
                S_DONE: st <= S_IDLE;
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
