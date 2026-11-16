// Wake spotter keyword scorer (docs/golden-wake.md, section DP; golden sw/wake/golden/golden.py kws()).
// Once per network frame: reads the wake template from config SRAM (byte reader with a one-word cache), runs the
// CTC keyword DP over every variant's 2K+1 states (int16 saturating scores in a 128-word RAM, free start,
// per-frame deficit d = max(z - zmax, -clip)), then decides: fire when the refractory count is zero and the best
// variant's score + K * sens >= 0. A fire resets all states (applied on the next pass) and loads the refractory.
module wk_dp (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               clr,
    input  wire               start,
    input  wire               fresh,         // with start: first frame of a run (states impossible, refractory 0)
    output reg                busy,
    output reg                done,
    output reg                fire,
    input  wire [14:0]        tpl_addr,
    input  wire [15:0]        sens,
    output reg                cfg_req,
    output reg  [14:0]        cfg_addr,
    input  wire               cfg_gnt,
    input  wire [31:0]        cfg_rdata,
    output reg  [5:0]         z_idx,
    input  wire signed [15:0] z_val,
    input  wire signed [15:0] zmax,
    output reg                dbg_v,
    output reg  [31:0]        dbg_d
);
    localparam signed [17:0] NEG = -18'sd32768;
    // state RAM
    reg         a_we, a_re;
    reg  [6:0]  a_wa, a_ra;
    reg  [15:0] a_wd;
    wire [15:0] a_rd;
    wk_ram #(.WORDS(128), .WIDTH(16)) u_a (.clk(clk), .we(a_we), .waddr(a_wa), .wdata(a_wd), .re(a_re), .raddr(a_ra), .rdata(a_rd));

    // byte reader with a one-word cache
    reg [31:0] cw;
    reg [12:0] ci;
    reg        cv;
    reg [7:0]  boff;            // byte offset in the template to read
    reg        breq;            // a cfg request is in flight (waiting for the grant or rvalid)
    reg        bwait;           // grant seen: capture rdata this clock
    wire [14:0] ba  = tpl_addr + {7'd0, boff};
    wire        hit = cv && (ci == ba[14:2]);
    wire [7:0]  byt = cw[8*ba[1:0] +: 8];

    localparam S_IDLE = 0, S_HDR = 1, S_VAR = 2, S_ST0 = 3, S_ST1 = 4, S_NEXT = 5, S_DEC = 6, S_STW = 7;
    reg [2:0]  st;
    reg [1:0]  hp;              // header byte index 1..3
    reg [2:0]  nv, vi;
    reg [7:0]  clip, refr, rcnt;
    reg [7:0]  o;               // offset of the current variant's K byte
    reg [4:0]  K;
    reg [5:0]  s;               // state index 0..2K
    reg [6:0]  bs;              // state RAM base of the variant
    reg        rst_pend, rst_use;
    reg signed [17:0] p1, p2, fa;
    reg [5:0]  lastph, lab;
    reg        skp;
    reg signed [31:0] ks, best;

    wire signed [17:0] old  = rst_use ? NEG : $signed({{2{a_rd[15]}}, a_rd});
    wire signed [17:0] dz0  = $signed({{2{z_val[15]}}, z_val}) - $signed({{2{zmax[15]}}, zmax});
    wire signed [17:0] ncl  = -$signed({10'd0, clip});
    wire signed [17:0] dd   = (dz0 < ncl) ? ncl : dz0;
    wire signed [17:0] c1   = (s >= 6'd1 && p1 > old) ? p1 : old;
    wire signed [17:0] c2   = (skp && p2 > c1) ? p2 : c1;
    wire signed [17:0] c3   = (s <= 6'd1 && c2 < 0) ? 18'sd0 : c2;
    wire signed [17:0] sum  = c3 + dd;
    wire signed [17:0] nw   = (sum < NEG) ? NEG : sum;
    wire signed [17:0] fmx  = (fa > nw) ? fa : nw;
    wire signed [31:0] sv   = $signed({{14{fmx[17]}}, fmx}) + ks;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; busy <= 0; done <= 0; fire <= 0; cfg_req <= 0; cfg_addr <= 0; z_idx <= 0; dbg_v <= 0; dbg_d <= 0;
            a_we <= 0; a_re <= 0; a_wa <= 0; a_ra <= 0; a_wd <= 0; cw <= 0; ci <= 0; cv <= 0; boff <= 0; breq <= 0; bwait <= 0;
            hp <= 0; nv <= 0; vi <= 0; clip <= 0; refr <= 0; rcnt <= 0; o <= 0; K <= 0; s <= 0; bs <= 0;
            rst_pend <= 1; rst_use <= 1; p1 <= NEG; p2 <= NEG; fa <= NEG; lastph <= 0; lab <= 0; skp <= 0; ks <= 0; best <= 0;
        end else begin
            done <= 0; fire <= 0; a_we <= 0; a_re <= 0; dbg_v <= 0;
            // config read handshake (independent of the state machine)
            if (breq && !bwait && cfg_gnt) begin cfg_req <= 0; bwait <= 1; end
            if (bwait) begin cw <= cfg_rdata; ci <= cfg_addr[14:2]; cv <= 1; bwait <= 0; breq <= 0; end
            if (clr) begin
                st <= S_IDLE; busy <= 0; rst_pend <= 1; rcnt <= 0; cv <= 0; cfg_req <= 0; breq <= 0; bwait <= 0;
            end else case (st)
            S_IDLE: if (start) begin
                st <= S_HDR; busy <= 1; hp <= 1; boff <= 8'd1; cv <= 0; rst_use <= rst_pend | fresh;
                if (fresh) rcnt <= 0;
                best <= 32'sh80000000; vi <= 0; o <= 8'd4; bs <= 0;
            end
            // header bytes 1..3: n_var, clip, refractory
            S_HDR: if (hit) begin
                case (hp)
                    2'd1: nv <= byt[2:0];
                    2'd2: clip <= byt;
                    default: refr <= byt;
                endcase
                if (hp == 2'd3) begin st <= S_VAR; boff <= 8'd4; end
                else begin hp <= hp + 1; boff <= boff + 1; end
            end else if (!breq) begin cfg_req <= 1; cfg_addr <= ba; breq <= 1; end
            // variant: read K
            S_VAR: if (hit) begin
                K <= byt[4:0]; s <= 0; p1 <= NEG; p2 <= NEG; fa <= NEG; ks <= 0; lastph <= 0;
                boff <= o + 8'd1;
                st <= S_ST0;
            end else if (!breq) begin cfg_req <= 1; cfg_addr <= ba; breq <= 1; end
            // state s: label (phone byte for odd s), issue the old-score read
            S_ST0: if (!s[0] || hit) begin
                lab <= s[0] ? byt[5:0] : 6'd0;
                skp <= s[0] && (s >= 6'd3) && (byt[5:0] != lastph);
                z_idx <= s[0] ? byt[5:0] : 6'd0;
                if (s[0]) begin lastph <= byt[5:0]; ks <= ks + $signed({16'd0, sens}); end
                a_re <= 1; a_ra <= bs + 7'(s);
                st <= S_STW;
            end else if (!breq) begin cfg_req <= 1; cfg_addr <= ba; breq <= 1; end
            S_STW: st <= S_ST1;                    // read address registered: data arrives next clock
            // compute and write the new score
            S_ST1: begin
                a_we <= 1; a_wa <= bs + 7'(s); a_wd <= nw[15:0];
                p2 <= p1; p1 <= old;
                if (s == {K, 1'b0} - 6'd1) fa <= nw;
                if (s == {K, 1'b0}) begin
                    if (sv > best) best <= sv;
                    st <= S_NEXT;
                end else begin
                    s <= s + 1;
                    if (s[0]) boff <= o + 8'd1 + 8'((s + 6'd1) >> 1);   // next odd state reads the next phone
                    st <= S_ST0;
                end
            end
            S_NEXT: begin
                bs <= bs + 7'({K, 1'b0}) + 7'd1;
                o <= o + 8'd1 + {3'd0, K};
                boff <= o + 8'd1 + {3'd0, K};
                if (vi == nv - 3'd1) st <= S_DEC;
                else begin vi <= vi + 1; st <= S_VAR; end
            end
            S_DEC: begin
                dbg_v <= 1; dbg_d <= best;
                st <= S_IDLE; busy <= 0; done <= 1;
                if (rcnt != 0) begin rcnt <= rcnt - 1; rst_pend <= 0; end
                else if (best >= 0) begin fire <= 1; rcnt <= refr; rst_pend <= 1; end
                else rst_pend <= 0;
            end
            default: st <= S_IDLE;
            endcase
        end
    end
endmodule
