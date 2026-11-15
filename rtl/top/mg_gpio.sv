// GPIO rule engine (docs/architecture.md section 8.5), island clock domain. go with an action id (a command's id,
// or 0xF0 / 0xF1 / 0xF2 for WAKE / NOMATCH / TIMEOUT) finds that id in the action table and applies its rules
// gpio_first .. gpio_first + gpio_count - 1 (stopping at n_gpio) in order:
//   mode 0 pulse:  pins of pin_mask to level[0], back to the opposite level after pulse_ms milliseconds;
//   mode 1 latch:  pins to level[0];   mode 2 toggle: pins invert;
//   mode 3 binary: the clamped value of slot `slot` (if the action carries it) on the pins, LSB on the lowest pin,
//                  saturated to [0, 2^k - 1] for k pins;   any mode but 0 cancels a pulse running on its pins.
// Table fields are read a byte at a time (the tables need not be word aligned). A pulse ends on the pulse_ms-th tick
// of a millisecond prescaler (3072 clocks) that runs only while some pulse does, so it lasts pulse_ms ms when it
// starts the prescaler and between pulse_ms - 1 and pulse_ms ms when another pulse is already running.
module mg_gpio #(
    parameter MS = 3072
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         go,
    input  wire [7:0]   id,
    input  wire [7:0]   slot_mask,
    input  wire [255:0] vals,
    input  wire [7:0]   n_actions,
    input  wire [7:0]   n_gpio,
    input  wire [14:0]  act_base,
    input  wire [14:0]  gpio_base,
    output reg          req,
    output reg  [14:0]  addr,
    input  wire         gnt,
    input  wire         rvalid,
    input  wire [31:0]  rdata,
    output wire         busy,
    output reg  [7:0]   gpio_o
);
    localparam G_IDLE = 3'd0, G_FIND = 3'd1, G_RULE = 3'd2, G_APPLY = 3'd3;
    reg  [2:0]  st;
    assign busy = (st != G_IDLE);
    reg  [7:0]  e_id, e_mask;
    reg  [7:0]  j, r, rend;

    // byte fetch: fn bytes little endian from fa into fv
    reg         fbusy;
    reg  [2:0]  fn, fk;
    reg  [1:0]  fl;
    reg  [47:0] fv;
    wire [7:0]  fbyte = fl == 2'd0 ? rdata[7:0] : fl == 2'd1 ? rdata[15:8] : fl == 2'd2 ? rdata[23:16] : rdata[31:24];
    wire [47:0] fv_n  = fv | ({40'd0, fbyte} << {fk, 3'b000});
    wire        fdone = fbusy && rvalid && (fk + 3'd1 == fn);

    // the rule being applied: {pulse_ms, slot, level, mode, pin_mask}
    reg  [7:0]  pm, md, lv, sl;
    reg  [15:0] pms;
    wire [3:0]  npin = pm[0] + pm[1] + pm[2] + pm[3] + pm[4] + pm[5] + pm[6] + pm[7];
    wire signed [31:0] sv = $signed(vals[32*sl[2:0] +: 32]);
    wire [8:0]  vmax = (9'd1 << npin) - 9'd1;
    wire [7:0]  bv = (sv < 0) ? 8'd0 : (sv > $signed({23'd0, vmax})) ? vmax[7:0] : sv[7:0];
    reg  [7:0]  bin;                       // bv spread over the pins of pm
    integer p, c;
    always @(*) begin
        bin = 8'd0; c = 0;
        for (p = 0; p < 8; p = p + 1) if (pm[p]) begin bin[p] = bv[c]; c = c + 1; end
    end

    // pulses
    reg  [7:0]  pact, pend;                // pulse running; level at its end
    reg  [15:0] ptim [0:7];
    reg  [11:0] pre;
    wire        tick = (pre == MS - 1);

    wire gen = go || busy || (|pact);
    wire gclk;
    clk_gate u_cg (.clk(clk), .en(gen), .gclk(gclk));

    integer i;
    always @(posedge gclk or negedge rst_n) begin
        if (!rst_n) begin
            st <= G_IDLE; req <= 1'b0; addr <= 15'd0; gpio_o <= 8'd0; e_id <= 8'd0; e_mask <= 8'd0; j <= 8'd0;
            r <= 8'd0; rend <= 8'd0; fbusy <= 1'b0; fn <= 3'd0; fk <= 3'd0; fl <= 2'd0; fv <= 48'd0;
            pm <= 8'd0; md <= 8'd0; lv <= 8'd0; sl <= 8'd0; pms <= 16'd0; pact <= 8'd0; pend <= 8'd0; pre <= 12'd0;
            for (i = 0; i < 8; i = i + 1) ptim[i] <= 16'd0;
        end else begin
            // millisecond ticks while a pulse runs
            if (|pact) begin
                pre <= tick ? 12'd0 : pre + 12'd1;
                if (tick) for (i = 0; i < 8; i = i + 1)
                    if (pact[i]) begin
                        if (ptim[i] <= 16'd1) begin pact[i] <= 1'b0; gpio_o[i] <= pend[i]; end
                        else ptim[i] <= ptim[i] - 16'd1;
                    end
            end else pre <= 12'd0;
            // byte fetch engine (before the state machine, so a fetch started there wins)
            if (fbusy) begin
                if (gnt) begin req <= 1'b0; fl <= addr[1:0]; end
                if (rvalid) begin
                    fv <= fv_n; fk <= fk + 3'd1;
                    if (fk + 3'd1 == fn) fbusy <= 1'b0;
                    else begin req <= 1'b1; addr <= addr + 15'd1; end
                end
            end
            case (st)
                G_IDLE: if (go) begin st <= G_FIND; e_id <= id; e_mask <= slot_mask; j <= 8'd0; end
                G_FIND: begin
                    if (!fbusy && !fdone) begin
                        if (j == n_actions) st <= G_IDLE;                    // no entry: nothing to do
                        else begin
                            fbusy <= 1'b1; fn <= 3'd4; fk <= 3'd0; fv <= 48'd0; req <= 1'b1;
                            addr <= act_base + {5'd0, j, 2'b00};
                        end
                    end else if (fdone) begin
                        if (fv_n[7:0] == e_id) begin
                            r <= fv_n[23:16]; rend <= fv_n[23:16] + fv_n[31:24]; st <= G_RULE;
                        end else j <= j + 8'd1;
                    end
                end
                G_RULE: begin
                    if (!fbusy && !fdone) begin
                        if (r == rend || r >= n_gpio) st <= G_IDLE;
                        else begin
                            fbusy <= 1'b1; fn <= 3'd6; fk <= 3'd0; fv <= 48'd0; req <= 1'b1;
                            addr <= gpio_base + {4'd0, r, 3'b000};
                        end
                    end else if (fdone) begin
                        pm <= fv_n[7:0]; md <= fv_n[15:8]; lv <= fv_n[23:16]; sl <= fv_n[31:24]; pms <= fv_n[47:32];
                        st <= G_APPLY;
                    end
                end
                G_APPLY: begin
                    for (i = 0; i < 8; i = i + 1) if (pm[i]) begin
                        if (md == 8'd0) begin
                            gpio_o[i] <= lv[0]; pact[i] <= 1'b1; pend[i] <= ~lv[0]; ptim[i] <= pms;
                        end else begin
                            pact[i] <= 1'b0;
                            if (md == 8'd1) gpio_o[i] <= lv[0];
                            else if (md == 8'd2) gpio_o[i] <= ~gpio_o[i];
                            else if (md == 8'd3 && sl < 8'd8 && e_mask[sl[2:0]]) gpio_o[i] <= bin[i];
                        end
                    end
                    if (md == 8'd0 && !(|pact)) pre <= 12'd0;                // a first pulse starts the prescaler
                    r <= r + 8'd1; st <= G_RULE;
                end
                default: st <= G_IDLE;
            endcase
        end
    end
endmodule
