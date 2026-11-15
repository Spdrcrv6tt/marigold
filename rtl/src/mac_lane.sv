// One multiply-accumulate lane: the building block of the MAC array.
//   load: acc <= bias                      (start of a dot product; bias may be 0)
//   en:   acc <= sat(acc + a * w)          a: signed AW-bit activation, w: signed WW-bit weight
//   q = sat_OW(round_half_up(acc >>> shift)), valid combinationally from acc (requantise)
// WW = 8 for int8 weights, 4 for int4. load wins over en. The accumulator saturates instead
// of wrapping, and sat_flag stays high until the next load if it ever did.
// PIPE = 1 registers the product: a term is added one clock after its en, so q is valid one
// clock later too, and a load drops a term still in the pipeline. This halves the logic
// depth and is what the array should use at 50 MHz.
// Reference: rtl/model/mac_model.py.
module mac_lane #(
    parameter AW   = 8,
    parameter WW   = 8,
    parameter ACCW = 24,
    parameter OW   = 8,
    parameter SW   = 5,
    parameter PIPE = 0
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   load,
    input  wire                   en,
    input  wire signed [AW-1:0]   a,
    input  wire signed [WW-1:0]   w,
    input  wire signed [ACCW-1:0] bias,
    input  wire [SW-1:0]          shift,
    output reg  signed [ACCW-1:0] acc,
    output wire signed [OW-1:0]   q,
    output reg                    sat_flag
);
    localparam PW = AW + WW;

    // PIPE 2 (Poinsetta, clock margin): a and w are registered first (a_r, w_r, valid e1), then the product
    reg  signed [AW-1:0]   a_r;
    reg  signed [WW-1:0]   w_r;
    reg                    e1;
    wire signed [PW-1:0]   prod_c = (PIPE >= 2) ? a_r * w_r : a * w;
    wire                   en_p   = (PIPE >= 2) ? e1 : en;
    reg  signed [PW-1:0]   prod_r;
    reg                    en_r;
    wire signed [PW-1:0]   prod  = (PIPE != 0) ? prod_r : prod_c;
    wire                   en_s  = (PIPE != 0) ? en_r : en;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prod_r <= {PW{1'b0}};
            en_r   <= 1'b0;
        end else begin
            en_r <= en_p & ~load;
            if (en_p) prod_r <= prod_c;
        end
    end
    generate if (PIPE >= 2) begin : g_in
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                a_r <= {AW{1'b0}}; w_r <= {WW{1'b0}}; e1 <= 1'b0;
            end else begin
                e1 <= en & ~load;
                if (en) begin a_r <= a; w_r <= w; end
            end
        end
    end endgenerate

    wire signed [ACCW:0]   sum   = {acc[ACCW-1], acc} + {{(ACCW+1-PW){prod[PW-1]}}, prod};
    wire                   s_ovf = sum[ACCW] ^ sum[ACCW-1];
    wire signed [ACCW-1:0] s_sat = s_ovf ? {sum[ACCW], {(ACCW-1){~sum[ACCW]}}} : sum[ACCW-1:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc      <= {ACCW{1'b0}};
            sat_flag <= 1'b0;
        end else if (load) begin
            acc      <= bias;
            sat_flag <= 1'b0;
        end else if (en_s) begin
            acc      <= s_sat;
            sat_flag <= sat_flag | s_ovf;
        end
    end

    // requantise: round half up, arithmetic shift, saturate to OW bits
    wire signed [ACCW:0] one   = {{ACCW{1'b0}}, 1'b1};
    wire signed [ACCW:0] half  = (shift == {SW{1'b0}}) ? {(ACCW+1){1'b0}} : (one <<< (shift - 1'b1));
    wire signed [ACCW:0] accx  = {acc[ACCW-1], acc};
    wire signed [ACCW:0] rnd   = (accx + half) >>> shift;
    wire                 q_ovf = (rnd[ACCW:OW-1] != {(ACCW-OW+2){1'b0}}) && (rnd[ACCW:OW-1] != {(ACCW-OW+2){1'b1}});
    assign q = q_ovf ? {rnd[ACCW], {(OW-1){~rnd[ACCW]}}} : rnd[OW-1:0];
endmodule
