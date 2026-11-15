// CIC decimator for a 1-bit PDM stream (bit 1 = +1, bit 0 = -1).
// N registered integrators at the bit rate, decimate by R, N combs (differential delay 1).
// W must hold R^N: W >= ceil(N*log2(R)) + 1. Arithmetic wraps, which is correct for a CIC.
// Two gated clocks: the integrators only run when a bit arrives, the decimated side (v, y, d)
// only while a decimated sample is rippling through the combs.
// Reference: rtl/model/pdm_model.py cic().
module cic_decim #(
    parameter R = 48,
    parameter N = 5,
    parameter W = 29
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                in_valid,
    input  wire                in_bit,
    output wire                out_valid,
    output wire signed [W-1:0] out_data
);
    localparam CW = $clog2(R);

    reg signed [W-1:0] integ [0:N-1];
    reg signed [W-1:0] y     [0:N-1];
    reg signed [W-1:0] d     [0:N-1];
    reg signed [W-1:0] dec;
    reg [N:0]          v;
    reg [CW-1:0]       cnt;
    integer k;

    wire trig = in_valid && (cnt == R - 1);
    wire gclk_i, gclk_c;
    clk_gate u_cgi (.clk(clk), .en(in_valid),   .gclk(gclk_i));
    clk_gate u_cgc (.clk(clk), .en(trig | (|v)), .gclk(gclk_c));

    always @(posedge gclk_i or negedge rst_n) begin
        if (!rst_n) begin
            for (k = 0; k < N; k = k + 1) integ[k] <= {W{1'b0}};
            dec <= {W{1'b0}};
            cnt <= {CW{1'b0}};
        end else begin
            integ[0] <= integ[0] + (in_bit ? {{(W-1){1'b0}}, 1'b1} : {W{1'b1}});
            for (k = 1; k < N; k = k + 1)
                integ[k] <= integ[k] + integ[k-1];
            if (cnt == R - 1) begin
                cnt <= {CW{1'b0}};
                dec <= integ[N-1];
            end else begin
                cnt <= cnt + 1'b1;
            end
        end
    end

    always @(posedge gclk_c or negedge rst_n) begin
        if (!rst_n) begin
            for (k = 0; k < N; k = k + 1) begin
                y[k] <= {W{1'b0}};
                d[k] <= {W{1'b0}};
            end
            v <= {(N+1){1'b0}};
        end else begin
            v <= {v[N-1:0], trig};
            // combs ripple through one stage per clock after each decimated sample
            for (k = 0; k < N; k = k + 1) begin
                if (v[k]) begin
                    if (k == 0) begin
                        y[k] <= dec - d[k];
                        d[k] <= dec;
                    end else begin
                        y[k] <= y[k-1] - d[k];
                        d[k] <= y[k-1];
                    end
                end
            end
        end
    end

    assign out_valid = v[N];
    assign out_data  = y[N-1];
endmodule
