// UART transmitter, 8N1, LSB first. Bit time = (div + 1) clk cycles (50 MHz / 115200 -> div = 433).
// in_ready is high when a byte can be taken; back-to-back bytes are one stop bit plus one clk apart.
module uart_tx #(
    parameter DW = 16
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire [DW-1:0] div,
    input  wire [7:0]    in_data,
    input  wire          in_valid,
    output wire          in_ready,
    output reg           tx_o
);
    reg [DW-1:0] cnt;
    reg [3:0]    nbit;       // bits left to send including start and stop; 0 = idle
    reg [8:0]    sh;

    assign in_ready = (nbit == 4'd0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt  <= {DW{1'b0}};
            nbit <= 4'd0;
            sh   <= 9'h1FF;
            tx_o <= 1'b1;
        end else if (nbit == 4'd0) begin
            tx_o <= 1'b1;
            if (in_valid) begin
                sh   <= {1'b1, in_data};     // data then stop bit
                tx_o <= 1'b0;                // start bit
                nbit <= 4'd10;
                cnt  <= div;
            end
        end else if (cnt == {DW{1'b0}}) begin
            cnt  <= div;
            nbit <= nbit - 4'd1;
            if (nbit != 4'd1) begin
                tx_o <= sh[0];
                sh   <= {1'b1, sh[8:1]};
            end
        end else begin
            cnt <= cnt - 1'b1;
        end
    end
endmodule
