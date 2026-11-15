// Clock gate: the enable is latched while clk is low, so gclk is clk with whole pulses
// removed and never glitches. A rising edge of gclk at clock edge E happens exactly when
// en was 1 just before E, i.e. it behaves like a flip-flop enable on everything it clocks.
// Synthesises to a latch and an AND gate (the PDK also has gf180mcu_fd_sc_mcu7t5v0__icgtp_1
// with the same function, which a later pass can substitute).
module clk_gate (
    input  wire clk,
    input  wire en,
    output wire gclk
);
    reg en_l;
    /* verilator lint_off LATCH */
    always @* begin
        if (!clk) en_l = en;
    end
    /* verilator lint_on LATCH */
    assign gclk = clk & en_l;
endmodule
