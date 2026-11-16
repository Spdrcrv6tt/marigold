// Wake spotter memories (behavioral models).
//   wk_ram: 1R1W synchronous RAM, registered read data (holds when re = 0). Maps to a two-port register
//           file or SRAM macro.
//   wk_rom: the weight ROM (one bit per cell, rtl/wake/mem/wake_rom.mem) behind a request/grant port:
//           rom_req with rom_addr, rom_gnt when accepted, rom_rvalid + rom_rdata the clock after a grant.
//           STALL = 1 inserts pseudo-random grant stalls (LFSR) to exercise the consumer's back-pressure.
module wk_ram #(
    parameter WORDS = 64,
    parameter WIDTH = 32
) (
    input  wire                     clk,
    input  wire                     we,
    input  wire [$clog2(WORDS)-1:0] waddr,
    input  wire [WIDTH-1:0]         wdata,
    input  wire                     re,
    input  wire [$clog2(WORDS)-1:0] raddr,
    output reg  [WIDTH-1:0]         rdata
);
    reg [WIDTH-1:0] mem [0:WORDS-1];
    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        if (re) rdata <= mem[raddr];
    end
endmodule

module wk_rom #(
    parameter WORDS = 17916,
    parameter AW    = 15,
    parameter STALL = 0,
    parameter MEMFILE = "wake_rom.mem"
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          rom_req,
    input  wire [AW-1:0] rom_addr,
    output wire          rom_gnt,
    output reg           rom_rvalid,
    output reg  [15:0]   rom_rdata
);
    reg [15:0] mem [0:WORDS-1];
    initial $readmemh(MEMFILE, mem);
    reg [15:0] lfsr;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) lfsr <= 16'hACE1;
        else        lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
    assign rom_gnt = rom_req & ((STALL == 0) | (lfsr[2:0] != 3'd0) & (lfsr[7:5] != 3'd7));
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rom_rvalid <= 1'b0;
            rom_rdata  <= 16'd0;
        end else begin
            rom_rvalid <= rom_gnt;
            if (rom_gnt) rom_rdata <= mem[rom_addr];
        end
    end
endmodule
