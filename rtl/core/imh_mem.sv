// Memories and weight sources of the Imhotep core. All in-house, all behavioural at this level:
// a physical build replaces each with SRAM / ROM macros of the same interface.

// Activation RAM: WORDS words of LANES x 16 bits, one write enable per byte, registered read.
module imh_ram #(
    parameter LANES = 8,
    parameter BYTES = 1 << 20
) (
    clk, en, we, addr, wdata, rdata
);
    localparam W     = LANES * 16;
    localparam NB    = LANES * 2;
    localparam WORDS = (BYTES + NB - 1) / NB;
    localparam AW    = $clog2(WORDS);
    input  wire          clk;
    input  wire          en;
    input  wire [NB-1:0] we;
    input  wire [31:0]   addr;       // word address
    input  wire [W-1:0]  wdata;
    output reg  [W-1:0]  rdata;

    reg [W-1:0] mem [0:WORDS-1];
    integer i;
    always @(posedge clk) begin
        if (en) begin
            rdata <= mem[addr[AW-1:0]];
            for (i = 0; i < NB; i = i + 1)
                if (we[i]) mem[addr[AW-1:0]][8*i +: 8] <= wdata[8*i +: 8];
        end
    end
endmodule

// Program memory: 32-bit words, written a byte at a time by the boot sequencer, registered read.
module imh_pmem #(
    parameter BYTES = 32768
) (
    input  wire        clk,
    input  wire        en,
    input  wire [31:0] addr,         // byte address
    output reg  [31:0] rdata,
    input  wire        bwe,
    input  wire [31:0] baddr,
    input  wire [7:0]  bdata
);
    localparam WORDS = BYTES / 4;
    localparam AW    = $clog2(WORDS);
    reg [31:0] mem [0:WORDS-1];
    always @(posedge clk) begin
        if (en) rdata <= mem[addr[AW+1:2]];
        if (bwe) mem[baddr[AW+1:2]][8*baddr[1:0] +: 8] <= bdata;
    end
endmodule

// Weight source port (both versions): seek_valid + seek_addr start a sequential read at a byte address and
// drop the read in flight; then `data` carries `avail` valid bytes (the next bytes of the stream, low byte
// first) and the consumer says with `take` how many it used this clock.
//
// On-die ROM version: WB bytes per clock at any byte offset (a physical ROM would be WB byte-banks with a
// rotator). The image is the same file the flash carries (rtl/core/tools/mkimage.py).
module imh_wsrc_rom #(
    parameter BYTES = 1 << 25,
    parameter WB    = 16,
    parameter INIT  = ""
) (
    input  wire            clk,
    input  wire            rst_n,
    input  wire            seek_valid,
    input  wire [31:0]     seek_addr,
    output reg  [8*WB-1:0] data,
    output wire [5:0]      avail,
    input  wire [5:0]      take
);
    localparam AW = $clog2(BYTES);
    reg [7:0] mem [0:BYTES-1];
    reg [AW-1:0] ptr;
    reg          valid;
    initial if (INIT != "") $readmemh(INIT, mem);
    assign avail = valid ? WB : 6'd0;
    wire [AW-1:0] nptr = ptr + take;
    // clocked per read (power fix 5, reports/p1-power.md section 7): a seek, a fill or a take; otherwise nothing here
    // changes, as with a ROM macro's output latch
    wire gclk;
    clk_gate u_cg (.clk(clk), .en(seek_valid || !valid || (take != 6'd0)), .gclk(gclk));
    integer k;
    always @(posedge gclk or negedge rst_n) begin
        if (!rst_n) begin
            ptr <= {AW{1'b0}}; valid <= 1'b0; data <= {(8*WB){1'b0}};
        end else if (seek_valid) begin
            ptr <= seek_addr[AW-1:0]; valid <= 1'b0;
        end else if (!valid) begin
            for (k = 0; k < WB; k = k + 1) data[8*k +: 8] <= mem[ptr + k[AW-1:0]];
            valid <= 1'b1;
        end else if (take != 6'd0) begin
            for (k = 0; k < WB; k = k + 1) data[8*k +: 8] <= mem[nptr + k[AW-1:0]];
            ptr <= nptr;
        end
    end
endmodule

// External NOR flash version: the front-end agent's spi_flash_reader behind the same port, one byte at a time.
// A seek while a read is running stops it (CS# high), waits for the reader to go idle and starts again.
module imh_wsrc_flash #(
    parameter IOW  = 4,
    parameter HALF = 1,
    parameter WB   = 16
) (
    input  wire            clk,
    input  wire            rst_n,
    input  wire [2:0]      mode,
    input  wire [4:0]      dummy,
    input  wire [1:0]      rx_dly,
    input  wire            seek_valid,
    input  wire [31:0]     seek_addr,
    output wire [8*WB-1:0] data,
    output wire [5:0]      avail,
    input  wire [5:0]      take,
    output wire            sck_o,
    output wire            cs_n_o,
    output wire [IOW-1:0]  io_o,
    output wire [IOW-1:0]  io_oe,
    input  wire [IOW-1:0]  io_i
);
    localparam [1:0] S_IDLE = 2'd0, S_STOP = 2'd1, S_START = 2'd2, S_RUN = 2'd3;
    reg [1:0]  st;
    reg [31:0] a;
    wire       busy, out_valid;
    wire [7:0] byte_o;
    wire       data_valid = out_valid && (st == S_RUN) && !seek_valid;
    assign avail = data_valid ? 6'd1 : 6'd0;
    assign data  = {{(8*WB-8){1'b0}}, byte_o};
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; a <= 32'd0;
        end else if (seek_valid) begin
            a  <= seek_addr;
            st <= (st == S_IDLE) ? S_START : S_STOP;
        end else begin
            case (st)
                S_STOP:  if (!busy) st <= S_START;
                S_START: st <= S_RUN;
                default: ;
            endcase
        end
    end
    spi_flash_reader #(.IOW(IOW), .AW(32), .HALF(HALF)) u_rd (
        .clk(clk), .rst_n(rst_n), .start(st == S_START), .stop(st == S_STOP), .addr(a),
        .mode(mode), .dummy(dummy), .rx_dly(rx_dly), .busy(busy),
        .out_data(byte_o), .out_valid(out_valid), .out_ready(data_valid && take != 6'd0),
        .sck_o(sck_o), .cs_n_o(cs_n_o), .io_o(io_o), .io_oe(io_oe), .io_i(io_i));
endmodule
