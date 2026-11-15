// Imhotep controller: RV32IM, written in-house. Multi-cycle, no CSRs, no interrupts.
//   2 clocks per ALU / branch / jump instruction (execute overlaps the next fetch request),
//   3 for load / store, 5 for multiply, 35 for divide. ebreak / ecall stop the core (halted).
// Bus: `req` is high for one clock with addr / wdata / wstrb (wstrb = 0 is a read); the slave answers with
// one clock of `ready` (and rdata) one or more clocks later. One request is outstanding at a time.
module imh_cpu #(
    parameter [31:0] RESET_PC = 32'h0000_0000
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        run,          // held low until the program is in PMEM
    output reg         req,
    output reg  [31:0] addr,
    output reg  [31:0] wdata,
    output reg  [3:0]  wstrb,
    input  wire [31:0] rdata,
    input  wire        ready,
    output wire        halted
);
    localparam [2:0] S_INIT = 3'd0, S_FETCH = 3'd1, S_EXEC = 3'd2, S_MEM = 3'd3, S_DIV = 3'd4, S_HALT = 3'd5, S_MUL = 3'd6, S_MULW = 3'd7;
    reg [2:0]  st;
    reg [31:0] pc, ir;
    reg [31:0] x [1:31];

    wire [6:0] opc = ir[6:0];
    wire [4:0] rd  = ir[11:7];
    wire [2:0] f3  = ir[14:12];
    wire [4:0] rs1 = ir[19:15];
    wire [4:0] rs2 = ir[24:20];
    wire [6:0] f7  = ir[31:25];
    wire [31:0] a = (rs1 == 5'd0) ? 32'd0 : x[rs1];
    wire [31:0] b = (rs2 == 5'd0) ? 32'd0 : x[rs2];
    wire [31:0] imm_i = {{20{ir[31]}}, ir[31:20]};
    wire [31:0] imm_s = {{20{ir[31]}}, ir[31:25], ir[11:7]};
    wire [31:0] imm_b = {{19{ir[31]}}, ir[31], ir[7], ir[30:25], ir[11:8], 1'b0};
    wire [31:0] imm_u = {ir[31:12], 12'd0};
    wire [31:0] imm_j = {{11{ir[31]}}, ir[31], ir[19:12], ir[20], ir[30:21], 1'b0};

    wire is_lui = (opc == 7'b0110111), is_auipc = (opc == 7'b0010111), is_jal = (opc == 7'b1101111);
    wire is_jalr = (opc == 7'b1100111), is_br = (opc == 7'b1100011), is_load = (opc == 7'b0000011);
    wire is_store = (opc == 7'b0100011), is_opi = (opc == 7'b0010011), is_op = (opc == 7'b0110011);
    wire is_sys = (opc == 7'b1110011), is_fence = (opc == 7'b0001111);
    wire is_m   = is_op && (f7 == 7'b0000001);
    wire is_mul = is_m && !f3[2], is_div = is_m && f3[2];

    // ALU
    wire [31:0] opb  = is_op ? b : imm_i;
    wire        sub  = is_op && f7[5];
    wire [31:0] sum  = sub ? a - b : a + opb;
    wire        lt   = $signed(a) < $signed(opb);
    wire        ltu  = a < opb;
    wire [4:0]  shamt = opb[4:0];
    reg  [31:0] alu;
    always @* begin
        case (f3)
            3'd0: alu = sum;
            3'd1: alu = a << shamt;
            3'd2: alu = {31'd0, lt};
            3'd3: alu = {31'd0, ltu};
            3'd4: alu = a ^ opb;
            3'd5: alu = ir[30] ? $unsigned($signed(a) >>> shamt) : a >> shamt;
            3'd6: alu = a | opb;
            default: alu = a & opb;
        endcase
    end
    // multiplier: 33 x 33 signed covers mul / mulh / mulhsu / mulhu
    wire signed [32:0] ma = {(f3[1:0] != 2'b11) & a[31], a};
    wire signed [32:0] mb = {(f3[1:0] == 2'b01) & b[31], b};
    // built in two clocks from four 17 x 17 products (two multipliers), so that no path holds a 33 x 33 array
    wire signed [16:0] m_al = {1'b0, ma[15:0]}, m_ah = ma[32:16];
    wire signed [16:0] m_bl = {1'b0, mb[15:0]}, m_bh = mb[32:16];
    reg                mph;
    reg  signed [65:0] macc;
    wire signed [33:0] mp0 = (mph ? m_ah : m_al) * m_bl;
    wire signed [33:0] mp1 = (mph ? m_ah : m_al) * m_bh;
    wire signed [65:0] mp0x = {{32{mp0[33]}}, mp0};
    wire signed [65:0] mp1x = {{32{mp1[33]}}, mp1};
    wire signed [65:0] msum = mph ? macc + (mp0x <<< 16) + (mp1x <<< 32) : mp0x + (mp1x <<< 16);
    wire [31:0] mulr = (f3[1:0] == 2'b00) ? macc[31:0] : macc[63:32];

    // branches
    wire eq = (a == b), blt = $signed(a) < $signed(b), bltu = a < b;
    reg  take;
    always @* begin
        case (f3)
            3'd0: take = eq;
            3'd1: take = !eq;
            3'd4: take = blt;
            3'd5: take = !blt;
            3'd6: take = bltu;
            default: take = !bltu;
        endcase
    end
    wire [31:0] pc4 = pc + 32'd4;
    wire [31:0] npc = is_jal ? pc + imm_j : is_jalr ? ((a + imm_i) & ~32'd1) : (is_br && take) ? pc + imm_b : pc4;
    wire [31:0] ex_res = is_lui ? imm_u : is_auipc ? pc + imm_u : (is_jal | is_jalr) ? pc4 : alu;
    wire        ex_wr  = is_lui | is_auipc | is_jal | is_jalr | is_opi | (is_op & !is_m);

    // memory access
    wire [31:0] maddr = a + (is_store ? imm_s : imm_i);
    reg  [1:0]  mlo;
    reg  [31:0] ld;
    always @* begin
        case (f3[1:0])
            2'd0: begin
                case (mlo)
                    2'd0: ld = {{24{rdata[7]  & ~f3[2]}}, rdata[7:0]};
                    2'd1: ld = {{24{rdata[15] & ~f3[2]}}, rdata[15:8]};
                    2'd2: ld = {{24{rdata[23] & ~f3[2]}}, rdata[23:16]};
                    default: ld = {{24{rdata[31] & ~f3[2]}}, rdata[31:24]};
                endcase
            end
            2'd1: ld = mlo[1] ? {{16{rdata[31] & ~f3[2]}}, rdata[31:16]} : {{16{rdata[15] & ~f3[2]}}, rdata[15:0]};
            default: ld = rdata;
        endcase
    end
    wire [3:0]  st_strb = (f3[1:0] == 2'd0) ? (4'b0001 << maddr[1:0]) : (f3[1:0] == 2'd1) ? (maddr[1] ? 4'b1100 : 4'b0011) : 4'b1111;
    wire [31:0] st_data = (f3[1:0] == 2'd0) ? {4{b[7:0]}} : (f3[1:0] == 2'd1) ? {2{b[15:0]}} : b;

    // divider (restoring, one bit per clock)
    reg  [5:0]  dcnt;
    reg  [31:0] dq, dr, dd;
    reg         dneg_q, dneg_r;
    wire [32:0] dsub = {dr[30:0], dq[31]} - {1'b0, dd};
    wire        dsgn = !f3[0];
    wire [31:0] div_q = dneg_q ? -dq : dq;
    wire [31:0] div_r = dneg_r ? -dr : dr;
    wire [31:0] div_res = f3[1] ? div_r : div_q;

    assign halted = (st == S_HALT);

    // Unit clock gate (power fix 1, reports/p1-power.md section 7): no clock while the controller waits for an answer
    // (FETCH or MEM without ready: a unit running, the UART, sleep), before run, or halted. No register changes then,
    // so the gate acts like an enable on every register here.
    wire stall = ((st == S_FETCH || st == S_MEM) && !ready) || (st == S_INIT && !run) || (st == S_HALT);
    wire gclk;
    clk_gate u_cg (.clk(clk), .en(!stall), .gclk(gclk));

    always @* begin
        req = 1'b0; addr = pc; wdata = 32'd0; wstrb = 4'd0;
        case (st)
            S_INIT: if (run) begin req = 1'b1; addr = pc; end
            S_EXEC: begin
                if (is_load | is_store) begin
                    req = 1'b1; addr = {maddr[31:2], 2'b00};
                    wstrb = is_store ? st_strb : 4'd0; wdata = st_data;
                end else if (!is_m && !is_sys) begin
                    req = 1'b1; addr = npc;
                end
            end
            S_MEM: if (ready) begin req = 1'b1; addr = pc4; end
            S_DIV: if (dcnt == 6'd0) begin req = 1'b1; addr = pc4; end
            S_MULW: begin req = 1'b1; addr = pc4; end
            default: ;
        endcase
    end

    integer k;
    always @(posedge gclk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_INIT; pc <= RESET_PC; ir <= 32'h0000_0013; mlo <= 2'd0; mph <= 1'b0; macc <= 66'sd0;
            dcnt <= 6'd0; dq <= 32'd0; dr <= 32'd0; dd <= 32'd0; dneg_q <= 1'b0; dneg_r <= 1'b0;
        end else begin
            case (st)
                S_INIT: if (run) st <= S_FETCH;
                S_FETCH: if (ready) begin ir <= rdata; st <= S_EXEC; end
                S_EXEC: begin
                    if (is_load | is_store) begin
                        mlo <= maddr[1:0]; st <= S_MEM;
                    end else if (is_mul) begin
                        mph <= 1'b0; st <= S_MUL;
                    end else if (is_div) begin
                        dcnt   <= 6'd32;
                        dq     <= (dsgn && a[31]) ? -a : a;
                        dd     <= (dsgn && b[31]) ? -b : b;
                        dr     <= 32'd0;
                        dneg_q <= dsgn && (a[31] ^ b[31]) && (b != 32'd0);
                        dneg_r <= dsgn && a[31];
                        st     <= S_DIV;
                    end else if (is_sys) begin
                        st <= S_HALT;
                    end else begin
                        if (ex_wr && rd != 5'd0) x[rd] <= ex_res;
                        pc <= npc; st <= S_FETCH;
                    end
                end
                S_MUL: begin
                    macc <= msum; mph <= 1'b1;
                    if (mph) st <= S_MULW;
                end
                S_MULW: begin
                    if (rd != 5'd0) x[rd] <= mulr;
                    pc <= pc4; st <= S_FETCH;
                end
                S_MEM: if (ready) begin
                    if (is_load && rd != 5'd0) x[rd] <= ld;
                    pc <= pc4; st <= S_FETCH;
                end
                S_DIV: begin
                    if (dcnt != 6'd0) begin
                        dcnt <= dcnt - 6'd1;
                        if (dsub[32]) begin dr <= {dr[30:0], dq[31]}; dq <= {dq[30:0], 1'b0}; end
                        else          begin dr <= dsub[31:0];        dq <= {dq[30:0], 1'b1}; end
                    end else begin
                        if (rd != 5'd0) x[rd] <= div_res;
                        pc <= pc4; st <= S_FETCH;
                    end
                end
                default: ;
            endcase
        end
    end
endmodule
