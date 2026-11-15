# Reset entry: stack at the top of the activation RAM, clear .bss, run main, stop.
    .section .init, "ax"
    .globl _start
_start:
    li   t0, 0x20000024
    lw   t1, 0(t0)
    li   sp, 0x10000000
    add  sp, sp, t1
    la   a0, __bss_start
    la   a1, _end
1:  bgeu a0, a1, 2f
    sw   zero, 0(a0)
    addi a0, a0, 4
    j    1b
2:  call main
    ebreak
