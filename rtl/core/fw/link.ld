/* Program and constants in PMEM (copied from the image at boot), zero-initialised data at the bottom of the activation RAM. */
MEMORY { PM (rx) : ORIGIN = 0x00000000, LENGTH = 0xFFF8   RAM (rw) : ORIGIN = 0x10000000, LENGTH = 0x10000 }
ENTRY(_start)
SECTIONS {
    .text : { KEEP(*(.init)) *(.text .text.*) *(.rodata .rodata.* .srodata .srodata.*) . = ALIGN(4); } > PM
    .data : { *(.data .data.* .sdata .sdata.*) } > RAM
    .bss (NOLOAD) : { . = ALIGN(4); __bss_start = .; *(.bss .bss.* .sbss .sbss.* COMMON) . = ALIGN(16); _end = .; } > RAM
    ASSERT(SIZEOF(.data) == 0, "initialised data is not supported: make it const or zero")
    /DISCARD/ : { *(.comment) *(.riscv.attributes) *(.eh_frame) *(.note*) }
}
