OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY { OCR (rwx) : ORIGIN = 0xf9000000, LENGTH = 8K }
SECTIONS {
 .text : { KEEP(*(.init)) *(.text .text.*) } > OCR
 .rodata : { *(.rodata .rodata.*) } > OCR
 .data : { . = ALIGN(4); __global_pointer$ = . + 0x800; *(.sdata .sdata.* .data .data.*) } > OCR
 .bss (NOLOAD) : { . = ALIGN(4); __bss_start = .; *(.sbss .sbss.* .bss .bss.* COMMON) . = ALIGN(4); __bss_end = .; } > OCR
 __stack_top = ORIGIN(OCR) + LENGTH(OCR);
 ASSERT(__bss_end + 768 <= __stack_top,"Firmware exceeds OCR with required stack reserve")
 /DISCARD/ : { *(.comment .eh_frame .riscv.attributes) }
}
