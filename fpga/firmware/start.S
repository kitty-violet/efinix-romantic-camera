.section .init,"ax",@progbits
.globl _start
_start:
 .option push
 .option norelax
 la gp,__global_pointer$
 .option pop
 la sp,__stack_top
 la t0,firmware_trap
 csrw mtvec,t0
 csrw mie,zero
 csrw mstatus,zero
 la t0,__bss_start
 la t1,__bss_end
1:
 bgeu t0,t1,2f
 sw zero,0(t0)
 addi t0,t0,4
 j 1b
2:
 call firmware_main
 call firmware_trap
3:
 j 3b
