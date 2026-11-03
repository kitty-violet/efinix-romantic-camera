/* Short adjacent-integration image only. Includes the actual production C
 * drivers; never a board image or a replacement for its full boot tests. */
#define firmware_main production_main
#include "main.c"
#undef firmware_main
void firmware_main(void) {
 if(io_read(R_ID)!=0x52563233)fail(0xd0);
 if(camera(0x60,0x3107,0xcd,1)!=0xcd)fail(0xc4);
 io_write(R_CAMERA_PUBLISH,1);
 if(transfer(D_SPI_COPY,0x340000,0xe1000200,64)!=0x100ece8c)fail(0x23);
 if(transfer(D_CRC,0xe1000200,0,64)!=0x100ece8c)fail(0x24);
 wr(0xe1000300,0x73b9e12a);
 if(rd(0xe1000300)!=0x73b9e12a || rd(0xe1000200)!=0x03020100)fail(0x26);
 io_write(R_READY,1);io_write(R_FLAGS,3);io_write(R_UI,0x111);
 PLIC(16*4)=1;PLIC(0x2000)=1u<<16;PLIC(0x200000)=0;
 __asm__ volatile("csrs mie,%0;csrs mstatus,%1"::"r"(1u<<11),"r"(8u):"memory");
 while(!io_read(R_IRQ_HANDLED)) { }
 io_write(R_RUN_STATE,0xa5);
 for(;;)__asm__ volatile("nop");
}
