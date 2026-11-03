/* Bare-metal RV32I control-plane owner.  S4R1 arithmetic and per-tile issue
 * remain in the hardware accelerator. No network weights are compiled here. */
#include "control.h"
#include "camera_table.h"

#define APB 0xf8100000u
#define PLIC(o) (*(volatile uint32_t *)(uintptr_t)(0xf8c00000u+(o)))
#define MB  0xf9000f00u
#define FLASH_LIMIT 0x00800000u
#define BUS_TIMEOUT 1000000u
#define LONG_TIMEOUT 2000000000u

static uint32_t directory[144];
static uint32_t selected,count,flags,view,request_seq,isp_ready;
static uint32_t last_count,progress_time;
static uint32_t poll_time,progress_poll_time;
static uint32_t manual_mode,finite_frames,frame_base,resident_valid;
static volatile uint32_t irq_seen,irq_count;

static void heartbeat(void) { io_write(R_HEARTBEAT,1); }
static void fail(uint32_t code) __attribute__((noreturn));
static void fail(uint32_t code) {
 io_write(R_READY,0);
 io_write(R_FLAGS,0);
 io_write(R_FAULT,code);
 for (;;) __asm__ volatile("nop");
}
void firmware_trap(void) {
 uint32_t cause;
 __asm__ volatile("csrr %0,mcause":"=r"(cause));
 if(cause!=0x8000000bu)fail(0xd3);
 uint32_t id=PLIC(0x200004);
 if(id!=16)fail(0xd3);
 /* Mask a level IRQ until the main loop has consumed the queued command.
  * Completion latches are W1C; no event is cleared merely by reading it. */
 __asm__ volatile("csrc mie,%0"::"r"(1u<<11):"memory");
 io_write(R_IRQ_ACK,io_read(R_IRQ_STATUS)&3u);
 irq_seen=1;irq_count++;
 io_write(R_IRQ_HANDLED,irq_count);
 PLIC(0x200004)=id;
}

static uint32_t wait_complete(unsigned reg,uint32_t limit) {
 uint32_t begin=io_read(R_TIMER),status;
 do {
  status=io_read(reg);
  if(status&IO_ERROR) fail(0xd1);
  if((uint32_t)(io_read(R_TIMER)-begin)>limit) fail(0xd2);
 } while(!(status&IO_DONE));
 return status;
}
static void delay(uint32_t cycles) {
 uint32_t t=io_read(R_TIMER);
 while((uint32_t)(io_read(R_TIMER)-t)<cycles) { }
}
static uint32_t native(uint32_t address,uint32_t data,unsigned writing) {
 io_write(R_NATIVE_ADDRESS,address);
 io_write(R_NATIVE_DATA,data);
 io_write(R_NATIVE_GO,writing?2:1);
 wait_complete(R_NATIVE_STATUS,BUS_TIMEOUT);
 return io_read(R_NATIVE_RESULT);
}
static uint32_t rd(uint32_t address) { return native(address,0,0); }
static void wr(uint32_t address,uint32_t data) { (void)native(address,data,1); }

static uint32_t camera(unsigned device,unsigned address,unsigned data,unsigned reading) {
 io_write(R_CAMERA_WORD,(address<<16)|(device<<8)|data);
 io_write(R_CAMERA_GO,reading?2:1);
 wait_complete(R_CAMERA_STATUS,BUS_TIMEOUT);
 return io_read(R_CAMERA_RESULT);
}
static void configure_camera(void) {
 uint32_t device=0x60;
 io_write(R_RUN_STATE,1);
 uint32_t power_wait=io_read(R_TIMER);
 while(!(io_read(R_STATUS)&S_CAMERA_POWER))
  if((uint32_t)(io_read(R_TIMER)-power_wait)>BUS_TIMEOUT)fail(0xc2);
 delay(500000u);  /* 5ms at the fixed 100MHz CPU clock */
 /* A read-only alternate-address probe is permitted on address NACK only.
  * All subsequent errors are terminal; no blind configuration writes. */
 io_write(R_CAMERA_WORD,(0x3107u<<16)|(device<<8)|0xcdu);
 io_write(R_CAMERA_GO,2);
 uint32_t t=io_read(R_TIMER),s;
 do {
  s=io_read(R_CAMERA_STATUS);
  if((uint32_t)(io_read(R_TIMER)-t)>BUS_TIMEOUT)fail(0xc2);
 } while(!(s&(IO_DONE|IO_ERROR)));
 if(s&IO_ERROR) {
  if((io_read(R_CAMERA_RESULT)&0xff)!=0x01)fail(0xc1);
  device=0x64;
  if(camera(device,0x3107,0xcd,1)!=0xcd)fail(0xc4);
 } else if(io_read(R_CAMERA_RESULT)!=0xcd)fail(0xc4);
 if(camera(device,0x3108,0x6b,1)!=0x6b)fail(0xc4);
 for(unsigned i=0;i<CAMERA_CONFIG_COUNT;i++) {
  uint32_t entry=camera_config[i];
  (void)camera(device,entry>>8,entry&255,0);
 }
 for(unsigned i=0;i<CAMERA_VERIFY_COUNT;i++) {
  uint32_t entry=camera_verify[i];
  if((camera(device,entry>>16,(entry>>8)&255,1)&(entry&255))!=((entry>>8)&(entry&255))) fail(0xc5);
 }
 io_write(R_CAMERA_PUBLISH,1);
}

static uint32_t crc_word(uint32_t c,uint32_t w) {
 c^=w;
 for(unsigned i=0;i<32;i++)c=(c>>1)^((0u-(c&1u))&0xedb88320u);
 return c;
}
static uint32_t slot_base(unsigned n) { return 0xe1800000u+(n<<20); }
static uint32_t *entry(unsigned n) { return directory+16+(n<<4); }

static void read_catalog(void) {
 io_write(R_SPI_ADDRESS,0x340000);
 io_write(R_SPI_LENGTH,sizeof(directory));
 io_write(R_SPI_GO,1);
 for(unsigned i=0;i<144;i++) {
  uint32_t start=io_read(R_TIMER);
  while(!(io_read(R_SPI_STATUS)&8)) {
   if(io_read(R_SPI_STATUS)&IO_ERROR)fail(0x12);
   if((uint32_t)(io_read(R_TIMER)-start)>BUS_TIMEOUT)fail(0x12);
  }
  directory[i]=io_read(R_SPI_DATA);
 }
 wait_complete(R_SPI_STATUS,BUS_TIMEOUT);
 count=directory[2];selected=directory[3];
 if(directory[0]!=0x31433153 || directory[1]!=0x20755 || count<1 || count>6 ||
    selected>=count || directory[4]!=576 || directory[6]!=0x3cf000+((count-1)*0x6c000) ||
    directory[7]!=0x341000 || directory[8]!=132432 || directory[10]!=0x362000 || directory[11]!=768)
  fail(0x12);
 for(unsigned i=13;i<16;i++)if(directory[i])fail(0x12);
 uint32_t crc=~0u;
 for(unsigned i=0;i<144;i++)crc=crc_word(crc,i==5?0:directory[i]);
 if((~crc)!=directory[5])fail(0x12);
 for(unsigned n=0;n<8;n++) {
  uint32_t *p=entry(n);
  if(n>=count) { for(unsigned i=0;i<16;i++)if(p[i])fail(0x12);continue; }
  if(p[0]!=1 || !p[1] || p[1]>65535 || p[2]!=0x363000+n*0x6c000 || p[3]!=418944 ||
     p[5]!=p[2]+0x67000 || p[6]!=20480 || p[8]!=slot_base(n) || p[5]+p[6]>FLASH_LIMIT)
   fail(0x12);
  for(unsigned i=9;i<16;i++)if(p[i])fail(0x12);
  for(unsigned j=0;j<n;j++)if(entry(j)[1]==p[1])fail(0x12);
 }
 io_write(R_STYLE,selected|(count<<8));
}

/* All bulk data movement is hardware owned; CPU validates source metadata,
 * starts bounded DMA, and checks complete-transfer CRC and independent readback. */
static uint32_t transfer(unsigned op,uint32_t src,uint32_t dst,uint32_t length) {
 io_write(R_DMA_SOURCE,src);io_write(R_DMA_TARGET,dst);io_write(R_DMA_LENGTH,length);
 io_write(R_DMA_GO,op);
 wait_complete(R_DMA_STATUS,LONG_TIMEOUT);
 heartbeat();
 return io_read(R_DMA_CRC);
}
static void copy_verified(unsigned op,uint32_t src,uint32_t dst,uint32_t bytes,uint32_t expected) {
 if(transfer(op,src,dst,bytes)!=expected)fail(0x23);
 if(transfer(D_CRC,dst,0,bytes)!=expected)fail(0x24);
}
static void preload(void) {
 wr(APB+8,0);wr(APB+0x50,0);wr(APB+0x54,0);wr(MB,0);
 copy_verified(D_SPI_COPY,directory[7],0xe1100000,directory[8],directory[9]);
 copy_verified(D_SPI_COPY,directory[10],0xe1300000,directory[11],directory[12]);
 for(unsigned n=0;n<count;n++) {
  uint32_t *p=entry(n),base=slot_base(n);
  copy_verified(D_SPI_COPY,p[2],base,p[3],p[4]);
  copy_verified(D_SPI_COPY,p[5],base+0xc0000,p[6],p[7]);
 }
}
static void prepare_style(unsigned n) {
 uint32_t *p=entry(n),base=slot_base(n);
 if(n>=count || !resident_valid)fail(0x13);
 io_write(R_READY,0);io_write(R_RUN_STATE,11);
 flags=F_STOPPED;io_write(R_FLAGS,flags);
 /* 222 already fetches descriptors from a verified immutable resident slot.
  * Select it while fully drained, never recopy its 418944-byte program at hot switch.
  * The coefficient engine also reads directly from that resident slot. */
 io_write(R_STYLE,n|(count<<8));
 wr(MB+8,p[1]);wr(MB+12,p[7]);wr(MB+16,p[4]);wr(MB+20,directory[9]);
 wr(APB+8,0);wr(APB+0x10,1);wr(APB+0x14,17792);wr(APB+0x18,p[1]);wr(APB+0x1c,1);
 /* The coefficient DMA fills and independently verifies all five coefficient banks. */
 if(transfer(D_COEFFICIENTS,base+0xc0000,p[1],p[6])!=p[7])fail(0x25);
 wr(MB+28,p[7]);wr(APB+0x48,1);
 if((rd(APB+4)&9)!=1)fail(0x26);
 wr(APB+0x50,p[4]);wr(APB+0x54,directory[9]);
 selected=n;flags=F_PREPARED|F_STOPPED;
 io_write(R_STYLE,selected|(count<<8));
 io_write(R_RUN_STATE,7);io_write(R_FLAGS,flags);io_write(R_READY,isp_ready);
 manual_mode=0;finite_frames=0;io_write(R_FRAME_BUDGET,1023);
}
/* Legacy host uploads remain explicit. An external DDR/metadata write revokes
 * the resident cache; LOAD/CNN1 validate the active buffers before publishing
 * slot7 (the native fetcher's established active-address bypass). */
static void prepare_manual(uint32_t finite) {
 if(flags&(F_RUNNING|F_DRAINING))fail(1);
 uint32_t tag=rd(MB+8),coeff_crc=rd(MB+12),compute_crc=rd(MB+16),dma_crc=rd(MB+20);
 if(!tag || tag>65535 || finite>512)fail(1);
 io_write(R_READY,0);flags=F_STOPPED;io_write(R_FLAGS,flags);
 io_write(R_STYLE,7);io_write(R_RUN_STATE,11);
 if(transfer(D_CRC,0xe1000000,0,418944)!=compute_crc ||
    transfer(D_CRC,0xe1100000,0,132432)!=dma_crc)fail(0x24);
 wr(APB+8,0);wr(APB+0x10,1);wr(APB+0x14,17792);wr(APB+0x18,tag);wr(APB+0x1c,1);
 if(transfer(D_COEFFICIENTS,0xe1200000,tag,20480)!=coeff_crc)fail(0x25);
 wr(MB+28,coeff_crc);wr(APB+0x48,1);
 if((rd(APB+4)&9)!=1)fail(0x26);
 wr(APB+0x50,compute_crc);wr(APB+0x54,dma_crc);
 manual_mode=1;finite_frames=finite;io_write(R_FRAME_BUDGET,finite?finite:1023);
 io_write(R_CACHE_INVALIDATED,1);
 flags=F_PREPARED|F_STOPPED;io_write(R_FLAGS,flags);io_write(R_RUN_STATE,7);io_write(R_READY,1);
}
static void restore_isp(void) {
 uint32_t rejected=io_read(R_ISP_CONTROL)>>16;
 wr(MB+0x28,0x49400001);
 for(unsigned i=0;i<768;i+=4) {
  uint32_t word=rd(0xe1300000+i);
  for(unsigned b=0;b<4;b++)wr(MB+0x28,0x49000000|((i+b)<<8)|((word>>(8*b))&255));
 }
 wr(MB+0x28,0x49800000|(directory[12]&65535));
 wr(MB+0x28,0x49c00000|(directory[12]>>16));
 uint32_t start=io_read(R_TIMER);
 while(((io_read(R_ISP_CONTROL)>>8)&0x1f)!=0x17) {
  if((uint32_t)(io_read(R_TIMER)-start)>BUS_TIMEOUT)fail(0xe2);
 }
 if((io_read(R_ISP_CONTROL)&255) || (io_read(R_ISP_CONTROL)>>16)!=rejected ||
    io_read(R_ISP_ACCEPTED)!=directory[12])fail(0xe2);
 for(unsigned i=0;i<256;i++) {
  uint32_t rgb=0;
  for(unsigned c=0;c<3;c++)rgb|=((rd(0xe1300000+(c<<8)+(i&~3u))>>(8*(i&3)))&255)<<(8*c);
  wr(MB+0x28,0x4a000000|i);
  start=io_read(R_TIMER);
  while(((io_read(R_ISP_CONTROL)>>8)&0x60) || (io_read(R_ISP_WRITTEN)>>24)!=i) {
   if((uint32_t)(io_read(R_TIMER)-start)>BUS_TIMEOUT)fail(0xe2);
  }
  if((io_read(R_ISP_WRITTEN)&0xffffff)!=rgb || (io_read(R_ISP_CONTROL)>>16)!=rejected)fail(0xe2);
 }
}
static void stop(void) {
 if(!(flags&F_RUNNING))return;
 flags=F_PREPARED|F_DRAINING;io_write(R_FLAGS,flags);io_write(R_RUN_STATE,56);
 uint32_t start=io_read(R_TIMER);
 while(!(io_read(R_STATUS)&S_DRAINED)) {
  if((uint32_t)(io_read(R_TIMER)-start)>200000000u)fail(0xd5);
 }
 wr(APB+8,0);if(rd(APB+8))fail(6);
 flags=F_PREPARED|F_STOPPED;io_write(R_FLAGS,flags);io_write(R_RUN_STATE,7);wr(MB+4,5);
 finite_frames=0;io_write(R_FRAME_BUDGET,1023);
 io_write(R_UI,0x100);
}
static void start(void) {
 if(flags&F_RUNNING)return;
 if(!(flags&F_PREPARED))fail(3);
 if((io_read(R_STATUS)&(S_BOARD|S_DRAINED))!=(S_BOARD|S_DRAINED))fail(3);
 last_count=rd(APB+0x5c);progress_time=io_read(R_TIMER);
 progress_poll_time=progress_time;
 frame_base=last_count;
 io_write(R_UI,0x100);
 uint32_t mode=1u|(view<<1);
 wr(APB+8,mode);if(rd(APB+8)!=mode)fail(6);
 flags=F_PREPARED|F_RUNNING;io_write(R_FLAGS,flags);io_write(R_RUN_STATE,47);wr(MB+4,4);
}

void firmware_main(void) {
 if(io_read(R_ID)!=0x52563233)fail(0xd0);
 configure_camera();read_catalog();
 uint32_t start_time=io_read(R_TIMER);
 while((io_read(R_STATUS)&(S_BOARD|S_DRAINED))!=(S_BOARD|S_DRAINED)) {
  if((uint32_t)(io_read(R_TIMER)-start_time)>LONG_TIMEOUT)fail(0xd6);
 }
 preload();resident_valid=1;prepare_style(selected);restore_isp();isp_ready=1;io_write(R_READY,1);
 view=1;io_write(R_VIEW,view);
 io_write(R_UI,0x100);
 PLIC(16*4)=1;PLIC(0x2000)=1u<<16;PLIC(0x200000)=0;
 __asm__ volatile("csrs mie,%0;csrs mstatus,%1"::"r"(1u<<11),"r"(8u):"memory");
 for(;;) {
  /* Frame completions, keys and bus faults interrupt immediately. Legacy
   * host-mailbox/cache polling is bounded at50us at the unchanged CPU100MHz,
   * rather than occupying the shared native bus in every idle iteration.
   * The local timer access does not cross the CPU/service RPC. Modulo-u32
   * subtraction also preserves the schedule across the42.9s timer wrap. */
  uint32_t poll_now=io_read(R_TIMER);
  if(!irq_seen && (uint32_t)(poll_now-poll_time)<5000u)continue;
  unsigned host_due=(uint32_t)(poll_now-poll_time)>=5000u;
  unsigned progress_due=irq_seen ||
   (uint32_t)(poll_now-progress_poll_time)>=100000u;
  if(host_due)poll_time=poll_now;
  if(io_read(R_STATUS)&S_BUS_FAULT)fail(0xd4);
  if(io_read(R_CACHE_INVALIDATED)) {
   resident_valid=0;flags=F_STOPPED;io_write(R_FLAGS,flags);io_write(R_READY,0);
  }
  uint32_t events=io_read(R_EVENTS);
  /* All coarse controls converge here; neither JTAG nor keys own a second
   * task state machine. Mailbox commands are acknowledged before execution. */
  /* A frame/key IRQ does not require another remote host-mailbox read.
   * Keep the legacy host cadence independent of result/keyboard traffic. */
  uint32_t host=host_due?rd(MB):0,host_event=0;
  if(host) {
   wr(MB,0);
   if(host==0x4c495645)host_event=E_START;
   else if(host==0x53544f50)host_event=E_STOP;
   else if(host==0x4c4f4144 || host==0x434e4e31) {
    uint32_t n=host==0x434e4e31?rd(MB+24):0;
    if(host==0x434e4e31 && !n)fail(1);
    prepare_manual(n);
   } else fail(1);
  }
  if(host_event) {
   if(host_event==E_STOP)stop();else start();
  }
  if(events) {
   request_seq=io_read(R_REQUEST_SEQ);
   unsigned target=io_read(R_TARGET_STYLE);
   io_write(R_EVENT_ACK,events);
   if(events&E_VIEW) {
    view=(rd(APB+8)>>1)^1u;view&=1u;
    wr(APB+8,(flags&F_RUNNING?1u:0u)|(view<<1));io_write(R_VIEW,view);
   }
   if(events&E_STOP)stop();
   else if(events&E_START)start();
   else if(events&E_TOGGLE) { if(flags&F_RUNNING)stop();else start(); }
   else if(events&(E_NEXT|E_SELECT)) {
    target=(events&E_SELECT)?target:(selected+1==count?0:selected+1);
    if(resident_valid && !manual_mode && target<count && target!=selected) {
     unsigned resume=flags&F_RUNNING;
     io_write(R_WARM,resume?1:0);
     stop();prepare_style(target);if(resume)start();
     io_write(R_WARM,0);
    }
   }
   io_write(R_COMPLETED_SEQ,request_seq);heartbeat();
  }
  /* Hardware faults still stop issue immediately. Software progress and
   * counter reads need only a 1ms fallback, or the completion IRQ; they do
   * not belong on the shared native bus every 50us host poll. */
  if((flags&F_RUNNING) && progress_due) {
   progress_poll_time=io_read(R_TIMER);
   if(rd(APB+4)&24)fail(6);
   uint32_t completed=rd(APB+0x5c);
   if(completed!=last_count) { last_count=completed;progress_time=io_read(R_TIMER);heartbeat(); }
   if(finite_frames && (uint32_t)(completed-frame_base)>=finite_frames) { stop();continue; }
   if((uint32_t)(io_read(R_TIMER)-progress_time)>200000000u)fail(0xd7);

  }
  if(irq_seen) {
   irq_seen=0;
   __asm__ volatile("csrs mie,%0"::"r"(1u<<11):"memory");
  }
 }
}
