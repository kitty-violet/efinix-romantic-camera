#ifndef RV32_CONTROL_H
#define RV32_CONTROL_H
#include <stdint.h>

enum {
 R_ID=0x00, R_STATUS=0x04, R_EVENTS=0x08, R_EVENT_ACK=0x0c,
 R_NATIVE_ADDRESS=0x10, R_NATIVE_DATA=0x14, R_NATIVE_GO=0x18,
 R_NATIVE_STATUS=0x1c, R_NATIVE_RESULT=0x20,
 R_SPI_ADDRESS=0x24, R_SPI_LENGTH=0x28, R_SPI_GO=0x2c,
 R_SPI_STATUS=0x30, R_SPI_DATA=0x34,
 R_DMA_SOURCE=0x38, R_DMA_TARGET=0x3c, R_DMA_LENGTH=0x40,
 R_DMA_GO=0x44, R_DMA_STATUS=0x48, R_DMA_CRC=0x4c,
 R_CAMERA_WORD=0x50, R_CAMERA_GO=0x54, R_CAMERA_STATUS=0x58,
 R_CAMERA_RESULT=0x5c, R_CAMERA_PUBLISH=0x60,
 R_FLAGS=0x64, R_RUN_STATE=0x68, R_STYLE=0x6c,
 R_FAULT=0x70, R_HEARTBEAT=0x74, R_TIMER=0x78, R_VIEW=0x7c,
 R_ISP_CRC=0x80, R_ISP_CONTROL=0x84, R_ISP_WRITTEN=0x88,
 R_ISP_ACCEPTED=0x8c, R_READY=0x90, R_TARGET_STYLE=0x94,
 R_REQUEST_SEQ=0x98, R_COMPLETED_SEQ=0x9c,
 R_UI=0xa0,R_IRQ_STATUS=0xa4,R_IRQ_ACK=0xa8,R_IRQ_HANDLED=0xac,
 R_CACHE_INVALIDATED=0xb0,R_WARM=0xb4,R_FRAME_BUDGET=0xb8
};
enum { S_BOARD=1, S_DRAINED=2, S_BUS_FAULT=4,S_CAMERA_POWER=8 };
enum { E_TOGGLE=1, E_NEXT=2, E_VIEW=4, E_SELECT=8, E_START=16, E_STOP=32 };
enum { D_SPI_COPY=1, D_DDR_COPY=2, D_CRC=3, D_COEFFICIENTS=4 };
enum { F_PREPARED=1, F_RUNNING=2, F_STOPPED=4, F_DRAINING=8 };
enum { IO_BUSY=1, IO_DONE=2, IO_ERROR=4 };

static inline uint32_t io_read(unsigned offset) {
 return *(volatile uint32_t *)(uintptr_t)(0xf8100000u+offset);
}
static inline void io_write(unsigned offset,uint32_t value) {
 *(volatile uint32_t *)(uintptr_t)(0xf8100000u+offset)=value;
 __asm__ volatile("fence iorw,iorw" ::: "memory");
}
void firmware_main(void);
void firmware_trap(void) __attribute__((interrupt("machine")));
#endif
