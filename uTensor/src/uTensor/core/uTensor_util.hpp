#ifndef UTENSOR_UTIL
#define UTENSOR_UTIL
#include <stdint.h>
#include <stdio.h>

//#include <vector>

// #define MAX(A, B) ((A > B)? A:B)


#ifdef ENABLE_PRINTING
#define uTensor_printf(...) printf(__VA_ARGS__)
#else
#define uTensor_printf(...) /*DUMMY*/
#endif

void return_error(int ret_val);
#if MBED_CONF_APP_DEBUG_MSG

// void errno_error(void* ret_val) {
//   if (ret_val == NULL) {
//     printf(" [**Failure**] %d \r\n", errno);
//     printf("Exiting...\r\n");
//     fflush(stdout);
//     exit(-1);
//   } else {
//     printf("  [DONE]\r\n");
//   }
// }

#define ON_ERR(FUNC, MSG) \
  {                       \
    uTensor_printf(" * ");        \
    uTensor_printf(MSG);          \
    return_error(FUNC);   \
  }

#define DEBUG(MSG, ...)         \
  {                             \
    uTensor_printf(MSG, ##__VA_ARGS__); \
    fflush(stdout);             \
  }

#else  // MBED_CONF_APP_DEBUG_MSG

// void errno_error(void* ret_val) { /*DOES NOTHING*/
// }

#define ON_ERR(FUNC, MSG) FUNC
#define DEBUG(MSG, ...)

#endif

void utensor_exit(void);

#define ERR_EXIT(MSG, ...)                                      \
  {                                                             \
    uTensor_printf("[Error] %s:%d @%s ", __FILE__, __LINE__, __func__); \
    uTensor_printf(MSG, ##__VA_ARGS__);                                 \
    fflush(stdout);                                             \
    utensor_exit();                                             \
  }

// typedef std::vector<uint32_t> Shape;

// void printVector(std::vector<uint32_t> vec);
#ifdef TARGET_SIMULATOR
// noop
#elif defined(_WIN32)
// define something for Windows (32-bit and 64-bit, this part is common)
#ifdef _WIN64
// define something for Windows (64-bit only)
#else
// define something for Windows (32-bit only)
#endif
#elif __APPLE__
#include "TargetConditionals.h"
#if TARGET_IPHONE_SIMULATOR
// iOS Simulator
#elif TARGET_OS_IPHONE
// iOS device
#elif TARGET_OS_MAC
// Other kinds of Mac OS
#include <arpa/inet.h>
#include <dirent.h>
#include <sys/stat.h>

#else
#error "Unknown Apple platform"
#endif
#elif __linux__
// linux
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#elif __unix__  // all unices not caught above
// Unix
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#elif defined(_POSIX_VERSION)
// POSIX
#elif ARM_PROJECT
// ARM Project
#else
#if ARDUINO
#include "Arduino.h"

#undef max
#undef min
#undef round
#undef abs
#else
//#include "mbed.h"
#define MBED_PROJECT
#endif
//#   error "Unknown compiler"
// little endian to big endian
// uint32_t htonl(uint32_t& val);
inline uint32_t htonl(uint32_t& val) {
  const uint32_t mask = 0b11111111;
  uint32_t ret = 0;

  ret |= val >> 24;
  ret |= (val & (mask << 16)) >> 8;
  ret |= (val & (mask << 8)) << 8;
  ret |= val << 24;

  return ret;
}
#endif

// big endian to little endian
uint16_t ntoh16(uint16_t val);
uint32_t ntoh32(uint32_t val);

void init_env();
#endif
