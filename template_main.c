#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "cnn.h"
#include "sampledata.h"
#include "sampleoutput.h"
#include "camera.h"
#include "camera_tft_funcs.h"
#include "tft_fthr.h"

volatile uint32_t cnn_time; // Stopwatch
#define TFT_BUFF_SIZE   50    // TFT buffer size
#define CAMERA_FREQ   (10 * 1000 * 1000)
int font_1 = (int)&Arial12x12[0];

/***** Globals *****/
// data is 128 x 128 px = 16,384 px each word is 0RGB, byte for each
uint32_t cnn_buffer[16384];
// buffer for touch screen text
char buff[TFT_BUFF_SIZE];

void load_input(void)
{
  // This function loads the sample data input -- replace with actual data

  int i;
  const uint32_t *in0 = cnn_buffer;

  for (i = 0; i < 16384; i++) {
    while (((*((volatile uint32_t *) 0x50000004) & 1)) != 0); // Wait for FIFO 0
    *((volatile uint32_t *) 0x50000008) = *in0++; // Write FIFO 0
  }
}

// Expected output of layer 11 for simplesortbno given the sample input (known-answer test)
// Delete this function for production code
static const uint32_t sample_output[] = SAMPLE_OUTPUT;
int check_output(void)
{
  int i;
  uint32_t mask, len;
  volatile uint32_t *addr;
  const uint32_t *ptr = sample_output;

  while ((addr = (volatile uint32_t *) *ptr++) != 0) {
    mask = *ptr++;
    len = *ptr++;
    for (i = 0; i < len; i++)
      if ((*addr++ & mask) != *ptr++) {
        printf("Data mismatch (%d/%d) at address 0x%08x: Expected 0x%08x, read 0x%08x.\n",
               i + 1, len, addr - 1, *(ptr - 1), *(addr - 1) & mask);
        return CNN_FAIL;
      }
  }

  return CNN_OK;
}

// Classification layer:
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

void softmax_layer(void)
{
  cnn_unload((uint32_t *) ml_data);
  softmax_q17p14_q15((const q31_t *) ml_data, CNN_NUM_OUTPUTS, ml_softmax);
}

int main(void)
{
  int i;
  int digs, tens;

  MXC_ICC_Enable(MXC_ICC0); // Enable cache

  // Switch to 100 MHz clock
  MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
  SystemCoreClockUpdate();

  printf("Waiting...\n");

  // DO NOT DELETE THIS LINE:
  MXC_Delay(SEC(2)); // Let debugger interrupt if needed


  // Initialize DMA for camera interface
	MXC_DMA_Init();
	int dma_channel = MXC_DMA_AcquireChannel();

  // Initialize TFT display.
  init_LCD();
  MXC_TFT_ClearScreen();

  // Initialize camera.
  printf("Init Camera.\n");
  camera_init(CAMERA_FREQ);
  
  set_image_dimensions(128, 128);

  /* Set the screen rotation because camera flipped*/
	//MXC_TFT_SetRotation(ROTATE_180);

  // Setup the camera image dimensions, pixel format and data acquiring details.
  // four bytes because each pixel is 2 bytes, can get 2 pixels at a time
	int ret = camera_setup(get_image_x(), get_image_y(), PIXFORMAT_RGB565, FIFO_FOUR_BYTE, USE_DMA, dma_channel);
  if (ret != STATUS_OK) 
  {
		printf("Error returned from setting up camera. Error %d\n", ret);
		return -1;
	}
  
  MXC_TFT_SetBackGroundColor(4);
  MXC_TFT_SetForeGroundColor(YELLOW);

  // Enable peripheral, enable CNN interrupt, turn on CNN clock
  // CNN clock: 50 MHz div 1
  cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);

  printf("\n*** CNN Inference Test ***\n");

  cnn_init(); // Bring state machine into consistent state
  cnn_load_weights(); // Load kernels
  cnn_load_bias();
  cnn_configure(); // Configure state machine
  printf("\nready\n");

  // area_t clear_word = {0, 0, 120, 20};
  // area_t clear_time = {0,200,200,20};
  // area_t clear_block = {}
  char* class_names[] = {"CUP", "TRAPEZOID", "HEXAGON", "OTHER","CAN","BOTTLE","NONE"};

  while(true)
  {
    capture_camera_img();
    display_RGB565_img(56,140, cnn_buffer);

    // Enable CNN clock
    MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_CNN);

    cnn_init(); // Bring state machine into consistent state
    cnn_configure(); // Configure state machine

    cnn_start();
    load_input();
    

    while (cnn_time == 0)
    __WFI(); // Wait for CNN

    softmax_layer();

    cnn_stop();
    // Disable CNN clock to save power
    MXC_SYS_ClockDisable(MXC_SYS_PERIPH_CLOCK_CNN);

    #ifdef CNN_INFERENCE_TIMER
    printf("Approximate inference time: %u us\n\n", cnn_time);
    memset(buff,32,TFT_BUFF_SIZE);
    //MXC_TFT_FillRect(&clear_time, 4);
    //TFT_Print(buff, 0, 280, font_1, sprintf(buff, "Inference time: %u us", cnn_time));
    #endif

    printf("Classification results:\n");
    int max = 0;
    int max_i = 0;
    for (i = 0; i < CNN_NUM_OUTPUTS; i++) {
      digs = (1000 * ml_softmax[i] + 0x4000) >> 15;
      tens = digs % 10;
      digs = digs / 10;
      printf("[%7d] -> Class %d: %d.%d%%\n", ml_data[i], i, digs, tens);
      if(digs > max)
      {
          max = digs;
          max_i = i;
      }
      memset(buff,32,TFT_BUFF_SIZE);
      //MXC_TFT_FillRect(&clear_word, 4);
      //TFT_Print(buff, 0, 26+16*i, font_1, sprintf(buff, "%s:%d.%d%%\n", class_names[i], digs, tens));
    }

    memset(buff,32,TFT_BUFF_SIZE);
    //MXC_TFT_FillRect(&clear_word, 4);
    TFT_Print(buff, 0, 0, font_1, sprintf(buff, "Class: %s", class_names[max_i]));
    printf("\033[0;0f");
  }

  return 0;
}