#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include "xgpiops.h"
#include "xparameters.h"
#include "platform.h"
#include "xscugic.h"
#include "ff.h"
#include "xtime_l.h"
#include "xil_io.h"

#define ZERO_POINT_IN 213

static constexpr int NUM_CLASSES = 10;
static constexpr bool REVERSE_CLASS_INDEX = false;
static constexpr int INPUT_DIM = 4096;
static constexpr int MAX_EVENT_FILE_BYTES = 4000000;
static constexpr int MAX_MANIFEST_BYTES = 2000000;
static constexpr int MAX_WEIGHT_FILE_BYTES = 300000;
static constexpr int MAX_BIAS_FILE_BYTES = 4096;
static constexpr int MAX_EVENTS = 200000;
// Training used time_window=100000 (100ms) to filter and normalize events.
// Hardware TIME_WINDOW=200000, so we scale timestamps x2 before sending:
//   t_norm_fpga = (t*2 * 128) / 200000 = t * 128 / 100000  (matches training)
// Only events with t < TRAIN_TIME_WINDOW_US are sent (matching training's filter).
static constexpr u32 TRAIN_TIME_WINDOW_US = 100000;
static constexpr u32 HW_TIME_WINDOW_US    = 200000; // TIME_WINDOW in normalize.sv
static constexpr u32 MAX_TIMESTAMP_US     = TRAIN_TIME_WINDOW_US;

XTime tStart, tEnd;
static FIL fil;
static FATFS fatfs;
const TCHAR *Path = "0:/";

char data_name[128] = "";
char data_name_weights[32] = "mw.txt";
char data_name_bias[32] = "mb.txt";
char data_name_manifest[32] = "manifest.txt";
char destinationAddress[MAX_EVENT_FILE_BYTES + 1];
char manifestBuffer[MAX_MANIFEST_BYTES + 1];
char weightsBuffer[MAX_WEIGHT_FILE_BYTES + 1];
char biasBuffer[MAX_BIAS_FILE_BYTES + 1];

u32 event_data[MAX_EVENTS][4];
int weights_data[INPUT_DIM][NUM_CLASSES];
int bias_data[NUM_CLASSES] = {0};
int features[INPUT_DIM];

u32 data_in = 0;
int mod_cnt = 0;
volatile bool inference_done = false;
volatile int predicted_class = -1;
volatile int current_true_label = -1;

int SDsetup(void);
int loadFile(char *buffer, char *FileName, int nBytes, UINT *NumBytesRead);
int ScuGicInterrupt_Init();
void InterruptHandler(void *data);

XScuGic InterruptController;
static XScuGic_Config *GicConfig;

int main()
{
    init_platform();
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "MNIST-DVS batch inference (fast mode)" << std::endl;

    int xstatus = SDsetup();
    if (xstatus != XST_SUCCESS)
    {
        xil_printf("SD Card Setup Fail\r\n");
        cleanup_platform();
        return 1;
    }
    xil_printf("SD Card Setup Success\r\n");

    xstatus = ScuGicInterrupt_Init();
    if (xstatus != XST_SUCCESS)
    {
        print("GIC Init Fail\r\n");
        cleanup_platform();
        return 1;
    }
    print("GIC Init Success\r\n");

    UINT bytesReadWeights = 0;
    int data = loadFile(weightsBuffer, data_name_weights, MAX_WEIGHT_FILE_BYTES, &bytesReadWeights);
    if (data != XST_SUCCESS)
    {
        std::cout << "Failed to read mw.txt" << std::endl;
        cleanup_platform();
        return 1;
    }
    weightsBuffer[bytesReadWeights] = '\0';

    int expected_weight_vals = INPUT_DIM * NUM_CLASSES;
    int counter = 0;
    char *saveptrW = NULL;
    char *tokenW = strtok_r(weightsBuffer, " \n\r\t", &saveptrW);
    while (tokenW != NULL && counter < expected_weight_vals)
    {
        weights_data[counter / NUM_CLASSES][counter % NUM_CLASSES] = (int)strtol(tokenW, NULL, 10);
        counter += 1;
        tokenW = strtok_r(NULL, " \n\r\t", &saveptrW);
    }

    if (counter != expected_weight_vals)
    {
        std::cout << "Unexpected number of weights in mw.txt: " << counter
                  << " (expected " << expected_weight_vals << ")" << std::endl;
        cleanup_platform();
        return 1;
    }

    UINT bytesReadBias = 0;
    data = loadFile(biasBuffer, data_name_bias, MAX_BIAS_FILE_BYTES, &bytesReadBias);
    if (data == XST_SUCCESS)
    {
        biasBuffer[bytesReadBias] = '\0';
        int bias_count = 0;
        char *saveptrB = NULL;
        char *tokenB = strtok_r(biasBuffer, " \n\r\t", &saveptrB);
        while (tokenB != NULL && bias_count < NUM_CLASSES)
        {
            bias_data[bias_count] = (int)strtol(tokenB, NULL, 10);
            bias_count += 1;
            tokenB = strtok_r(NULL, " \n\r\t", &saveptrB);
        }
        if (bias_count == NUM_CLASSES)
        {
            std::cout << "Loaded mb.txt bias vector (" << bias_count << " classes)" << std::endl;
        }
        else
        {
            for (int i = 0; i < NUM_CLASSES; i++)
            {
                bias_data[i] = 0;
            }
            std::cout << "mb.txt present but invalid; falling back to zero bias" << std::endl;
        }
    }
    else
    {
        std::cout << "mb.txt not found; using zero bias" << std::endl;
    }

    UINT bytesReadManifest = 0;
    data = loadFile(manifestBuffer, data_name_manifest, MAX_MANIFEST_BYTES, &bytesReadManifest);
    if (data != XST_SUCCESS)
    {
        std::cout << "Failed to read manifest.txt" << std::endl;
        cleanup_platform();
        return 1;
    }
    manifestBuffer[bytesReadManifest] = '\0';

    int total = 0;
    int correct = 0;
    int timed_out = 0;
    int class_correct[NUM_CLASSES] = {0};
    int class_total[NUM_CLASSES] = {0};

    XTime batchStart, batchEnd;
    XTime_GetTime(&batchStart);

    char *saveptrLine = NULL;
    char *line = strtok_r(manifestBuffer, "\r\n", &saveptrLine);
    int sample_idx = 0;
    while (line != NULL)
    {
        if (line[0] == '\0')
        {
            line = strtok_r(NULL, "\r\n", &saveptrLine);
            continue;
        }

        int true_label = -1;
        if (sscanf(line, "%127s %d", data_name, &true_label) != 2)
        {
            line = strtok_r(NULL, "\r\n", &saveptrLine);
            continue;
        }

        sample_idx += 1;
        std::cout << "[Sample " << sample_idx << "] Loading " << data_name
                  << " (label=" << true_label << ")" << std::endl;

        UINT bytesReadEvents = 0;
        data = loadFile(destinationAddress, data_name, MAX_EVENT_FILE_BYTES, &bytesReadEvents);
        if (data != XST_SUCCESS)
        {
            std::cout << "failed on event open" << std::endl;
            line = strtok_r(NULL, "\r\n", &saveptrLine);
            continue;
        }
        std::cout << "[Sample " << sample_idx << "] Read OK, bytes=" << bytesReadEvents << std::endl;

        destinationAddress[bytesReadEvents] = '\0';

        inference_done = false;
        predicted_class = -1;
        current_true_label = true_label;
        mod_cnt = 0;

        int tokenCounter = 0;
        char *saveptrE = NULL;
        char *token = strtok_r(destinationAddress, " \n\r\t", &saveptrE);
        while (token != NULL && (tokenCounter / 4) < MAX_EVENTS)
        {
            event_data[tokenCounter / 4][tokenCounter % 4] = (u32)strtoul(token, NULL, 10);
            tokenCounter += 1;
            token = strtok_r(NULL, " \n\r\t", &saveptrE);
        }

        XTime_GetTime(&tStart);
        int event_count = 0;
        int parsed_events = tokenCounter / 4;
        u32 next_timestamp = 0;
        for (int i = 0; i < parsed_events; i++)
        {
            u32 x = event_data[i][0];
            u32 y = event_data[i][1];
            u32 timestamp = event_data[i][2];
            u32 polarity = event_data[i][3];

            if (i < parsed_events - 1)
            {
                next_timestamp = event_data[i + 1][2];
            }
            else
            {
                next_timestamp = timestamp;
            }

            if (timestamp >= TRAIN_TIME_WINDOW_US)
            {
                break;
            }

            u32 valid = 1;
            u32 data_to_send1 = valid + 2 * polarity + 4 * y + 256 * 4 * x;
            // Scale timestamp x2 so hardware t_norm matches training:
            // t_norm = (t*2 * 128) / HW_TIME_WINDOW_US = t * 128 / TRAIN_TIME_WINDOW_US
            u32 data_to_send2 = timestamp * 2;
            Xil_Out32(XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR, data_to_send1);
            Xil_Out32(XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR, data_to_send2);
            event_count += 1;

            u32 wait = (next_timestamp >= timestamp) ? (next_timestamp - timestamp) : 0;
            usleep(wait);
        }

        if (event_count <= 0)
        {
            std::cout << "[Sample " << sample_idx << "] Skipped (no valid events after parse/cutoff)" << std::endl;
            line = strtok_r(NULL, "\r\n", &saveptrLine);
            continue;
        }
        std::cout << "[Sample " << sample_idx << "] Parsed/Sent events=" << event_count << std::endl;

        // Send a sentinel event with scaled timestamp > HW_TIME_WINDOW_US (200000)
        // to trigger the FPGA context reset and output serialization.
        // TRAIN_TIME_WINDOW_US * 2 + 1 = 200001 > 200000 = HW_TIME_WINDOW_US.
        {
            u32 sentinel_data1 = 1; // valid=1, x=0, y=0, polarity=0
            u32 sentinel_data2 = TRAIN_TIME_WINDOW_US * 2 + 1; // 200001 > HW_TIME_WINDOW_US
            Xil_Out32(XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR, sentinel_data1);
            Xil_Out32(XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR, sentinel_data2);
        }

        int timeout_us = 5000000;
        while (!inference_done && timeout_us > 0)
        {
            usleep(100);
            timeout_us -= 100;
        }

        if (!inference_done)
        {
            timed_out += 1;
            std::cout << "[Sample " << sample_idx << "] Inference timeout" << std::endl;
            line = strtok_r(NULL, "\r\n", &saveptrLine);
            continue;
        }

        total += 1;
        class_total[true_label] += 1;
        if (predicted_class == true_label)
        {
            correct += 1;
            class_correct[true_label] += 1;
        }
        std::cout << "[Sample " << sample_idx << "] Inference OK: pred="
                  << predicted_class << ", true=" << true_label << std::endl;

        if ((total % 500) == 0)
        {
            std::cout << "Processed " << total << " samples, accuracy: "
                      << correct << "/" << total << std::endl;
        }

        line = strtok_r(NULL, "\r\n", &saveptrLine);
    }

    XTime_GetTime(&batchEnd);

    std::cout << "==============================================================" << std::endl;
    std::cout << "Batch finished." << std::endl;
    std::cout << "Total inferred: " << total << std::endl;
    std::cout << "Correct: " << correct << std::endl;
    std::cout << "Timed out: " << timed_out << std::endl;
    if (total > 0)
    {
        float accuracy = 100.0f * float(correct) / float(total);
        std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Per-class results:" << std::endl;
    for (int c = 0; c < NUM_CLASSES; c++)
    {
        if (class_total[c] > 0)
        {
            float class_acc = 100.0f * float(class_correct[c]) / float(class_total[c]);
            std::cout << "  Class " << c << ": "
                      << class_correct[c] << "/" << class_total[c]
                      << " (" << class_acc << "%)" << std::endl;
        }
    }

    double elapsed_s = 0.0;
    if (COUNTS_PER_SECOND != 0)
    {
        elapsed_s = double(batchEnd - batchStart) / double(COUNTS_PER_SECOND);
    }
    if (elapsed_s > 0.0)
    {
        std::cout << "Elapsed: " << elapsed_s << " s" << std::endl;
        std::cout << "Throughput: " << (double(total) / elapsed_s) << " samples/s" << std::endl;
    }

    cleanup_platform();
    return 0;
}

int ScuGicInterrupt_Init()
{
    int Status;
    Xil_ExceptionInit();
    GicConfig = XScuGic_LookupConfig(XPAR_SCUGIC_0_DEVICE_ID);
    if (NULL == GicConfig)
    {
        return XST_FAILURE;
    }

    Status = XScuGic_CfgInitialize(&InterruptController, GicConfig, GicConfig->CpuBaseAddress);
    if (Status != XST_SUCCESS)
    {
        return XST_FAILURE;
    }

    Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_IRQ_INT,
                                 (Xil_ExceptionHandler)XScuGic_InterruptHandler,
                                 (void *)&InterruptController);
    Status = XScuGic_Connect(&InterruptController,
                             XPS_FPGA0_INT_ID,
                             (Xil_ExceptionHandler)InterruptHandler,
                             (void *)&InterruptController);
    XScuGic_Enable(&InterruptController, XPS_FPGA0_INT_ID);

    Xil_ExceptionEnable();
    XScuGic_SetPriorityTriggerType(&InterruptController, XPS_FPGA0_INT_ID, 0xa0, 3);

    if (Status != XST_SUCCESS)
    {
        return XST_FAILURE;
    }
    return XST_SUCCESS;
}

void InterruptHandler(void *data)
{
    (void)data;

    for (int i = 0; i < 1024; i++)
    {
        data_in = Xil_In32(XPAR_AXI_BRAM_CTRL_1_S_AXI_BASEADDR + 4 * i);
        features[((i / 64) % 4) * 1024 + (i / 256) * 256 + i % 64 + mod_cnt * 64] = (int)data_in - ZERO_POINT_IN;
    }
    mod_cnt += 1;

    if (mod_cnt == 4)
    {
        mod_cnt = 0;
        int output_vals[NUM_CLASSES] = {0};

        for (int out = 0; out < NUM_CLASSES; out++)
        {
            output_vals[out] = bias_data[out];
        }

        for (int out = 0; out < NUM_CLASSES; out++)
        {
            int sum = output_vals[out];
            for (int w = 0; w < INPUT_DIM; w++)
            {
                sum += weights_data[w][out] * features[w];
            }
            output_vals[out] = sum;
        }

        XTime_GetTime(&tEnd);

        int index = 0;
        int value = output_vals[0];
        for (int i = 1; i < NUM_CLASSES; i++)
        {
            if (output_vals[i] > value)
            {
                value = output_vals[i];
                index = i;
            }
        }

        if (REVERSE_CLASS_INDEX)
        {
            index = (NUM_CLASSES - 1) - index;
        }

        predicted_class = index;
        inference_done = true;
    }
}

int SDsetup(void)
{
    FRESULT res;
    res = f_mount(&fatfs, Path, 0);
    if (res != FR_OK)
    {
        xil_printf("F_MOUNT ERROR CODE: %d; \r\n", res);
        return XST_FAILURE;
    }
    return XST_SUCCESS;
}

int loadFile(char *buffer, char *FileName, int nBytes, UINT *NumBytesRead)
{
    FRESULT Res;
    UINT localNumBytesRead = 0;

    Res = f_open(&fil, FileName, FA_OPEN_EXISTING | FA_READ);
    if (Res)
    {
        std::cout << "Failed on open" << std::endl;
        std::cout << Res << std::endl;
        return XST_FAILURE;
    }

    Res = f_lseek(&fil, 0);
    if (Res)
    {
        std::cout << "Failed on lseek" << std::endl;
        f_close(&fil);
        return XST_FAILURE;
    }

    Res = f_read(&fil, (void *)buffer, nBytes, &localNumBytesRead);
    if (Res)
    {
        std::cout << "Failed on read" << std::endl;
        f_close(&fil);
        return XST_FAILURE;
    }

    if (NumBytesRead != NULL)
    {
        *NumBytesRead = localNumBytesRead;
    }

    Res = f_close(&fil);
    if (Res)
    {
        std::cout << "Failed on close" << std::endl;
        return XST_FAILURE;
    }

    return XST_SUCCESS;
}
