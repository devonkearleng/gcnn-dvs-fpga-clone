#include <iostream>
#include <cstring>
#include "xgpiops.h"
#include "xparameters.h"
#include "platform.h"
#include <unistd.h>
#include "xscugic.h"
#include "ff.h"
#include "xtime_l.h"
#include "xil_io.h"
#define ZERO_POINT_IN 213

XTime tEnd, tStart;
static FIL fil;
static FATFS fatfs;
const TCHAR *Path = "0:/";
char data_name[128] = "";
char data_name_weights[32] = "mw.txt";
char data_name_manifest[32] = "manifest.txt";
char log_msg[256];
int log_msg_nChars;
int SDsetup(void);
int loadFile(char *buffer, char *FileName, int nBytes, UINT *NumBytesRead);

int ScuGicInterrupt_Init();
void InterruptHandler(void *data);
XScuGic InterruptController;
static XScuGic_Config *GicConfig;

char go;
char mnist_class;
char mnist_sample;
u32 data_in = 0;
const int MAX_EVENT_FILE_BYTES = 4000000;
const int MAX_MANIFEST_BYTES = 2000000;
const int MAX_EVENTS = 200000;
char DestinationAddress[MAX_EVENT_FILE_BYTES + 1];
char ManifestBuffer[MAX_MANIFEST_BYTES + 1];
u32 event_data[MAX_EVENTS][4];
int charsW = 156712;
int weights_data[4096][10];
int features[4096];
int mod_cnt = 0;
bool done = 0;
volatile bool inference_done = false;
volatile int predicted_class = -1;
volatile int current_true_label = -1;

int main()
{
    init_platform();
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "Successfully initialised GCN application\n";

    int xstatus;
	xstatus = SDsetup();
	if (xstatus != XST_SUCCESS)
	{
		xil_printf("SD Card Setup Fail\r\n");
	}
	else
	{
		xil_printf("SD Card Setup Success\r\n");
	}

	xstatus = ScuGicInterrupt_Init();
	if (xstatus != XST_SUCCESS)
	{
		print("GIC Init Fail\r\n");
	}
	print("GIC Init Success\r\n");

	while(1)
	{
		int size = 12279+50;
	std::cout << "Reading weights... ";
	UINT bytesReadWeights = 0;
	int data = loadFile(DestinationAddress, data_name_weights, charsW, &bytesReadWeights);
	if (data != XST_SUCCESS)
	{
		std::cout << "failed!" << std::endl;
		cleanup_platform();
		return 1;
	}
	DestinationAddress[bytesReadWeights] = '\0';
	int counter = 0;
	char *saveptrW = NULL;
	char *tokenW = strtok_r(DestinationAddress, " \n\r\t", &saveptrW);
	while (tokenW != NULL)
	{
		weights_data[int(counter / 10)][int(counter % 10)] = std::__cxx11::stoi(tokenW);
		counter = counter + 1;
		tokenW = strtok_r(NULL, " \n\r\t", &saveptrW);
	}
	std::cout << "done!" << std::endl;

	std::cout << "Reading manifest... ";
	UINT bytesReadManifest = 0;
	data = loadFile(ManifestBuffer, data_name_manifest, MAX_MANIFEST_BYTES, &bytesReadManifest);
	if (data != XST_SUCCESS)
	{
		std::cout << "failed! (expected manifest.txt on SD root)" << std::endl;
		cleanup_platform();
		return 1;
	}
	ManifestBuffer[bytesReadManifest] = '\0';
	std::cout << "done!" << std::endl;

	int total = 0;
	int correct = 0;

	char *saveptrLine = NULL;
	char *line = strtok_r(ManifestBuffer, "\r\n", &saveptrLine);

	while (line != NULL)
		std::cout << "--------------------------------------------------------------" << std::endl;
		if (line[0] != '\0')
			std::cout << "Input file number (0-9): ";
			int true_label = -1;
			if (sscanf(line, "%127s %d", data_name, &true_label) == 2)
				number.append(std::__cxx11::to_string(int(mnist_class)-48));
				std::cout << "--------------------------------------------------------------" << std::endl;
				std::cout << "Processing " << data_name << " (label=" << true_label << ")..." << std::endl;

				UINT bytesRead = 0;
				data = loadFile(DestinationAddress, data_name, MAX_EVENT_FILE_BYTES, &bytesRead);
				if (data != XST_SUCCESS)
					token = strtok(NULL, " \n");
					std::cout << "Failed to read sample file: " << data_name << std::endl;
					line = strtok_r(NULL, "\r\n", &saveptrLine);
					continue;
				size = int(counter / 4);
				DestinationAddress[bytesRead] = '\0';
				dataPtr = 0;
				int tokenCounter = 0;
				char *saveptrE = NULL;
				char *token = strtok_r(DestinationAddress, " \n\r\t", &saveptrE);
				while (token != NULL && int(tokenCounter / 4) < MAX_EVENTS)
				{
					event_data[int(tokenCounter / 4)][int(tokenCounter % 4)] = std::__cxx11::stoi(token);
					tokenCounter = tokenCounter + 1;
					token = strtok_r(NULL, " \n\r\t", &saveptrE);
				}

				int size = int(tokenCounter / 4);
				if (size <= 0)
				{
					std::cout << "No events found in file: " << data_name << std::endl;
					line = strtok_r(NULL, "\r\n", &saveptrLine);
					continue;
				}
				data = loadFile(&dataPtr, data_name_weights, charsW);
				inference_done = false;
				predicted_class = -1;
				current_true_label = true_label;
				mod_cnt = 0;
				u32 next_timestamp = 0;
				u32 x = 0;
				u32 y = 0;
				u32 polarity = 0;
				u32 valid = 0;
				u32 data_to_send1 = 0;
				u32 data_to_send2 = 0;

				XTime_GetTime(&tStart);
				for(int i = 0; i < size; i++)
				{
					timestamp = event_data[i][2];
					if(i < size - 1)
						next_timestamp = event_data[i+1][2];
					x = event_data[i][0];
					y = event_data[i][1];
					polarity = event_data[i][3];
					valid = 1;
					data_to_send1 = valid + 2*polarity + 4*y + 256*4*x;
					data_to_send2 = timestamp;
					Xil_Out32(XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR, data_to_send1);
					Xil_Out32(XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR, data_to_send2);
					wait = next_timestamp - timestamp;
					usleep(wait);
					if(timestamp > 200000)
						break;
				}

				int timeout_us = 5000000;
				while (!inference_done && timeout_us > 0)
				{
					usleep(1000);
					timeout_us -= 1000;
				}

				if (!inference_done)
				{
					std::cout << "Timed out waiting for inference result." << std::endl;
				}
				else
				{
					total += 1;
					if (predicted_class == true_label)
					{
						correct += 1;
					}
					std::cout << "Running accuracy: " << correct << "/" << total << std::endl;
				}
			}
		}
		line = strtok_r(NULL, "\r\n", &saveptrLine);
	}

	std::cout << "==============================================================" << std::endl;
	std::cout << "Dataset inference finished." << std::endl;
	std::cout << "Total samples: " << total << std::endl;
	std::cout << "Correct: " << correct << std::endl;
	if (total > 0)
	{
		float accuracy = 100.0f * float(correct) / float(total);
		std::cout << "Accuracy: " << accuracy << "%" << std::endl;
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

	Status = XScuGic_CfgInitialize(&InterruptController, GicConfig,	GicConfig->CpuBaseAddress);
	if (Status != XST_SUCCESS)
	{
		return XST_FAILURE;
	}

	Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_IRQ_INT, (Xil_ExceptionHandler) XScuGic_InterruptHandler, (void *) &InterruptController);
	Status = XScuGic_Connect(&InterruptController, XPS_FPGA0_INT_ID, (Xil_ExceptionHandler)InterruptHandler, (void *)&InterruptController);
	XScuGic_Enable(&InterruptController, XPS_FPGA0_INT_ID);

	Xil_ExceptionEnable();
	XScuGic_SetPriorityTriggerType(&InterruptController, XPS_FPGA0_INT_ID, 0xa0, 3);

	if (Status != XST_SUCCESS)
	{
		return XST_FAILURE;
	}
	return XST_SUCCESS;
}

void InterruptHandler(void *data) {

	xil_printf("Feature vector %d received... ", mod_cnt+1);
    for(int i = 0; i < 1024; i++)
	{
		data_in = Xil_In32(XPAR_AXI_BRAM_CTRL_1_S_AXI_BASEADDR + 4*i);
		features[int((i/64)%4) * 1024 + int(i/256) * 256 + i%64 + mod_cnt*64] = data_in - ZERO_POINT_IN;
	}
    mod_cnt = mod_cnt + 1;
	std::cout << "done!\n";

	if(mod_cnt == 4)
	{
		mod_cnt = 0;
		int output_dim = 10;
		int input_dim = 4096;
		int output_vals[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

		for(int out = 0; out < output_dim; out++)
		{
			int sum = 0;
			for(int w = 0; w < input_dim; w++)
			{
				sum += weights_data[w][out] * features[w];
			}
			output_vals[out] = sum;
		}

		XTime_GetTime(&tEnd);
		xil_printf("Elapsed time: %d ms (%d ticks)\r\n", (tEnd - tStart)/100000, tEnd - tStart);

		std::cout << "Output result: ";
		for(int i = 9; i >= 0; i--)
		{
			if(i > 0)
				std::cout << output_vals[i] << ", ";
			else
				std::cout << output_vals[i] << std::endl;
		}

		int index = 0;
		int value = -1000000;
		for(int i = 0; i < 10; i++)
		{
			if(output_vals[i] > value)
			{
				value = output_vals[i];
				index = i;
			}
		}
		index = 9-index;
		predicted_class = index;
		inference_done = true;
		std::cout << "True class: " << current_true_label << ", predicted class: " << index << std::endl;
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
		return XST_FAILURE;
	}

	Res = f_lseek(&fil, 0);
	if (Res)
	{
		return XST_FAILURE;
	}

	Res = f_read(&fil, (void*)buffer, nBytes, &localNumBytesRead);
	if (Res)
	{
		return XST_FAILURE;
	}

	if (NumBytesRead != NULL)
	{
		*NumBytesRead = localNumBytesRead;
	}

	Res = f_sync(&fil);
	if (Res)
	{
		return XST_FAILURE;
	}

	Res = f_close(&fil);
	if (Res)
	{
		return XST_FAILURE;
	}

	return XST_SUCCESS;
}
