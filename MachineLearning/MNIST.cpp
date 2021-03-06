#include <stdio.h>
#include <stdlib.h>		
#include <string.h>
#include <iostream>
#include "MNIST.h"
#include "layer.h"
#include <time.h>
#include "animations.h"
#include "ProgressBar.h";

#define clear() printf("\033[H\033[J")
#define goto(x,y) printf("\033[%d;%dH", (y), (x))

void delay(float number_of_seconds)
{
	int milli_seconds = (int)(1000.0 * number_of_seconds);
	clock_t start_time = clock();
	while (clock() < start_time + milli_seconds);
}



int main423(int argv, char* argc[]) {
	title_block("MNIST Digit Categorization");
	const int meta_data_size = 16;

	char* file_inputs;
	char* file_labels;
	file_inputs = "data/MNIST/images.bruh";
	file_labels = "data/MNIST/labels.bruh";

	FILE* fp_inputs;
	FILE* fp_labels;
	fp_inputs = fopen(file_inputs, "rb");
	fp_labels = fopen(file_labels, "rb");
	if (!fp_inputs || !fp_labels) {
		printf("Error: File not found or trouble reading the data(training)\n");
		return 0;
	}

	int num_pics = 0;
	int pic_height = 0;
	int	pic_width = 0;
	unsigned char buffer[16];
	fread(buffer, 1, meta_data_size, fp_inputs);
	
	int image_nums[4];
	for (int i = 0; i < 4; i++) {
		int temp = 0;
		for (int j = 0; j < sizeof(int); j++) {
			temp += (buffer[j + (i * 4)] << (sizeof(int) - 1 - j)*8);
		}
		image_nums[i] = temp;
	}
	num_pics = image_nums[1];
	pic_height = image_nums[2];
	pic_width = image_nums[3];

	unsigned char** input_data = (unsigned char**)malloc(num_pics * sizeof(unsigned char *));
	for (int i = 0; i < num_pics; i++) {
		unsigned char* buf = (unsigned char*)malloc(pic_width * pic_height);
		int ret = fread(buf, 1, pic_width * pic_height, fp_inputs);
		if (ferror(fp_inputs)) {
			printf("An error has occured\n");
		}
		input_data[i] = buf;
	}
	
	fclose(fp_inputs);

	//reads the labels from a file
	fread(buffer, 1,  meta_data_size/2, fp_labels);
	int label_nums[2];
	for (int i = 0; i < 2; i++) {
		int temp = 0;
		for (int j = 0; j < sizeof(int); j++) {
			temp += (buffer[j + (i * 4)] << (sizeof(int) - 1 - j) * 8);
		}
		label_nums[i] = temp;
	}
	int num_items = label_nums[1];
	unsigned char* label_data = (unsigned char*)malloc(num_items);
	fread(label_data, 1, num_items, fp_labels);
	fclose(fp_labels);
	printf("\n===============================================================\n");
	//int image_num = 1000;
	//display_image(input_data[image_num], label_data[image_num], 0);
	float** indata = (float**)malloc(sizeof(float*) * num_pics*784);
	for (int i = 0; i < num_pics; i++) {
		float* buffT = (float*)malloc(784 * sizeof(float));
		for (int j = 0; j < 784; j++) {
			if (input_data[i][j] > 50) {
				buffT[j] = 1.0;
			}
			else {
				buffT[j] = 0.0;
			}
			//buffT[j] = input_data[i][j];
			//buffT[j] = (float)input_data[i][j] / 255;
		}
		//display_imageFloat(buffT, 0, 0);
		indata[i] = buffT;
	}


	//NEURAL NETWORK CODE HERE
	//int layerNum = 3;
	//int* layers = (int*)malloc(sizeof(int) * layerNum);
	//layers[0] = 784;
	//layers[1] = 32;
	//layers[2] = 10;
	int layerNum = 4;
	int* layers = new int[4]{784,128,28,10};
	float lRate = .015;
	ActivationFunction *af = new ReLu();
	layer::FullyConnected net(layers, layerNum, lRate, af);
	 

	int numTested = 0;
	int numTestedCorrect = 0;
	ProgressBar pBar;
	for (int i = 0; i < 60000; i++) {
		float* x = indata[i];
		float* y = (float*)malloc(sizeof(float) * 10);
		for (int j = 0; j < 10; j++) {
			y[j] = 0;
		}
		y[label_data[i]] = 1;
		//float* nGuess = (float*)malloc(sizeof(float) * 10);
		//nGuess = net.feedForward(x);
		net.backProp(x, y);
		//int nGuessNum = 0;
		//float runMax = -10;
		//for (int j = 0; j < 10; j++) {
		//	if (nGuess[j] > runMax) {
	    //		runMax = nGuess[j];
		//		nGuessNum = j;
		//	}
		//}
		//gpuMath::blasOp::print_matrix(nGuess, 1, 10);
		//if (nGuessNum == label_data[i]) {
			//std::cout << "Correct" << std::endl;
		//}
		//else {
		//	std::cout << "Wrong" << std::endl;
		//}
		/*if (i > 55000) {
			numTested++;
			if (nGuessNum == label_data[i]) {
				numTestedCorrect++;
			}
			display_imageFloat(x, label_data[i], nGuessNum);
			//gpuMath::blasOp::print_matrix(y, 1, 10);
			//gpuMath::blasOp::print_matrix(nGuess, 1, 10);
			if (i > 20050) {
				std::cout << "Percent correct: " << 100 * (float)numTestedCorrect / (float)numTested << "%" << std::endl;
			}
			delay(0.5);
		}*/
		//gpuMath::blasOp::print_matrix(x, 1, 784);
		//progress_bar(i, 60000, "Training");
		pBar.display(i, 60000, "Training");
		free(input_data[i]);
	}
	pBar.close();
	//std::cout << std::endl;
	free(input_data);
	free(label_data);
	printf("Training complete.\n");

	//Runs the testing of the network and the display
	//opens the files
	char* test_images_file = "data/MNIST/test_images.bruh";
	char* test_labels_file = "data/MNIST/test_labels.bruh";
	FILE* fp_test_images = fopen(test_images_file, "rb");
	FILE* fp_test_labels = fopen(test_labels_file, "rb");
	if (!fp_test_images || !fp_test_labels) {
		printf("Error: File not found or trouble reading the data(training)\n");
		return 0;
	}

	//loads the buffers
	fread(buffer, 1, meta_data_size, fp_test_images);
	image_nums[4];
	for (int i = 0; i < 4; i++) {
		int temp = 0;
		for (int j = 0; j < sizeof(int); j++) {
			temp += (buffer[j + (i * 4)] << (sizeof(int) - 1 - j) * 8);
		}
		image_nums[i] = temp;
	}
	num_pics = image_nums[1];
	unsigned char** test_images = (unsigned char**)malloc(num_pics * sizeof(unsigned char*));
	for (int i = 0; i < num_pics; i++) {
		unsigned char* buf = (unsigned char*)malloc(pic_width * pic_height);
		int ret = fread(buf, 1, pic_width * pic_height, fp_inputs);
		if (ferror(fp_inputs)) {
			printf("An error has occured\n");
		}
		test_images[i] = buf;
	}
	fclose(fp_test_images);

	fread(buffer, 1, meta_data_size / 2, fp_test_labels);
	for (int i = 0; i < 2; i++) {
		int temp = 0;
		for (int j = 0; j < sizeof(int); j++) {
			temp += (buffer[j + (i * 4)] << (sizeof(int) - 1 - j) * 8);
		}
		label_nums[i] = temp;
	}
	num_items = label_nums[1];
	unsigned char* test_labels = (unsigned char*)malloc(num_items);
	fread(test_labels , 1, num_items, fp_test_labels);
	fclose(fp_test_labels);

	clear();

	indata = (float**)malloc(sizeof(float*) * num_pics * 784);
	for (int i = 0; i < num_pics; i++) {
		float* buffT = (float*)malloc(784 * sizeof(float));
		for (int j = 0; j < 784; j++) {
			if (test_images[i][j] > 50) {
				buffT[j] = 1.0;
			}
			else {
				buffT[j] = 0.0;
			}
			//buffT[j] = input_data[i][j];
			//buffT[j] = (float)input_data[i][j] / 255;
		}
		//display_imageFloat(buffT, 0, 0);
		indata[i] = buffT;
	}

	int numCorrect = 0;
	ProgressBar pBar2;
	for (int i = 0; i < num_pics; i++) {
		float* out = net.feedForward(indata[i]);
		int guess = 0;
		float runMax = -10;
		for (int i = 0; i < 10; i++) {
			if (out[i] > runMax) {
				runMax = out[i];
				guess = i;
			}
		}
		if (guess == test_labels[i]) {
			numCorrect++;
		}
		pBar2.display(i, num_pics, "Testing");
		//display_image(test_images[i], test_labels[i], guess);
		//printf("Enter 'q' to quit.\n");
		//char c = getchar();
		//if (c == 'q') {
		//	break;
		//}
	}
	pBar2.close();
	std::cout << "Network Accuracy: " << 100 * ((float)numCorrect / (float)num_pics) << "%" << std::endl;


	//displays answers and network guess
	//YOUR CODE HERE - display_image() can be used

	for (int i = 0; i < num_pics; i++) {
		free(test_images[i]);
	}
	free(test_images);
	free(test_labels);
	return 0;
}
void display_image2(unsigned char* image, int label, int guess) {
	goto(0, 0);
	int count = 0;
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			if (image[count] == 0) {
				printf(" ");
			}
			else if(image[count] < 50/255){
				printf(".");
			}
			else if (image[count] < 150/255) {
				printf("*");
			}
			else {
				printf("X");
			}
			count++;
		}
		printf("|\n|");
	}
	printf("\n\nLabel: %d\n", label);
	printf("Network guess: %d\n", guess);
}

void display_imageFloat(float* image, int label, int guess) {
	goto(0, 0);
	int count = 0;
	for (int i = 0; i < 28; i++) {
		for (int  j = 0; j < 28; j++) {
			if (image[count] == 0) {
				printf(" ");
			}
			else if (image[count] < 50/255) {
				printf(".");
			}
			else if (image[count] < 150/255) {
				printf("*");
			}
			else {
				printf("X");
			}
			count++;
		}
		printf("|\n|");
	}
	printf("\n\nLabel: %d\n", label);
	printf("Network guess: %d\n", guess);
}

