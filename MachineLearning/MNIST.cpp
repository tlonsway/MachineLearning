#include <stdio.h>
#include <stdlib.h>		
#include <string.h>
#include "MNIST.h"
#define clear() printf("\033[H\033[J")
#define goto(x,y) printf("\033[%d;%dH", (y), (x))
int main(int argv, char* argc[]) {
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
	int image_num = 1000;
	display_image(input_data[image_num], label_data[image_num], 0);
	

	//NEURAL NETWORK CODE HERE

	for (int i = 0; i < num_pics; i++) {
		free(input_data[i]);
	}
	free(input_data);
	free(label_data);
	printf("Training complete.\n");

	//Runs the testing of the network and the display
	//opens the files
	char* test_images_file = "data/MNIST/test_images.bruh";
	char* test_labels_file = "data/MNIST/test_labels.bruh";
	FILE* fp_test_images = fopen(test_images_file, "r");
	FILE* fp_test_labels = fopen(test_labels_file, "r");
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

	//displays answers and network guess
	//YOUR CODE HERE - display_image() can be used

	for (int i = 0; i < num_pics; i++) {
		free(test_images[i]);
	}
	free(test_images);
	free(test_labels);
	return 0;
}
void display_image(unsigned char *image, int label, int guess) {
	clear();
	goto(0, 0);
	int count = 0;
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			if (image[count] == 0) {
				printf(" ");
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
