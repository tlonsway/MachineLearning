#include <stdio.h>
#include <stdlib.h>		
#include "graphics.h"

/*
int main(int argv, char* argc[]) {
	const int meta_data_size = 16;

	char* file_inputs;
	char* file_labels;
	if (argv == 3) {
		file_inputs = argc[1];
		file_labels = argc[2];
	}
	else {
		file_inputs = "images.bruh";
		file_labels = "labels.bruh";
	}

	FILE* fp_inputs;
	FILE* fp_labels;
	fp_inputs = fopen(file_inputs, "r");
	fp_labels = fopen(file_labels, "r");
	if (!fp_inputs || !fp_labels) {
		printf("Error: File not found or trouble reading the data(training)\n");
		return 0;
	}
	char buffer[meta_data_size];

	//reads the images from a file
	fgets(buffer, meta_data_size, (FILE*)fp_inputs);
	int num_pics = buffer[4];
	int pic_height = buffer[8];
	int pic_width = buffer[12];
	char** input_data = (char**)malloc(num_pics);
	for (int i = 0; i < num_pics; i++) {
		char* buf = (char*)malloc(pic_width * pic_height);
		fgets(buf, pic_width * pic_height, fp_inputs);
		input_data[i] = buf;
	}
	fclose(fp_inputs);

	//reads the labels from a file
	fgets(buffer, meta_data_size / 2, fp_labels);
	int num_items = buffer[4];
	char* label_data = (char*)malloc(num_items);
	fgets(label_data, num_items, fp_labels);
	fclose(fp_labels);

	//NEURAL NETWORK CODE HERE

	for (int i = 0; i < num_pics; i++) {
		free(input_data[i]);
	}
	free(input_data);
	free(label_data);
	printf("Training complete.\n");

	//Runs the testing of the network and the display
	//opens the files
	char* test_images_file = "";
	char* test_labels_file = "";
	FILE* fp_test_images = fopen(test_images_file, "r");
	FILE* fp_test_labels = fopen(test_labels_file, "r");
	if (!fp_test_images || !fp_test_labels) {
		printf("Error: File not found or trouble reading the data(training)\n");
		return 0;
	}

	//loads the buffers
	fgets(buffer, meta_data_size, fp_test_images);
	int num_test_pics = buffer[4];
	char** test_images = (char**)malloc(num_test_pics);
	for (int i = 0; i < num_test_pics; i++) {
		char* buf = (char *)malloc(pic_height*pic_width);
		fgets(buf, pic_height * pic_width, fp_test_images);
		test_images[i] = buf;
	}
	fgets(buffer, meta_data_size / 2, fp_test_labels);
	int num_test_labels = buffer[4];
	char* test_labels = (char*)malloc(num_test_labels);
	fgets(test_labels, num_test_labels, fp_test_labels);

	//displays answers and network guess
	int i, j = 0, gd = DETECT, gm;
	initgraph(&gd, &gm, "C:\\TC\\BGI");
	settextstyle(DEFAULT_FONT, HORIZ_DIR, 2);
	for (int i = 0; i < num_test_labels; i++) {
		int guess = 0;
		for (int j = 0; j < pic_height * pic_width; j++) {
			int color;
			if (test_images[i][j] == 255) {
				color = WHITE;
			}
			else if(test_images[i][j] == 0){
				color = BLACK;
			}
			else if (test_images[i][j] > 127) {
				color = LIGHTGRAY;
			}
			else {
				color = DARKGRAY;
			}
			setcolor(color);
			setfillstyle(1, color);
			rectangle(i*20, j*20, 20, 20);
			floodfill((i*20) + 1, (j*20) + 1, color);
		}
		outtextxy(50, 600, strcat("Label: ",(char *)&(test_labels[i])));
		outtextxy(50, 630, strcat("Network Guess: ", (char*)&(guess)));
		getch();
	}
	closegraph();

	for (int i = 0; i < num_test_pics; i++) {
		free(test_images[i]);
	}
	free(test_images);
	free(test_labels);
	return 0;
}

*/
