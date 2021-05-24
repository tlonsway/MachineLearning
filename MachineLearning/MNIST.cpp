#include <stdio.h>
#include <stdlib.h>

int main(int argv, char *argc[]) {
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

	FILE *fp_inputs;
	FILE *fp_labels;
	fp_inputs = fopen(file_inputs, "r");
	fp_labels = fopen(file_labels, "r");
	if (!fp_inputs || !fp_labels) {
		printf("Error: File not found or trouble reading the data\n");
		return 0;
	}
	char buffer[meta_data_size];

	//reads the images from a file
	fgets(buffer, meta_data_size, (FILE *)fp_inputs);
	int num_pics = buffer[4];
	int pic_height = buffer[8];
	int pic_width = buffer[12];
	char **input_data = (char **)malloc(num_pics);
	for (int i = 0; i < num_pics; i++) {
		char* buf = (char *)malloc(pic_width * pic_height);
		fgets(buf, pic_width * pic_height, fp_inputs);
		input_data[i] = buf;
	}
	fclose(fp_inputs);

	//reads the labels from a file
	fgets(buffer, meta_data_size/2,fp_labels);
	int num_items = buffer[4];
	char* label_data = (char*)malloc(num_items);
	fgets(label_data, num_items, fp_labels);
	fclose(fp_labels);
	
	/*
	 YOUR CODE HERE
	*/

	for (int i = 0; i < num_pics; i++) {
		free(input_data[i]);
	}
	free(input_data);
	free(label_data);
	return 0;
}

