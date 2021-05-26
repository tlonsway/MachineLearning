#include <stdio.h>
#define goto(x,y) printf("\033[%d;%dH", (y), (x))
void display_image(unsigned char* image, int label, int guess) {
	goto(0, 0);
	int count = 0;
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			if (image[count] == 0) {
				printf(" ");
			}
			else if (image[count] < 50) {
				printf(".");
			}
			else if (image[count] < 150) {
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

void progress_bar(float position, float end_point) {
	goto(0,0);
	float percent = position / end_point;
	printf("[");
	int tip = 0;
	for (int i = 0; i < 20; i ++) {
		if (i/20.0 < percent) {
			printf("=");
		}
		else if (!tip) {
			printf(">");
			tip++;
		}
		else {
			printf(" ");
		}
	}
	printf("] %.2f%\n", percent);
}