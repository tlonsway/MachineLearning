#include <stdio.h>
#include <iostream>
#include <windows.h>

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

void progress_bar(float position, float end_point, std::string title) {
	int divnum = (int)(end_point / 1000.0);
	if (divnum == 0) {
		divnum = 1;
	}
	if (((int)position) % divnum == 0) {
		HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
		printf("\r                                                  \r");
		//goto(0,0);
		//printf("(%s) ", title);
		std::cout << "(" << title << ") ";
		float percent = position / end_point;
		printf("[");
		int tip = 0;
		//SetConsoleTextAttribute(hConsole, FOREGROUND_GREEN);
		for (int i = 0; i < 20; i++) {
			SetConsoleTextAttribute(hConsole, ((int)(i / 3) + 9));
			if (i / 20.0 < percent) {
				//printf("=");
				printf("%c", (char)(219));
			}
			else if (!tip) {
				printf(">");
				tip++;
			}
			else {
				SetConsoleTextAttribute(hConsole, 15);
				printf(" ");
			}
		}
		SetConsoleTextAttribute(hConsole, 15);
		printf("] %.2f%%", percent * 100);
	}
}