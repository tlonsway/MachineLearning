#include "ProgressBar.h"
#include <windows.h>

ProgressBar::ProgressBar() {

}
void ProgressBar::display(float position, float end_point, std::string title) {
	int divnum = (int)(end_point / 1000.0);
	if (divnum == 0) {
		divnum = 1;
	}
	if (((int)position) % divnum == 0) {
		HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
		printf("\r");
		for (int i = 0; i < 40; i++) {
			printf(" ");
		}
		printf("\r");
		std::cout << "(" << title << ") ";
		float percent = position / end_point;
		printf("[");
		int tip = 0;
		for (int i = 0; i < 20; i++) {
			SetConsoleTextAttribute(hConsole, ((int)(i / 3) + 9));
			if (i / 20.0 < percent) {
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