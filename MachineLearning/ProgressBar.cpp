#include "ProgressBar.h"
#include <windows.h>
#include <time.h>
#include <iostream>
#include <string>

using namespace std;

ProgressBar::ProgressBar() {
	startTime = clock();
	hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	colorOrder = new int[7]{ 12,14,10,11,9,13,5};
	lastTime = 0;
	printf("\r                                                                                                                                                                                                                      \r");
}

string getTimeStringFromMillis(long m) {
	long seconds = (long)(m / 1000) % 60;
	long minutes = (long)(m / (1000 * 60) % 60);
	long hours = (long)(m / (1000 * 60 * 60) % 24);
	string ret = "[Hours: " + to_string(hours) + "; Minutes: " + to_string(minutes) + "; Seconds: " + to_string(seconds) + "]";
	return ret;
}

void ProgressBar::display(float position, float end_point, std::string title) {
	int divnum = (int)(end_point / 1000.0);
	if (divnum == 0) {
		divnum = 1;
	}
	
	//if (((int)position) % divnum == 0 && clock()-lastTime >= 500) {
	if (clock() - lastTime >= 500) {
		lastTime = clock();
		long tsofar = clock() - startTime;
		long totalExpectedMill = ((float)tsofar/(float)position)*(float)(end_point-position);
		string tStr = getTimeStringFromMillis(totalExpectedMill);
		string lStr = "\r                                                                                                   \r";
		lStr = lStr + "(" + title + ") ";
		float percent = position / end_point;
		lStr = lStr + "[";
		SetConsoleTextAttribute(hConsole, 6 + (16 * 0));
		cout << lStr;
		int tip = 0;
		//SetConsoleTextAttribute(hConsole, FOREGROUND_GREEN);
		for (int i = 0; i < 20; i++) {
			//SetConsoleTextAttribute(hConsole, ((int)(i / 3) + 9));
			SetConsoleTextAttribute(hConsole, colorOrder[(int)i/3]);
			if (i / 20.0 < percent) {
				//printf("=");
				printf("%c", (char)(219));
				//lStr = lStr + (char)(219);
			}
			else if (!tip) {
				printf(">");
				//lStr = lStr + ">";
				tip++;
			}
			else {
				SetConsoleTextAttribute(hConsole, 15);
				printf(" ");
			}
		}
		SetConsoleTextAttribute(hConsole, 6 + (16 * 0));
		printf("] %.2f%% Time Remaining: ", percent * 100);
		cout << tStr;
		SetConsoleTextAttribute(hConsole, 15);
	}
}

void ProgressBar::close() {
	cout << endl << endl;
}