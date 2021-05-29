#pragma once
#include <iostream>
#include <windows.h>

class ProgressBar
{
public:
	ProgressBar();
	void display(float position, float end_point, std::string title);
	void close();
private:
	HANDLE hConsole;
	long startTime;
	int* colorOrder;
	long lastTime;
};

