#pragma once
#include <iostream>

class ProgressBar
{
public:
	ProgressBar();
	void display(float position, float end_point, std::string title);
};

