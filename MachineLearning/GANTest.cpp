#include <stdio.h>
#include <stdlib.h>		
#include <string.h>
#include <iostream>
#include "layer.h"
#include "animations.h"
#include "ProgressBar.h";
#include "gpuMath.h"
#include "stb_image.h"
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;


int main() {

	std::string path = "B:/LogoData/";
	try {
		for (const auto& entry : fs::directory_iterator(path)) {
			try {
				std::string s = &entry.path().filename;
			}
			catch (int e) {

			}
			//printf("%s\n", entry.path());
			//std::cout << entry.path() << std::endl;	
		}
	}
	catch (int e) {
	}

	const int ganInSize = 100;

	//construct generator network
	int layerNum1 = 4;
	int* layers1 = new int[4]{ ganInSize,2000,5000,30000 };
	float lRate1 = .01;
	ActivationFunction* af1 = new ReLu();
	layer::FullyConnected gen(layers1, layerNum1, lRate1, af1);
	//construct discriminator network
	int layerNum2 = 5;
	int* layers2 = new int[5]{ 30000,5000,1500,250,2 };
	float lRate2 = .01;
	ActivationFunction* af2 = new Sigmoid();
	layer::FullyConnected dis(layers2, layerNum2, lRate2, af2);

	//train the networks
	ProgressBar pb1 = ProgressBar();
	int image_num = 0;
	for (int i = 0; i < image_num; i++) {
		//generate image with generator
		float* randData = (float*)malloc(sizeof(float) * ganInSize);
		gpuMath::blasOp::randMatCPUMemNormal(randData, ganInSize, 0, 1.0);
		float* genOut = gen.feedForward(randData);
		//feed GAN image into discriminator & backprop both
		float* disOut = dis.feedForward(genOut);
		float genErr = disOut[0];
		gen.backPropGAN(randData,genErr);
		float* disY1 = (float*)malloc(sizeof(float) * 2);
		disY1[0] = 1.0; //image is fake
		disY1[1] = 0.0; //image is real
		dis.backProp(genOut, disY1);
		//feed real image into discriminator & backprop
		float* realIm = (float*)malloc(sizeof(float) * 30000);
		float* disY2 = (float*)malloc(sizeof(float) * 2);
		disY2[0] = 0.0; //image is fake
		disY2[1] = 1.0; //image is real
		dis.backProp(realIm, disY2);
	}
}