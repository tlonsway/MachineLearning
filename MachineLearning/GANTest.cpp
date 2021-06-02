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
#include "bitmap_image.hpp"

namespace fs = std::filesystem;


int main() {

	std::string path = "data/logodb/";
		
	//try {
	//	std::cout << entry.path() << std::endl;
	//}
	//catch (int e) {
    //
	//}

	//exit(26);
	int layerNum1 = 8;
	int* layers1 = new int[8]{ 100,2000,10000,19200,5000,2000,500,2 };
	float lRate1 = .001;
	ActivationFunction* af1 = new ReLu();
	layer::FullyConnected GAN(layers1, layerNum1, lRate1, af1);
	ProgressBar pb2 = ProgressBar();
	int image_num = 12213;
	int image_inc = 0;
	for (const auto& entry : fs::directory_iterator(path)) {
		//std::cout << "Image: " << entry.path() << std::endl;
		//generate image with generator
		if (rand() % 100 <= 1) {
			float* randData = (float*)malloc(sizeof(float) * 100);
			gpuMath::blasOp::randMatCPUMemNormal(randData, 100, 0, 1.0);
			//if (image_inc % 100 == 0) {
				float* genOut = GAN.feedForwardGAN(randData, 3);
				bitmap_image imageG(80, 80);
				//imageG.set_all_channels(0, 0, 0);
				int incG = 0;
				for (int r = 0; r < 80; r++) {
					for (int c = 0; c < 80; c++) {
						rgb_t tempCol;
						tempCol.red = (int)(genOut[incG] * 255);
						tempCol.green = (int)(genOut[incG + 1] * 255);
						tempCol.blue = (int)(genOut[incG + 2] * 255);
						imageG.set_pixel(r, c, tempCol);
						incG += 3;
					}
				}
				imageG.save_image("generated/generated_" + std::to_string(image_inc) + ".bmp");
			//}
			//feed GAN image into discriminator & backprop both
			float* disOut = GAN.feedForward(randData);
			float genErr = disOut[0];
			std::cout << "GEN DisOut0: " << disOut[0] << " DissOut1: " << disOut[1] << std::endl;
			float* disY1 = (float*)malloc(sizeof(float) * 2);
			disY1[0] = 1.0; //image is fake
			disY1[1] = 0.0; //image is real
			GAN.backProp(randData, disY1);
			free(disY1);
			free(randData);
			free(disOut);
		}
		//feed real image into discriminator & backprop
		if (rand() % 100 <= 100) {
			float* realIm = (float*)malloc(sizeof(float) * 19200);
			bitmap_image image(entry.path().u8string());
			int inc = 0;
			for (int r = 0; r < 80; r++) {
				for (int c = 0; c < 80; c++) {
					rgb_t color;
					image.get_pixel(r, c, color);
					realIm[inc] = (float)color.red / 255;
					realIm[inc + 1] = (float)color.green / 255;
					realIm[inc + 2] = (float)color.blue / 255;
					inc += 3;
				}
			}
			//float* disOut2 = GAN.feedForwardGAN()
			//std::cout << "GEN DisOut0: " << disOut[0] << " DissOut1: " << disOut[1] << std::endl;
			float* disY2 = (float*)malloc(sizeof(float) * 2);
			disY2[0] = 0.0; //image is fake
			disY2[1] = 1.0; //image is real
			GAN.backPropGAN(realIm, disY2, 3);
			free(realIm);
			free(disY2);
		}
		
		//free(genOut);
		
		pb2.display(image_inc, image_num, "Training GAN");
		image_inc++;

	}

	exit(26);






	/*
	const int ganInSize = 10;

	//construct generator network
	int layerNum1 = 4;
	int* layers1 = new int[4]{ ganInSize,2000,5000,19200 };
	float lRate1 = .1;
	ActivationFunction* af1 = new Sigmoid();
	layer::FullyConnected gen(layers1, layerNum1, lRate1, af1);
	//construct discriminator network
	int layerNum2 = 5;
	int* layers2 = new int[5]{ 19200,5000,1500,250,2 };
	float lRate2 = .1;
	ActivationFunction* af2 = new Sigmoid();
	layer::FullyConnected dis(layers2, layerNum2, lRate2, af2);

	//train the networks
	ProgressBar pb1 = ProgressBar();
	int image_num = 12213;
	int image_inc = 0;
	for (const auto& entry : fs::directory_iterator(path)) {
		//generate image with generator
		float* randData = (float*)malloc(sizeof(float) * ganInSize);
		gpuMath::blasOp::randMatCPUMemNormal(randData, ganInSize, 0, 1.0);
		float* genOut = gen.feedForward(randData);
		bitmap_image imageG(80, 80);
		//imageG.set_all_channels(0, 0, 0);
		int incG = 0;
		for (int r = 0; r < 80; r++) {
			for (int c = 0; c < 80; c++) {
				rgb_t tempCol;
				tempCol.red = (int)(genOut[incG]*255);
				tempCol.green = (int)(genOut[incG+1] * 255);
				tempCol.blue = (int)(genOut[incG+2] * 255);
				imageG.set_pixel(r, c, tempCol);
				incG+=3;
			}
		}
		imageG.save_image("generated/generated_" + std::to_string(image_inc)+".bmp");
		//feed GAN image into discriminator & backprop both
		float* disOut = dis.feedForward(genOut);
		float genErr = disOut[0];
		std::cout << "Generator Error: " << genErr << std::endl;
		gen.backPropGAN(randData,genErr);
		float* disY1 = (float*)malloc(sizeof(float) * 2);
		disY1[0] = 1.0; //image is fake
		disY1[1] = 0.0; //image is real
		dis.backProp(genOut, disY1);
		//feed real image into discriminator & backprop
		float* realIm = (float*)malloc(sizeof(float) * 19200);
		bitmap_image image(entry.path().u8string());
		int inc = 0;
		for (int r = 0; r < 80; r++) {
			for (int c = 0; c < 80; c++) {
				rgb_t color;
				image.get_pixel(r, c,color);
				realIm[inc] = (float)color.red/255;
				realIm[inc + 1] = (float)color.green/255;
				realIm[inc + 2] = (float)color.blue/255;
				inc+=3;
			}
		}
		inc++;
		float* disY2 = (float*)malloc(sizeof(float) * 2);
		disY2[0] = 0.0; //image is fake
		disY2[1] = 1.0; //image is real
		dis.backProp(realIm, disY2);
		free(realIm);
		free(disY2);
		free(disY1);
		free(randData);
		free(genOut);
		free(disOut);
		pb1.display(image_inc, image_num, "Training GAN");
		image_inc++;
	}*/
}