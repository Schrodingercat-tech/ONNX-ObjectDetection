#pragma once

#ifdef ONNX_INF_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "iostream"
#include "vector"
#include "src/common.h"

// YOLOv8s model details
TimeIt codeLifeSpan; // ~100-150 milli sec too slow ?

// create an env
const wchar_t* model_path = L"models/yolov8s.onnx"; // we got static and dynamic I/O shapes in a given onnx model
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "OBJ-Detection");
Ort::SessionOptions session_options;
session_options.SetInterOpNumThreads(1);


Ort::Session session{ env, model_path, session_options };

Ort::AllocatorWithDefaultOptions allocator; // set default allocator

// I/O nodes 
size_t input_nodes, output_nodes;
input_nodes = session.GetInputCount();
output_nodes = session.GetOutputCount();

// I/O names
std::vector<std::string> input_names(input_nodes), output_names(output_nodes);
std::vector<int64_t> input_shape, output_shape;


// Get input node names and shape
for (size_t i = 0; i < input_nodes; ++i) {
	input_names[i] = session.GetInputNameAllocated(i, allocator).get();
	input_shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

}

// Get input node names and shape
for (size_t i = 0; i < output_nodes; ++i) {
	output_names[i] = session.GetOutputNameAllocated(i, allocator).get();
	output_shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

}

print(

	"\n__________ONNX Model info__________\n",
	"\nInput Name : ", input_names[0],
	"\nInput shape : [", input_shape[0], input_shape[1], input_shape[2], input_shape[3], "]",
	"\nOutput Name : ", output_names[0],
	"\nOutput shape : [", output_shape[0], output_shape[1], output_shape[2], "]"

);

// the above part will be standard for any type of models

// process image data with cv

cv::Mat image = cv::imread("images/people.jpg");
if (!image.data) { cerr << "ERROR: Could not open or find the image." << endl; return -1; }
cv::Mat resized;
(input_shape[2] == -1) ? // checks if model is dynamic or static 
cv::resize(image, resized, cv::Size(640, 640)) : // for dynamic set default hw
	cv::resize(image, resized, cv::Size(input_shape[2], input_shape[2])); // for static its resized accordingly

resized.convertTo(resized, CV_32F, 1.0 / 255.0); //sucks this ain't 10 bit color map?
// transpose HWC -> CHW format since yolo is build on top of torch frame work

/*
	this is just for my reference

	HWC FORMAT [2,3,4]
	[
	  [[R G B] [R G B] [R G B] [R G B]]
	  [[R G B] [R G B] [R G B] [R G B]]
	] 2*3*3

	CHW FORMAT [3,2,4]
	[
		[[R R R R]
		 [R R R R]]
		[[G G G G]
		 [G G G G]]
		[[B B B B]
		 [B B B B]]
	] 3*2*4

*/

cv::Mat FlattenImage;
hwcToChw(resized, FlattenImage); // convert hwc to chw

// now convert this resized image into format Ort can understand
size_t input_tensor_size; // 1*3*640*640 it wouldn't work for dynamic sizes(-1)
std::vector<float> input_tensor_data((float*)FlattenImage.datastart, (float*)FlattenImage.dataend);// just provide the start and end pointers that gives flatten vec

print("\nInput Tensor Size : ",
	input_tensor_data.size(),
	"\nRaw Output from Image : ", resized.size(),
	"\ncompare both above results if they are matching"
);


// define memory info
Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault); // I'm too broke to run on GPU

std::vector<int64_t> input_shape_ = { 1, resized.channels(),resized.rows, resized.cols }; // since I didnt use vector<Mat> for batch of images thus it will be one

print(input_tensor_data.size(), input_shape.size());

// First, get the input and output node names as C-strings
const char* input_name = input_names[0].c_str();
const char* output_name = output_names[0].c_str();

// Create input tensor (your existing code is correct here)
Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
	memory_info,
	input_tensor_data.data(),
	input_tensor_data.size(),
	input_shape_.data(),
	input_shape_.size()
);

// Run inference - corrected version
try {
	auto output_tensors = session.Run(
		Ort::RunOptions{},
		&input_name,         // Input names (pointer to array of const char*)
		&input_tensor,       // Input tensor (pointer to array of Ort::Value)
		1,                   // Number of inputs
		&output_name,        // Output names (pointer to array of const char*)
		1                    // Number of outputs
	);

	// Add debug print to verify output
	print("\nInference completed. Output tensor count: ", output_tensors.size());

	// Access the output data (example)
	float* output_data = output_tensors[0].GetTensorMutableData<float>();
	size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
	print("Output tensor size: ", output_size);
}
catch (const Ort::Exception& e) {
	print("ONNX Runtime Error: ", e.what());
}





#endif // ONNX_INF_H