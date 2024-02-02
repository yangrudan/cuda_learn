//
// Created by yang on 24-2-2.
//
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 1024

// OpenCL kernel
const char* kernelSource =
        "__kernel void vectorAdd(__global const float* a, __global const float* b, __global float* result) {\n"
        "    int index = get_global_id(0);\n"
        "    result[index] = a[index] + b[index];\n"
        "}\n";

int main() {
    // Initialize input vectors
    float a[ARRAY_SIZE];
    float b[ARRAY_SIZE];
    float result[ARRAY_SIZE];

    for (int i = 0; i < ARRAY_SIZE; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Load OpenCL platform
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    // Load OpenCL device
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Create OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    // Create command queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // Create OpenCL program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);

    // Build OpenCL program
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Create OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vectorAdd", NULL);

    // Create OpenCL buffers
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, a, NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, b, NULL);
    cl_mem bufferResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * ARRAY_SIZE, NULL, NULL);

    // Set OpenCL kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferResult);

    // Execute OpenCL kernel
    size_t globalSize = ARRAY_SIZE;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    clFinish(queue);

    // Read the result from OpenCL buffer
    clEnqueueReadBuffer(queue, bufferResult, CL_TRUE, 0, sizeof(float) * ARRAY_SIZE, result, 0, NULL, NULL);

    // Display the result
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        printf("%f + %f = %f\n", a[i], b[i], result[i]);
    }

    // Clean up
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferResult);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
