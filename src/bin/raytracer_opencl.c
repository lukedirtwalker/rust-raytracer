#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x8000) // 16KB

#define PROG_NAME "raytracer.ocl"

#define LOG_SIZE 4096
char LOGBUF[LOG_SIZE];

#define WIDTH 1024
#define HEIGHT 768
#define IMG_SIZE (WIDTH * HEIGHT)

float clamp (float x) {
    if (x < 0) return 0.0f;
    else if (x > 1) return 1.0f;
    return x;
}

long long to_int(float x) {
    x = clamp(x);
    x = powf(x, 1.0f / 2.2f);
    x *= 255.0;
    x += 0.5;
    return x;
}

int main() {    
    FILE *fp;
    char *source_str;
    size_t source_size;
    
    fp = fopen(PROG_NAME, "r");
    if (!fp) {
        fprintf(stderr, "Failed to read the kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
    
    // Fetch the platforms (we only want one)
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_platforms;
    cl_int err = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (err != CL_SUCCESS) {
        printf("Error in get platform id: %d\n", err);
        exit(1);
    } else if (ret_num_platforms < 1) {
        printf("No platform found\n");
        exit(1);
    }
    
    // Get the default device for this platform
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);
    if (err != CL_SUCCESS) {
        printf("Error in get device id: %d\n", err);
        exit(1);
    } else if (ret_num_devices < 1) {
        printf("No platform found\n");
        exit(1);
    }
    
    
//     clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &ret_num_devices);
//     cl_device_id* devices = (cl_device_id*) malloc(sizeof(cl_device_id) * ret_num_devices);
//     clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, ret_num_devices, devices, NULL);
//     cl_uint start = 0;
//     for (; start < ret_num_devices; ++start) {
//         char info[100];
//         size_t foo;
//         clGetDeviceInfo(devices[start], CL_DEVICE_VENDOR, 100, info, &foo);
//         printf("%s\n", info);
//     }
//     return;
    
    // Create a memory context for the device we want to use (for NVidia)
    cl_context_properties properties[]={CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id,0};
    
    // Create an OpenCL context
    cl_context context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error while creating the context: %d\n", err);
        exit(1);
    }
    
    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (err != CL_SUCCESS) {
        printf("Error while creating the queue: %d\n", err);
        exit(1);
    }
    
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
                                                   &source_size, &err);
    if (err != CL_SUCCESS) {
        printf("Error while creating the program: %d\n", err);
        exit(1);
    }
    
    // Compile the kernel code
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error while building the program: %d\n", err);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, LOG_SIZE, LOGBUF, NULL );
        printf( "Build Log for %s:\n%s\n", PROG_NAME, LOGBUF );
        exit(1);
    }
    
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    
    // Allocate and zero host memory
    float *host_x = (float*)malloc(IMG_SIZE *sizeof(float));
    float *host_y = (float*)malloc(IMG_SIZE *sizeof(float));
    float *host_z = (float*)malloc(IMG_SIZE *sizeof(float));
    memset(host_x, 0, IMG_SIZE * sizeof(float));
    memset(host_y, 0, IMG_SIZE * sizeof(float));
    memset(host_z, 0, IMG_SIZE * sizeof(float));
    
    // Create memory buffers on the device for each vector
    cl_mem opencl_x = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     IMG_SIZE * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { exit(1); }
    cl_mem opencl_y = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     IMG_SIZE * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { exit(1); }
    cl_mem opencl_z = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     IMG_SIZE * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { exit(1); }
    
    // Copy the buffers
    err = clEnqueueWriteBuffer(command_queue, opencl_x, CL_TRUE, 0,
                               IMG_SIZE * sizeof(float), host_x, 0, NULL, NULL);
    if (err != CL_SUCCESS) { exit(1); }
    err = clEnqueueWriteBuffer(command_queue, opencl_y, CL_TRUE, 0,
                               IMG_SIZE * sizeof(float), host_y, 0, NULL, NULL);
    if (err != CL_SUCCESS) { exit(1); }
    err = clEnqueueWriteBuffer(command_queue, opencl_z, CL_TRUE, 0,
                               IMG_SIZE * sizeof(float), host_z, 0, NULL, NULL);
    if (err != CL_SUCCESS) { exit(1); }
    
    
    // Set the arguments of the kernel
    err = clSetKernelArg(kernel, 0, sizeof(opencl_x), (void *)&opencl_x);
    if (err != CL_SUCCESS) { exit(1); }
    err = clSetKernelArg(kernel, 1, sizeof(opencl_y), (void *)&opencl_y);
    if (err != CL_SUCCESS) { exit(1); }
    err = clSetKernelArg(kernel, 2, sizeof(opencl_z), (void *)&opencl_z);
    if (err != CL_SUCCESS) { exit(1); }
    
    // Execute the OpenCL kernel on the image
    size_t global_item_size[2] = {HEIGHT, WIDTH};
    err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
                                 global_item_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { printf("Error at line %d, err: %d", __LINE__, err); exit(1); }
    
    // Read the memory back
    err = clEnqueueReadBuffer(command_queue, opencl_x, CL_TRUE, 0,
                              IMG_SIZE * sizeof(float), host_x, 0, NULL, NULL);
    if (err != CL_SUCCESS) { printf("Error at line %d, err: %d", __LINE__, err); exit(1); }
    err = clEnqueueReadBuffer(command_queue, opencl_y, CL_TRUE, 0,
                            IMG_SIZE * sizeof(float), host_y, 0, NULL, NULL);
    if (err != CL_SUCCESS) { printf("Error at line %d, err: %d", __LINE__, err); exit(1); }
    err = clEnqueueReadBuffer(command_queue, opencl_z, CL_TRUE, 0,
                        IMG_SIZE * sizeof(float), host_z, 0, NULL, NULL);
    if (err != CL_SUCCESS) { printf("Error at line %d, err: %d", __LINE__, err); exit(1); }
    err = clFinish(command_queue);
    if (err != CL_SUCCESS) { printf("Error at line %d, err: %d", __LINE__, err); exit(1); }

    // Display the result to the screen
    printf("P3\n%d %d\n255\n", WIDTH, HEIGHT);
    int i, j;
    for (i = 0; i < HEIGHT; ++i)
        for (j = 0; j < WIDTH; ++j)
            printf("%d %d %d ", to_int(host_x[i + j * HEIGHT]), to_int(host_y[i + j * HEIGHT]), to_int(host_z[i + j * HEIGHT]));
    
    // Clean up
//     err = clFlush(command_queue);
    err = clReleaseKernel(kernel);
    err = clReleaseProgram(program);
    err = clReleaseMemObject(opencl_x);
    err = clReleaseMemObject(opencl_y);
    err = clReleaseMemObject(opencl_z);
    err = clReleaseCommandQueue(command_queue);
    err = clReleaseContext(context);
    free(host_x);
    free(host_y);
    free(host_z);
    return 0;
}