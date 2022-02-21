#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>
#include <omp.h>
#include <sys/time.h>

////////////////////////////////////////////////////////////////////////////////

// Simple compute kernel which computes the square of an input array 
//
const char * KernelSource = "\n"\
"__kernel void square(                                                       \n"\
"   __global float* input,                                              \n"\
"   __global float* output,                                             \n"\
"   const unsigned int count)                                           \n"\
"{                                                                      \n"\
"   int i = get_global_id(0);                                           \n"\
"   if(i < count)                                                       \n"\
"       output[i] = input[i] * input[i];                                \n"\
"}                                                                      \n"\
"\n";

////////////////////////////////////////////////////////////////////////////////

int GPUsquare(unsigned int count, float * data, float * results, double * time) {
  int err; // error code returned from api calls

  size_t global; // global domain size for our calculation
  size_t local; // local domain size for our calculation

  cl_device_id device_id; // compute device id 
  cl_context context; // compute context
  cl_command_queue commands; // compute command queue
  cl_program program; // compute program
  cl_kernel kernel; // compute kernel

  cl_mem input; // device memory used for the input array
  cl_mem output; // device memory used for the output array

  // Connect to a compute device
  //
  err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, & device_id, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to create a device group!\n");
    return EXIT_FAILURE;
  }

  // Create a compute context 
  //
  context = clCreateContext(0, 1, & device_id, NULL, NULL, & err);
  if (!context) {
    printf("Error: Failed to create a compute context!\n");
    return EXIT_FAILURE;
  }

  // Create a command commands
  //
  commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, & err);
  if (!commands) {
    printf("Error: Failed to create a command commands!\n");
    return EXIT_FAILURE;
  }

  // Create the compute program from the source buffer
  //
  program = clCreateProgramWithSource(context, 1, (const char ** ) & KernelSource, NULL, & err);
  if (!program) {
    printf("Error: Failed to create compute program!\n");
    return EXIT_FAILURE;
  }

  // Build the program executable
  //
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t len;
    char buffer[2048];

    printf("Error: Failed to build program executable!\n");
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, & len);
    printf("%s\n", buffer);
    exit(1);
  }

  // Create the compute kernel in the program we wish to run
  //
  kernel = clCreateKernel(program, "square", & err);
  if (!kernel || err != CL_SUCCESS) {
    printf("Error: Failed to create compute kernel!\n");
    exit(1);
  }

  // Create the input and output arrays in device memory for our calculation
  //
  input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, NULL);
  output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
  if (!input || !output) {
    printf("Error: Failed to allocate device memory!\n");
    exit(1);
  }

  // Write our data set into the input array in device memory 
  //
  err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to write to source array!\n");
    exit(1);
  }

  // Set the arguments to our compute kernel
  //
  err = 0;
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), & input);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), & output);
  err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), & count);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to set kernel arguments! %d\n", err);
    exit(1);
  }

  // Get the maximum work group size for executing the kernel on the device
  //
  err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), & local, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to retrieve kernel work group info! %d\n", err);
    exit(1);
  }

  // Execute the kernel over the entire range of our 1d input data set
  // using the maximum number of work group items for this device
  //
  cl_event event1;
  global = count;
  err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, & global, & local, 0, NULL, & event1);
  if (err) {
    printf("Error: Failed to execute kernel!\n");
    return EXIT_FAILURE;
  }

  // Wait for the command commands to get serviced before reading back results
  //
  clWaitForEvents(1, & event1);
  clFinish(commands);

  // Get time of execution on the GPU
  //
  cl_ulong time_start = 0;
  cl_ulong time_end = 0;
  double nanoSeconds;

  clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_START, sizeof(time_start), & time_start, NULL);
  clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_END, sizeof(time_end), & time_end, NULL);
  nanoSeconds = (double)(time_end - time_start);
  * time = nanoSeconds / 1000000000.0;

  // Read back the results from the device to verify the output
  //
  err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to read output array! %d\n", err);
    exit(1);
  }

  // Shutdown and cleanup
  //
  clReleaseMemObject(input);
  clReleaseMemObject(output);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  return 0;
}

int CPUsquare(unsigned int count, float * data, float * results) {
  for (int i = 0; i < count; i++) {
    results[i] = data[i] * data[i];
  }
  return 0;
}

int CPUsquare_p(unsigned int count, float * data, float * results) {
  #pragma omp parallel 
  {
    #pragma omp for
    for (int i = 0; i < count; i++) {
      results[i] = data[i] * data[i];
    }

  }
  return 0;
}

int time_GPU_CPU(unsigned int count) {
  float * data = malloc(sizeof(float) * count);
  if (!data){
    printf("malloc failed!\n");
    return 1;
  }
  float * GPUresults = malloc(sizeof(float) * count);
  if (!GPUresults){
    printf("malloc failed!\n");
    return 1;
  }
  float * CPUresults = malloc(sizeof(float) * count);
  if (!CPUresults){
    printf("malloc failed!\n");
    return 1;
  }
  float * CPUresults_p = malloc(sizeof(float) * count);
  if (!CPUresults_p){
    printf("malloc failed!\n");
    return 1;
  }
  int err;
  struct timeval tv1, tv2;

  // Generate array of input data
  for (int i = 0; i < count; i++) {
    data[i] = (float)rand();
  }
  
  /////////////////////
  // GPU calculation //
  /////////////////////
  double GPUtime = 0.0;
  err = GPUsquare(count, data, GPUresults, &GPUtime);
  if (err) {
    printf("GPU error!\n");
    return 1;
  }
  
  ////////////////////////////
  // OpenMP CPU calculation //
  ////////////////////////////
  double OpenMP_CPUtime = 0.0;
  gettimeofday( & tv1, NULL);
  err = CPUsquare_p(count, data, CPUresults_p);
  if (err) {
    printf("CPU error!\n");
    return 1;
  }
  gettimeofday( & tv2, NULL);
  OpenMP_CPUtime = (double)(tv2.tv_usec - tv1.tv_usec) * 1.0e-6 +
                   (double)(tv2.tv_sec - tv1.tv_sec);

  /////////////////////
  // CPU calculation //
  /////////////////////
  double CPUtime = 0.0;
  gettimeofday( & tv1, NULL);
  err = CPUsquare(count, data, CPUresults);
  if (err) {
    printf("CPU error!\n");
    return 1;
  }
  gettimeofday( & tv2, NULL);
  CPUtime = (double)(tv2.tv_usec - tv1.tv_usec) * 1.0e-6 +
            (double)(tv2.tv_sec - tv1.tv_sec); 

  int correct = 0;
  for (int i = 0; i < count; i++) {
    if (GPUresults[i] == CPUresults[i] && CPUresults_p[i] == CPUresults[i]) {
      correct++;
    }
  }

  printf("Correct results: %i/%i\n", correct, count);
  printf("GPU time        = %f milliseconds,  GPU/GPU        = %i\n", GPUtime*1.0e3, (int)(GPUtime/GPUtime));
  printf("OpenMP CPU time = %f milliseconds, OpenMP/GPU     = %i\n", OpenMP_CPUtime*1.0e3, (int)(OpenMP_CPUtime/GPUtime));
  printf("Serial CPU time = %f milliseconds, Serial CPU/GPU = %i\n", CPUtime*1.0e3, (int)(CPUtime/GPUtime));

  free(data);
  free(GPUresults);
  free(CPUresults);
  free(CPUresults_p);

  return 0;
}

int main() {
  unsigned int count = 1024 * 100000;
  int err = time_GPU_CPU(count);
  return err;
}

