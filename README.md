# OpenCL-Apple-M1-Benchmark

`main.c` computes the square of each value in a long array of floats with
- GPU using OpenCL
- Multithreaded CPU using OpenMP
- Serial CPU

Results for Apple M1 for squaring 102,400,000 single precision floats:
|            | Time (milliseconds) | Time Relative to GPU |
| ---------- | ------------------- | -------------------- |
| GPU        | 0.335079            | 1x                   |
| OpenMP CPU | 31.831000           | 90x                  |
| Serial CPU | 38.473000           | 120x                 |


