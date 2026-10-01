# Parallel Face Detection with MPI and OpenCV

A parallel batch face detection project implemented with **C++, OpenCV, MPI, and OpenMP**, evaluated on the **NCHC Taiwania 3** high-performance computing platform.

This project was developed as the final project for a Parallel Programming course. The goal was not only to accelerate batch face detection, but also to investigate how task scheduling, process configuration, library-level multithreading, and hardware characteristics affect parallel performance.

---

## Project Overview

The project uses OpenCV Haar Cascade for batch face detection and explores multiple parallel processing strategies.

The implementation evolved through several architectures:

- Serial Baseline
- Pure MPI Dynamic Scheduling
- Hybrid MPI + OpenMP
- Computing Master
- Adaptive Master
- Controlled Pure MPI

The development process focused on:

- Correctness verification
- Dynamic workload balancing
- Strong scaling analysis
- Parallel efficiency
- Thread oversubscription
- Worker starvation
- MPI Master-Worker scheduling
- NUMA and cache locality observations

---

## Technologies

- **Language:** C++
- **Image Processing:** OpenCV
- **Parallel Computing:** MPI, OpenMP
- **Job Scheduler:** Slurm
- **HPC Platform:** NCHC Taiwania 3
- **Compiler / MPI Environment:** Intel C++ Compiler, Intel MPI

---

## System Architecture

The final Controlled Pure MPI implementation adopts a **Master-Worker architecture**.

### Master Process

Rank 0 is responsible for:

- Maintaining the global image task queue
- Receiving results from workers
- Dynamically assigning the next image
- Collecting the total number of detected faces
- Recording worker load distribution
- Measuring total execution time

### Worker Processes

Each worker:

1. Requests or receives an image task
2. Loads the image in grayscale
3. Performs Haar Cascade face detection
4. Returns the number of detected faces to the Master
5. Receives the next available task

This dynamic scheduling strategy allows faster workers to process more images and helps reduce workload imbalance.

---

## Controlled Pure MPI

One important issue observed during development was interference caused by OpenCV's internal multithreading.

To better isolate MPI-level parallelism, the final implementation explicitly disables OpenCV internal threads:

```cpp
cv::setNumThreads(1);
```

This allows the parallel execution behavior to be controlled primarily by MPI processes and provides a cleaner environment for strong-scaling analysis.

---

## Experimental Dataset

Two workloads were used to evaluate the system:

| Dataset | Images | Detected Faces | Description |
|---|---:|---:|---|
| High Load | 248 | 1,253 | Images containing relatively dense face regions |
| Low Load | 600 | 1,106 | Images with fewer faces or relatively lighter workloads |

The Serial and parallel implementations produced the same total number of detected faces, confirming that task distribution did not introduce image loss or duplicate processing.

---

## Experimental Architectures

### 1. Serial Baseline

A single-process implementation using OpenCV Haar Cascade.

The serial version also uses:

```cpp
cv::setNumThreads(1);
```

to maintain a controlled comparison with the final MPI implementation.

---

### 2. Pure MPI Dynamic Scheduling

A Master-Worker architecture using dynamic task assignment.

Workers receive new work after completing their current task, allowing the workload to adapt automatically to differences in image processing time.

---

### 3. Hybrid MPI + OpenMP

A two-level parallel architecture combining:

- MPI for process-level task distribution
- OpenMP for thread-level parallelism within workers

During testing, this version revealed performance degradation associated with thread oversubscription and interaction with OpenCV internal multithreading.

---

### 4. Computing Master

The Master process was modified to participate in face detection while monitoring worker requests.

Although this improved CPU utilization in some small-process configurations, larger configurations showed worker starvation because the Master could become occupied with expensive image processing while workers were waiting for new tasks.

---

### 5. Adaptive Master

An adaptive strategy was introduced:

- Small process counts: Master participates in computation
- Larger process counts: Master becomes a dedicated scheduler

This was designed to reduce worker starvation and scheduling bottlenecks.

---

### 6. Controlled Pure MPI

The final strong-scaling analysis version.

OpenCV internal multithreading is disabled, allowing MPI processes to control parallel execution more explicitly.

---

## Representative Performance Results

### Execution Time

| Architecture | High Load Best (s) | Low Load Best (s) |
|---|---:|---:|
| Serial Baseline | 142.82 | 282.04 |
| Pure MPI Dynamic | 11.99 | 23.63 |
| Hybrid MPI + OpenMP | 18.52 | 37.14 |
| Adaptive Master | 14.02 | 24.80 |
| Controlled Pure MPI | 14.40 | 28.46 |

Pure MPI Dynamic achieved the shortest observed execution time among the tested configurations.

However, Controlled Pure MPI was used for the main strong-scaling analysis because it better isolates MPI performance from OpenCV's internal multithreading behavior.

---

## Strong Scaling Analysis

For the Controlled Pure MPI experiment, the `np=2` configuration consists of:

- 1 dedicated Master
- 1 Worker

This configuration is used as the strong-scaling baseline.

When scaling to `np=12`:

- 1 Master
- 11 Workers

the experiment achieved approximately:

| Dataset | Speedup | Parallel Efficiency |
|---|---:|---:|
| High Load | **10.11×** | **92%** |
| Low Load | **10.14×** | **92%** |

> Note: These speedup values are relative to the Controlled Pure MPI `np=2` configuration with one Worker, rather than the independent Serial Baseline.

---

## Performance Issues Investigated

### Thread Oversubscription

During the Pure MPI and Hybrid experiments, increasing the number of processes did not always improve performance.

One observed reason was interaction between:

- MPI processes
- OpenMP threads
- OpenCV internal multithreading

This could create more software threads than available hardware resources and increase context-switching and cache contention.

The final Controlled Pure MPI version therefore uses:

```cpp
cv::setNumThreads(1);
```

to disable OpenCV internal parallelism.

---

### Worker Starvation

In the Computing Master architecture, Rank 0 performed both:

- task scheduling
- face detection

When the Master spent too much time performing image processing, workers could finish their assigned tasks and remain idle while waiting for new work.

This motivated the Adaptive Master strategy, where the Master becomes a dedicated scheduler when more processes are used.

---

### NUMA and Load Distribution

Experiments on Taiwania 3 showed that worker processing counts were not always uniform.

Possible factors include:

- NUMA architecture
- cache locality
- operating-system scheduling
- memory placement

These observations were not directly verified using hardware profiling tools such as `perf`, `numactl`, or `hwloc`, so they are treated as possible explanations and future investigation directions rather than confirmed causes.

---

## Repository Structure

```text
Parallel-Face-Detection-MPI-OpenCV/
│
├── src/
│   ├── serial_face.cpp
│   └── parallel_face.cpp
│
├── results/
│   ├── benchmark_1318056.log
│   ├── final_1318114.log
│   ├── final_1318139.log
│   ├── final_1318213.log
│   └── hybrid_1318076.log
│
├── docs/
│   └── Final_Report.pdf
│
├── Makefile
├── README.md
└── README_run.txt
```

> The exact directory structure may vary slightly depending on the uploaded repository organization.

---

## Source Code Availability

The repository currently contains reproducible source code for:

- **Serial Baseline**
- **Controlled Pure MPI**

Several additional architectures were implemented and evaluated during development, including:

- Pure MPI Dynamic
- Hybrid MPI + OpenMP
- Computing Master
- Adaptive Master

Their source-code versions were overwritten during iterative development, so their experimental results are preserved in the benchmark log files rather than as separate reproducible source files.

---

## Build

Use the provided `Makefile`:

```bash
make
```

---

## Run

### Serial Baseline

```bash
./serial_face dataset/high_load
```

or:

```bash
./serial_face dataset/low_load
```

### Controlled Pure MPI

For example, using 12 MPI processes:

```bash
mpirun -np 12 ./parallel_face dataset/high_load
```

or:

```bash
mpirun -np 12 ./parallel_face dataset/low_load
```

---

## Required External Files

The following files are not included in this repository:

- Haar Cascade XML model
- High Load image dataset
- Low Load image dataset

The face detector expects:

```text
haarcascade_frontalface_default.xml
```

to be available in the execution directory.

The experimental datasets were derived from the WIDER FACE dataset for coursework benchmarking purposes.

---

## What I Learned

Through this project, I gained practical experience in:

- C++ programming
- MPI communication
- Master-Worker architecture
- Dynamic workload scheduling
- OpenMP experimentation
- OpenCV image processing
- HPC job execution with Slurm
- Strong scaling analysis
- Parallel efficiency evaluation
- Performance debugging
- Thread oversubscription analysis
- System-level bottleneck investigation

More importantly, the project helped me understand that parallel performance is not determined simply by increasing the number of processes.

Library-level threading, task granularity, scheduling strategy, workload distribution, and hardware architecture can all affect the final performance of a parallel system.

---

## Future Work

Possible extensions include:

- NUMA-aware CPU affinity optimization
- Profiling with `perf`, `hwloc`, or `numactl`
- Non-blocking MPI communication
- GPU acceleration with CUDA or OpenCL
- Real-time video stream processing
- Parallelization of deep-learning-based face detectors

---

## Author

**Heng-Ting Liu (劉姮廷)**  
Department of Artificial Intelligence  
Chang Gung University
