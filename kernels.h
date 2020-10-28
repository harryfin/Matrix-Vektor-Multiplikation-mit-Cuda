#ifndef KERNELS_H_
#define KERNELS_H_

#include <cooperative_groups.h>
#include <cuda.h>
#include <iostream>

using namespace cooperative_groups;

// 0 -  SharedMemory Kernel
// Nutzt Shared Memory
// Reduktion der Teilsummen eines Blocks
// Speicherung der Teilsumme in Buffer
__global__ void smKernel(float *A, float *buffer, float *x, float *y,
                         const int M, const int N, const int N_PADDING,
                         const int N_Buffer, const int THREADS) {

  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  __shared__ float sm_A[1024 * sizeof(float)];
  __shared__ float sm_x[1024 * sizeof(float)];

  if (row < M && col < N) {
    sm_A[threadIdx.x] = A[row * N_PADDING + col];
    sm_x[threadIdx.x] = x[col]; // Wird öfters ausgeführt
    __syncthreads();
    sm_A[threadIdx.x] = sm_A[threadIdx.x] * sm_x[threadIdx.x];
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k /= 2) {
      if (threadIdx.x < k)
        sm_A[threadIdx.x] += sm_A[threadIdx.x + k];
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      buffer[blockIdx.x + N_Buffer * blockIdx.y] = sm_A[0];
    }
    // wird nur ausgeführt wenn keine Aufruf eines Reduktions Kernels notwendig
    if (N_Buffer == 1 && threadIdx.x == 0) {
      atomicAdd(&y[row], sm_A[0]);
    }
  }
}
// 0 - Shared Memory Reduktion Kernel
// Nutzt Shared Memory
// Reduktion der Teilsummen im Buffer
// Schreib Teilsumme in Buffer (im letzen Aufruf in Ergebnisvektor)
__global__ void reduktionSmKernel(float *buffer, float *y, int bufferXlength,
                                  int N_Buffer, const int M, int activeBlocks) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  extern __shared__ float s[];
  if (row < M && col < bufferXlength) {
    s[threadIdx.x] = buffer[row * N_Buffer + col];
  } else {
    s[threadIdx.x] = 0.0;
  }
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    if (threadIdx.x < k)
      s[threadIdx.x] += s[threadIdx.x + k];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    buffer[blockIdx.x + N_Buffer * blockIdx.y] = s[0];
  }
  __syncthreads();
  if (threadIdx.x == 0 && activeBlocks == 1) {
    atomicAdd(&y[row], s[0]);
  }
}
///////////////////////////////////////////////////////////////////////////////
// 1 - Atomic Oparions am Ende des Shared-Memory-Block
// Nutzt Shared Memory
// Reduktion der Teilsummen eines Block
// Schreibvorgang in Ergbenisvektor per Atomic-Add
__global__ void smAtomicKernel(float *A, float *x, float *y, const int M,
                               const int N, const int N_PADDING,
                               const int THREADS) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  __shared__ float sm_A[1024 * sizeof(float)];
  __shared__ float sm_x[1024 * sizeof(float)];
  if (row < M && col < N) {
    sm_A[threadIdx.x] = A[row * N_PADDING + col];
    sm_x[threadIdx.x] = x[col]; // Wird öfters ausgeführt
    __syncthreads();
    sm_A[threadIdx.x] = sm_A[threadIdx.x] * sm_x[threadIdx.x];
    __syncthreads();

    for (int k = blockDim.x / 2; k > 0; k /= 2) {
      if (threadIdx.x < k)
        sm_A[threadIdx.x] += sm_A[threadIdx.x + k];
      __syncthreads();
    }
    if (threadIdx.x == 0)
      atomicAdd(&y[row], sm_A[0]);
  }
}
///////////////////////////////////////////////////////////////////////////////
// 2 - SharedMemory intraGridKernel
// Nutzt Shared SharedMemory
// Reduktion der Teilsummen eines Blocks
// Schreibt Teilsummen in Buffer
// Reduktion der Teilsummen in Buffer für die Blocks im GridGroup
// Blocks in GridGroup immer Zeilenweise - Limitierung der Matrixgröße in
// N-Richtung auf maximale ladbare Blocks in GridGroup (Architektur der GPU auf
// Server GPU01-frz bietet ausreichend Kapazität für maximale Matrixgröße
// (Ausnahme: kleinste Blockgrößen mit größten Matrizen))
__global__ void smIntraGridKernel(float *A, float *x, float *y, int M, int N,
                                  int N_PADDING, int THREADS, float *buffer,
                                  int blocksInX, int rowShift, int rowsInWork) {

  grid_group grid = this_grid();
  grid.sync();
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  row = row + rowShift;
  __shared__ float sm_A[1024 * sizeof(float)];
  __shared__ float sm_x[1024 * sizeof(float)];

  if (row < row + rowsInWork && row < M && col < N) {
    // Berechnung Teilsumme
    sm_A[threadIdx.x] = A[row * N_PADDING + col];
    sm_x[threadIdx.x] = x[col]; // Wird öfters ausgeführt
    __syncthreads();
    sm_A[threadIdx.x] = sm_A[threadIdx.x] * sm_x[threadIdx.x];
    __syncthreads();
    // Teilsummen im Block zusammenfassen
    for (int k = blockDim.x / 2; k > 0; k /= 2) {
      if (threadIdx.x < k)
        sm_A[threadIdx.x] += sm_A[threadIdx.x + k];
      __syncthreads();
    }
    // Blocksummer in Buffer schreiben
    if (threadIdx.x == 0) {
      buffer[blockIdx.x + blocksInX * blockIdx.y] = sm_A[0];
    }
  }
  // Alle Grids synchronisieren
  grid.sync();
  if (row < M && col < N) {
    // Buffer zusammenfassen
    if (col == 0 && threadIdx.x == 0) {
      // for (int k = blocksInX / 2; k > 0; k /= 2) {
      for (int k = 1; k < blocksInX; k++) {
        // if (blockIdx.x + blocksInX * blockIdx.y < k)
        buffer[blockIdx.x + blocksInX * blockIdx.y] +=
            buffer[blockIdx.x + blocksInX * blockIdx.y + k];
        __syncthreads();
      }
      // eine Atomic am Ende
      atomicAdd(&y[row], buffer[blockIdx.x + blocksInX * blockIdx.y]);
    }
  }
}
///////////////////////////////////////////////////////////////////////////////
// 3 - Atomic Operations für alle notwendigen Operationen
// Reduktion der Teilsummen im Blöcken ohne Shared Memory
// Speicherung direkt in Ergebnisvektor mittels Atomic-Add
__global__ void allAtomicKernel(float *A, float *x, float *y, const int M,
                                const int N, const int N_PADDING,
                                const int THREADS) {
  // Jeder Thread mit eine Atomic Add auf den Erebnisvektoren
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float sum = 0.0;
    sum += A[row * N_PADDING + col] * x[col];
    atomicAdd(&y[row], sum);
  }
}
///////////////////////////////////////////////////////////////////////////////
// 4 -  Intra Block Communication
// Nicht relevant für das Projekt
// Nutzung von Shared Memory
// Reduktion der Teilsummen in den Blöcken
// Speicherung direkt in Ergebnisvektor mittels atomicAdd
// Indexierung über Intra-Block
__global__ void smIntraBlockKernel(float *A, float *x, float *y, const int M,
                                   const int N, const int N_PADDING,
                                   const int THREADS) {

  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  __shared__ float sm_A[1024 * sizeof(float)];
  __shared__ float sm_x[1024 * sizeof(float)];

  if (row < M && col < N) {
    sm_A[threadIdx.x] = A[row * N_PADDING + col];
    sm_x[threadIdx.x] = x[col];
    __syncthreads();
    sm_A[threadIdx.x] = sm_A[threadIdx.x] * sm_x[threadIdx.x];
    __syncthreads();
    auto g = this_thread_block();
    unsigned long long index = g.thread_rank();
    for (int k = g.size() / 2; k > 0; k /= 2) {
      if (index < k)
        sm_A[index] += sm_A[index + k];
      __syncthreads();
    }
    if (g.thread_rank() == 0) {
      atomicAdd(&y[row], sm_A[0]);
    }
  }
}
#endif /*KERNELS_H_*/
