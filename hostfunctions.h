#ifndef HOSTFUNCTIONS_H_
#define HOSTFUNCTIONS_H_

#include "kernels.h"
#include <cmath>
#include <cooperative_groups.h>
#include <iostream>
#include <random>
using namespace cooperative_groups;

///////////////////////////////////////////////////////////////////////////////
// Konsolenausgabe Matrix
__host__ void printMat(float *u, int rows, int cols, int realRowLength) {
  std::cout << '\n';
  for (size_t x = 0; x < rows; x++) {
    for (size_t y = 0; y < cols; y++) {
      std::cout << u[x * realRowLength + y] << " ";
    }
    std::cout << '\n';
  }
}
///////////////////////////////////////////////////////////////////////////////
// Konsolenausgabe Vektor
__host__ void printVec(float *u, int size) {
  std::cout << '\n';
  for (size_t x = 0; x < size; x++) {
    std::cout << u[x] << '\n';
  }
}
///////////////////////////////////////////////////////////////////////////////
// Initialisiere Vektor und Matrix mit Zufallszahlen
__host__ void init(float *A, float *x, int const M, int const N,
                   int const N_PADDING) {
  std::random_device rd;
  std::default_random_engine gen;
  gen.seed(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  // std::uniform_int_distribution<> dis(0, 9);

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      // A[i * N_PADDING + j] = 1;
      A[i * N_PADDING + j] = (float)dis(gen);
    }
  }
  for (size_t i = 0; i < N; i++) {
    // x[i] = 1.0;
    x[i] = (float)dis(gen);
  }
}
///////////////////////////////////////////////////////////////////////////////
// Ergebniss checken - float tauglich
__host__ int checkResults(float *a, float *b, float *y, int const M,
                          int const N, int const N_PADDING) {
  for (int j = 0; j < M; j++) {
    float sum = 0.0;
    for (int i = 0; i < N; i++) {
      sum += a[i + j * N_PADDING] * b[i];
    }
    if (fabs(1.0 - (float)y[j] / (float)sum) > 1e-4) {
      std::cout << "Chech Result Error:" << std::endl;
      std::cout << "Error occured in execution in line" << j << std::endl;
      std::cout << sum << "(sollte) ungleich " << y[j] << "(ist)" << '\n';
      return (-1);
    }
  }
  return 0;
}
///////////////////////////////////////////////////////////////////////////////
// Fehlermeldung bei Cuda-Error
__host__ void errorCheck(cudaError_t err, const char *message) {
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed in %s (error code %s)!\n", message,
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
///////////////////////////////////////////////////////////////////////////////
// starte den intraGridKernel
// Nutzt temporären Buffer
// Ruft IntraGridKernel mehrmals auf bis alle Ergebnisse berechnet sind
__host__ void launchIntraGridKernel(float *A, float *x, float *y, int kernel,
                                    int cacheConfig, int debug, int blocksInX,
                                    int M, int N, int N_PADDING, int THREADS,
                                    dim3 blocks, dim3 threads) {

  float *intraGridKernelBuffer;
  int bufferSize = blocksInX * M * sizeof(float);
  errorCheck(cudaMallocManaged(&intraGridKernelBuffer, bufferSize),
             "alloc intraGridKernelBuffer");
  int supportsCoopLaunch = 0;
  int dev = 0;
  cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch,
                         dev);
  int rowShift = 0;
  int numBlocksPerSm = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm,
                                                smIntraGridKernel, THREADS, 0);
  int blocksAvailable = deviceProp.multiProcessorCount * numBlocksPerSm;
  int rowsInWork = blocksAvailable / blocksInX;

  dim3 dimGrid(blocksInX, rowsInWork, 1);
  dim3 dimBlock(THREADS, 1, 1);
  void *kernelArgs[] = {(void *)&A,         (void *)&x,
                        (void *)&y,         (void *)&M,
                        (void *)&N,         (void *)&N_PADDING,
                        (void *)&THREADS,   (void *)&intraGridKernelBuffer,
                        (void *)&blocksInX, (void *)&rowShift,
                        (void *)&rowsInWork};

  if (blocksAvailable < blocksInX) {
    printf("Grid Kernel nicht ausführbar, da x Dimension zu groß für "
           "verfügbare Blocks\n");
    return;
  }
  while (rowShift < M) {
    cudaLaunchCooperativeKernel((void *)smIntraGridKernel, dimGrid, dimBlock,
                                kernelArgs, NULL);
    cudaDeviceSynchronize();
    rowShift += rowsInWork;
  }

  // free Buffer
  errorCheck(cudaFree(intraGridKernelBuffer), "free intraGridKernelBuffer");
}
///////////////////////////////////////////////////////////////////////////////
// starte Shared Memory Kernel und den zugehörigen SharedMemoryReduktionskernel
// Nutzt temporären Buffer
// Ruft SharedMemoryReduktionskernel mehrmals auf bis alle Ergebnisse berechnet
// sind
__host__ void launchSmKernel(float *A, float *x, float *y, int kernel,
                             int cacheConfig, int debug, int blocksInX, int M,
                             int N, int N_PADDING, int THREADS, dim3 blocks,
                             dim3 threads) {

  // Buffer anlegen zum Einsammeln der Zwischensummen
  float *smKernelBuffer;
  int bufferSize = blocksInX * M * sizeof(float);
  errorCheck(cudaMallocManaged(&smKernelBuffer, bufferSize),
             "alloc smKernelBuffer");
  // Shared Memory Kernel - Jeder Block nutzt Shared Memory und schreibt
  // Teilsumme in Buffer
  smKernel<<<blocks, threads>>>(A, smKernelBuffer, x, y, M, N, N_PADDING,
                                blocksInX, THREADS);

  if (blocksInX > 1) {
    // starte für jeden Block aus smKernel einen Thread
    int logBlockX = (int)ceil(log2((float)(blocksInX)));
    int blockNumberActive = blocksInX;
    int blockNumberAcitveOld = blocksInX;

    while (blockNumberActive > 1) {
      blockNumberActive = (int)ceil((float)blockNumberActive / (float)THREADS);
      dim3 thrReduktion(THREADS, 1);
      dim3 blReduktion(blockNumberActive, M);
      reduktionSmKernel<<<blReduktion, thrReduktion, THREADS * sizeof(float)>>>(
          smKernelBuffer, y, blockNumberAcitveOld, blocksInX, M,
          blockNumberActive);
      blockNumberAcitveOld = blockNumberActive;
    }
    errorCheck(cudaFree(smKernelBuffer), "free smKernelBuffer");
  }
}
///////////////////////////////////////////////////////////////////////////////
// ruft Kernel einzeln auf
__host__ void myKernels(float *A, float *x, float *y, int kernel,
                        int cacheConfig, int debug, int blocksInX, int M, int N,
                        int N_PADDING, int THREADS) {

  cudaMemset(y, 0, M * sizeof(float)); // reset to zero

  dim3 threads(THREADS, 1); // Eindimensionale Blocks
  dim3 blocks(blocksInX, M);

  if (debug == 1) {
  }
  switch (kernel) {
  case 0:
    if (debug == 1) {
      std::cout << "smKernel" << std::endl;
    }
    launchSmKernel(A, x, y, kernel, cacheConfig, debug, blocksInX, M, N,
                   N_PADDING, THREADS, blocks, threads);
    break;
  case 1:
    if (debug == 1)
      std::cout << "smAtomicKernel" << std::endl;
    smAtomicKernel<<<blocks, threads>>>(A, x, y, M, N, N_PADDING, THREADS);
    break;
  case 2:
    if (debug == 1)
      std::cout << "smIntraGridKernel" << std::endl;
    launchIntraGridKernel(A, x, y, kernel, cacheConfig, debug, blocksInX, M, N,
                          N_PADDING, THREADS, blocks, threads);
    break;
  case 3:
    if (debug == 1)
      std::cout << "allAtomicKernel" << std::endl;
    allAtomicKernel<<<blocks, threads>>>(A, x, y, M, N, N_PADDING, THREADS);
    break;
  case 4:
    if (debug == 1)
      std::cout << "smIntraBlockKernel" << std::endl;
    smIntraBlockKernel<<<blocks, threads>>>(A, x, y, M, N, N_PADDING, THREADS);
    break;
  }
}

///////////////////////////////////////////////////////////////////////////////
// startet den Aufruf der Kernels mit CacheConfig und Zeitmessung
__host__ void launchKernel(float *A, float *x, float *y, cudaEvent_t startEvent,
                           cudaEvent_t stopEvent, int kernel, int cacheConfig,
                           int debug, int blocksInX, int const M, int const N,
                           int const N_PADDING, int const THREADS, int unified,
                           int sizeM, float *y_h) {

  // set CacheConfig Befehle vor Zeitmessung
  switch (cacheConfig) {
  case 0:
    cudaFuncSetCacheConfig(smKernel, cudaFuncCachePreferNone);
    cudaFuncSetCacheConfig(smAtomicKernel, cudaFuncCachePreferNone);
    cudaFuncSetCacheConfig(smIntraGridKernel, cudaFuncCachePreferNone);
    cudaFuncSetCacheConfig(smIntraBlockKernel, cudaFuncCachePreferNone);
    cudaFuncSetCacheConfig(allAtomicKernel, cudaFuncCachePreferNone);
    break;
  case 1:
    cudaFuncSetCacheConfig(smKernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(smAtomicKernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(smIntraGridKernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(smIntraBlockKernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(allAtomicKernel, cudaFuncCachePreferL1);
    break;
  case 2:
    cudaFuncSetCacheConfig(smKernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(smAtomicKernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(smIntraGridKernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(smIntraBlockKernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(allAtomicKernel, cudaFuncCachePreferShared);
  case 3:
    cudaFuncSetCacheConfig(smKernel, cudaFuncCachePreferEqual);
    cudaFuncSetCacheConfig(smAtomicKernel, cudaFuncCachePreferEqual);
    cudaFuncSetCacheConfig(smIntraGridKernel, cudaFuncCachePreferEqual);
    cudaFuncSetCacheConfig(smIntraBlockKernel, cudaFuncCachePreferEqual);
    cudaFuncSetCacheConfig(allAtomicKernel, cudaFuncCachePreferEqual);
    break;
  }

  float timeKernelMatVec;

  // Zeitmessung über Kernelausführung
  errorCheck(cudaEventRecord(startEvent), "start Zeit");
  myKernels(A, x, y, kernel, cacheConfig, debug, blocksInX, M, N, N_PADDING,
            THREADS);
  cudaDeviceSynchronize();
  errorCheck(cudaGetLastError(), "kernel error");
  errorCheck(cudaEventRecord(stopEvent), "stop Zeit");
  errorCheck(cudaEventSynchronize(stopEvent), "Sync Zeit");
  errorCheck(cudaEventElapsedTime(&timeKernelMatVec, startEvent, stopEvent),
             " elapse Zeit");
  int checkResult;

  if (unified == 1 && debug == 1) {
    checkResult = checkResults(A, x, y, M, N, N_PADDING);
  } else if (debug == 1) {
    cudaMemcpy(y_h, y, sizeM, cudaMemcpyDeviceToHost);
    checkResult = checkResults(A, x, y_h, M, N, N_PADDING);
  }
  if (debug == 1) {
    std::cout << "---------------------------------------------------" << '\n'
              << "Zeitmessung: " << timeKernelMatVec << '\n'
              << "---------------------------------------------------" << '\n';
    if (checkResult == 0) {
      std::cout << "CacheConfig :" << cacheConfig
                << " - Kernel erfolgreich abgeschlossen" << '\n';
    } else {
      std::cout << "CacheConfig :" << cacheConfig
                << " -Kernel nicht erfolgreich" << '\n';
    }
    std::cout << "---------------------------------------------------\n\n";

  } else {
    std::cout << "," << timeKernelMatVec;
  }
}

#endif /*HOSTFUNCTIONS_H_*/
