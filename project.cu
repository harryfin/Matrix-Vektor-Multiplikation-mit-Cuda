///////////////////////////////////////////////////////////////////////////////
// Harry Findeis
// CDS19
// Projektaufgabe
// Matrix-Vektor-Implementierung
///////////////////////////////////////////////////////////////////////////////

#include "hostfunctions.h"
#include "kernels.h"
#include <cmath>
#include <cooperative_groups.h>
#include <cuda.h>
#include <iostream>
#include <random>

using namespace cooperative_groups;

const int NumberOfKernels = 4; // 4 -> relevant für Projekt

/*

Eingabeparameter: int m, int m, int k
  m..  m in [0,15]
          M = 2^m : Dimension für A und y
  n..  n in [0,15]
          N = 2^n : Dimension für A und x
  k..  k in [5,10]
          Threads = 2^k : Anzahl der Threads per Block


  Default (10,10,10)

///////////////////////////////////////////////////////////////////////////////

Beispielaufruf. ./a.out 10 10 1024

///////////////////////////////////////////////////////////////////////////////

Funktion des Programmes:
  Lösung einer Matrix-Vektor Multiplikation A * x = y mittels Cuda
  Matrix A und Vektor x werden mit einer Zufallszahlen initialisiert

Dimensionen der Operation:
  Matrix A..  (MxN) mit M Zeilen und N Spalten
  Vektor x..  (N) mit N Zeilen
  Vektor y..  (M) mit M Zeilen


///////////////////////////////////////////////////////////////////////////////

Bezeichnungen der Kernel:

Nr   | Aufgabenstellung |   Name
---------------------------------------------
0    |   1              |   Shared Memory Reduction Kernel
1    |   1.1            |   Shared Memory Atomic Kernel
2    |   1.2            |   Shared Memory Intra-Grid Kernel
3    |   2              |   All Atomic Kernel
4    |   ---            |   Shared Memory Intra-Block Kernel

//////////////////////////////////////////////////////////////////////////////

Print Ausgabe (debug-mode off)

  M  |  N  | Blocks  |  Threads |  Zeitmessungen: Kernel (4) - Cachemodi (4)
----------------------------------------------------
M,N,Blocks,Threads,0-0,0-1,0-2,0-3,1-0,1-1,.....
----------------------------------------------------
- M.. Anzahl der Zeilen von A und x
- N.. Anzahl der Spalten von A und Zeilen  von y
- Blocks.. Anzahl der Verwendeten Blocks insgesamt
- Zeitmessung.. Dauer der Kernelausführung im Millisekunden
  - Knernel-Nr.. {0,1,2,3} (siehe oben Bezeichnungen der Kernel)
    - Cachemodus:.. {0,1,2,3}
      - 0.. cudaFuncCachePreferNone
      - 1.. cudaFuncCachePreferL1
      - 2.. cudaFuncCachePreferShared
      - 3.. cudaFuncCachePreferEqual

///////////////////////////////////////////////////////////////////////////////

Print Ausgabe (debug-mode on)

- Programmparameter
- Ausgabe Kernelausführung
    - Kernelname
    - Zeitmessungen
    - CacheConfig
    - Kernel Checkresult
- Fehlermeldungen
*/

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  int debug = 1;   // 0.. off 1.. on
  int unified = 1; // 1.. Unified (empfohlen) sonst 0.. (experimentel)
  int mIn = 10;    // Bitverschiebung für M
  int nIn = 10;    // Bitverschiebung für N
  int thIn = 10;   // Threadvorschlag (Falls M<thin -> THREADS=M)
  if (argc > 1) {
    mIn = atoi(argv[1]);
    if (argc > 2)
      nIn = atoi(argv[2]);
    if (argc > 3)
      thIn = atoi(argv[3]);
  } else {
    printf("Usage: %s [m] [n] [k] \n", argv[0]);
  }
  if (mIn > 15) {
    std::cout << "Error: First Parameter to big -> set 10" << '\n';
    mIn = 10;
  }
  if (mIn < 0) {
    std::cout << "Error: First Parameter to small -> set 10" << '\n';
    mIn = 10;
  }
  if (nIn > 15) {
    std::cout << "Error: Second Parameter to big -> set 10" << '\n';
    nIn = 10;
  }
  if (nIn < 0) {
    std::cout << "Error: Second Parameter to small -> set 10" << '\n';
    nIn = 10;
  }
  if (thIn < 5) {
    std::cout << "Error: Third Parameter to small-> set 10 \n";
    thIn = 10;
  }
  if (thIn > 10) {
    std::cout << "Error: Threadanzahl zu groß -> set 10" << '\n';
    thIn = 10;
  }

  // Padding mit Addition erzwingbar
  const int M = (1 << mIn) + 0;
  const int N = (1 << nIn) + 0;

  // Threats per Blockdimension
  int thCheck = 1 << thIn;
  // Falls N<thCheck nur N Threads starten (Matrix hat keine gute Größe)
  const int THREADS = (N <= thCheck) ? N : thCheck;
  // Padding Matrix and Vektor
  const int N_PADDING = ceil((float)N / (float)THREADS) * THREADS;
  // Anzahl der Blocks in x-Richtung
  int blocksInX = N_PADDING / THREADS;

  if (debug == 1) {
    std::cout << "\nMatrix-Vector Multipikation"
                 "\n---------------------------------------------------"
              << "\nInputSize: M: " << M << " | N:         " << N
              << " | Threads: " << thCheck << '\n'
              << "REALSIZE:  M: " << M << " | N Padding: " << N_PADDING
              << " | Threads: " << THREADS << "\n\n";
    printf(
        "CUDA kernel launch with %d blocks of %d threads. \nBlockdimension:\n"
        "x-Direction: %d\ny-Direction: %d \n\n",
        blocksInX * M, THREADS, blocksInX, M);
    std::cout << "\n\n";
  }

  // Cuda Events für Zeitmessung
  cudaEvent_t startEvent, stopEvent;
  errorCheck(cudaEventCreate(&startEvent), "create start event");
  errorCheck(cudaEventCreate(&stopEvent), "create stop event");

  size_t sizeMN = N_PADDING * M * sizeof(float),
         sizeN = N_PADDING * sizeof(float), sizeM = M * sizeof(float);

  float *A, *x, *y;
  float *A_h, *x_h, *y_h;
  if (unified == 1) {
    // Unified Memory - kann von CPU und GPU gelesen und beschrieben werden.
    errorCheck(cudaMallocManaged(&A, sizeMN), "alloc A");
    errorCheck(cudaMallocManaged(&x, sizeN), "alloc x");
    errorCheck(cudaMallocManaged(&y, sizeM), "alloc y");
    // Initialisirie Matrix und Vektor
    init(A, x, M, N, N_PADDING);
  } else {
    // Allocier auf Host
    A_h = (float *)malloc(sizeMN);
    x_h = (float *)malloc(sizeN);
    y_h = (float *)malloc(sizeM);
    // Initialisirie Matrix und Vektor
    init(A_h, x_h, M, N, N_PADDING);

    errorCheck(cudaMalloc((void **)&A, sizeMN), "alloc d_A");
    errorCheck(cudaMalloc((void **)&x, sizeN), "alloc d_x");
    errorCheck(cudaMalloc((void **)&y, sizeM), "alloc d_y");
    cudaMemcpy(A, A_h, sizeMN, cudaMemcpyHostToDevice);
    cudaMemcpy(x, x_h, sizeN, cudaMemcpyHostToDevice);
  }

  if (debug != 1) {
    std::cout << M << "," << N << "," << blocksInX * M << "," << THREADS;
  }

  // Launch alle Kernel
  for (int kernel = 0; kernel < NumberOfKernels; kernel++) {
    // Alle CacheConfigurationen starten
    for (int cacheConfig = 0; cacheConfig < 4; cacheConfig++) {
      launchKernel(A, x, y, startEvent, stopEvent, kernel, cacheConfig, debug,
                   blocksInX, M, N, N_PADDING, THREADS, unified, sizeM, y_h);
    }
  }
  if (debug != 1) {
    std::cout << '\n';
  }
  // Konolenausgabe
  // printMat(A, M, N, N_PADDING);
  // printVec(x, N);
  // printVec(y, M);

  // Zerstöre Cuda Events
  errorCheck(cudaEventDestroy(startEvent), "destroy-StartEvent");
  errorCheck(cudaEventDestroy(stopEvent), "destroy-StopEvent");

  // Speicher freen
  errorCheck(cudaFree(A), "free A");
  errorCheck(cudaFree(x), "free x");
  errorCheck(cudaFree(y), "free y");

  if (unified != 1) {
    free(A_h);
    free(x_h);
    free(y_h);
  }

  if (debug == 1) {
    printf("\nHost beendet Programm\n");
  }
  return 0;
}
