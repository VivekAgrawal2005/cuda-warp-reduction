#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " \
                  << cudaGetErrorString(err) << "\n"; \
        return 1; \
    } \
} while (0)

constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 1024;


// ==========================================
// 1. SHARED-MEMORY REDUCTION
// ==========================================

__global__ void reduceShared(
    const int* input,
    int* partial,
    int n
) {
    __shared__ int s[THREADS];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int step = gridDim.x * blockDim.x;

    int sum = 0;

    // Grid-stride loop
    for (; i < n; i += step) {
        sum += input[i];
    }

    s[tid] = sum;
    __syncthreads();

    // Tree reduction in shared memory
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2) {

        if (tid < stride) {
            s[tid] += s[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = s[0];
    }
}


// ==========================================
// 2. WARP-SHUFFLE REDUCTION
// ==========================================

__global__ void reduceShuffle(
    const int* input,
    int* partial,
    int n
) {
    __shared__ int warpSums[THREADS / 32];

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;

    int i = blockIdx.x * blockDim.x + tid;
    int step = gridDim.x * blockDim.x;

    int sum = 0;

    // Grid-stride loop
    for (; i < n; i += step) {
        sum += input[i];
    }

    unsigned int mask = 0xffffffff;

    // Reduce inside each warp
    for (int offset = 16;
         offset > 0;
         offset /= 2) {

        sum += __shfl_down_sync(
            mask, sum, offset
        );
    }

    // One lane per warp stores its result
    if (lane == 0) {
        warpSums[warp] = sum;
    }

    __syncthreads();

    // First warp reduces the warp totals
    if (warp == 0) {

        int total = (lane < THREADS / 32)
            ? warpSums[lane]
            : 0;

        for (int offset = 16;
             offset > 0;
             offset /= 2) {

            total += __shfl_down_sync(
                mask, total, offset
            );
        }

        if (lane == 0) {
            partial[blockIdx.x] = total;
        }
    }
}


// ==========================================
// 3. FINAL REDUCTION KERNEL
// ==========================================

__global__ void finishShared(
    const int* partial,
    int* output,
    int count
) {
    __shared__ int s[THREADS];

    int tid = threadIdx.x;
    int sum = 0;

    // Each thread may process multiple partials
    for (int i = tid;
         i < count;
         i += blockDim.x) {

        sum += partial[i];
    }

    s[tid] = sum;
    __syncthreads();

    // Final tree reduction
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2) {

        if (tid < stride) {
            s[tid] += s[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        output[0] = s[0];
    }
}


// ==========================================
// 4. MAIN PROGRAM
// ==========================================

int main() {

    std::vector<int> sizes = {
        32,
        256,
        1000,
        65536,
        1000000
    };

    const int ITERS = 500;
    const int WARMUP = 50;

    for (int n : sizes) {

        std::cout << "\n-----------------------------\n";
        std::cout << "Input size: " << n << "\n";


        // ----------------------------------
        // A. CREATE INPUT ON CPU
        // ----------------------------------

        std::vector<int> h_input(n);

        for (int i = 0; i < n; i++) {
            h_input[i] = (i % 10) + 1;
        }

        long long expected = std::accumulate(
            h_input.begin(),
            h_input.end(),
            0LL
        );


        // ----------------------------------
        // B. CALCULATE GRID SIZE
        // ----------------------------------

        int blocks = std::min(
            (n + THREADS - 1) / THREADS,
            MAX_BLOCKS
        );


        // ----------------------------------
        // C. ALLOCATE GPU MEMORY
        // ----------------------------------

        int *d_input = nullptr;
        int *d_partial = nullptr;
        int *d_output = nullptr;

        CHECK(cudaMalloc(
            &d_input,
            n * sizeof(int)
        ));

        CHECK(cudaMalloc(
            &d_partial,
            blocks * sizeof(int)
        ));

        CHECK(cudaMalloc(
            &d_output,
            sizeof(int)
        ));


        // ----------------------------------
        // D. COPY INPUT TO GPU
        // ----------------------------------

        CHECK(cudaMemcpy(
            d_input,
            h_input.data(),
            n * sizeof(int),
            cudaMemcpyHostToDevice
        ));


        // ----------------------------------
        // E. CORRECTNESS TEST: SHARED
        // ----------------------------------

        int sharedResult = 0;
        int shuffleResult = 0;

        reduceShared<<<blocks, THREADS>>>(
            d_input,
            d_partial,
            n
        );
        CHECK(cudaGetLastError());

        finishShared<<<1, THREADS>>>(
            d_partial,
            d_output,
            blocks
        );
        CHECK(cudaGetLastError());

        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(
            &sharedResult,
            d_output,
            sizeof(int),
            cudaMemcpyDeviceToHost
        ));


        // ----------------------------------
        // F. CORRECTNESS TEST: SHUFFLE
        // ----------------------------------

        reduceShuffle<<<blocks, THREADS>>>(
            d_input,
            d_partial,
            n
        );
        CHECK(cudaGetLastError());

        finishShared<<<1, THREADS>>>(
            d_partial,
            d_output,
            blocks
        );
        CHECK(cudaGetLastError());

        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(
            &shuffleResult,
            d_output,
            sizeof(int),
            cudaMemcpyDeviceToHost
        ));


        // ----------------------------------
        // G. PRINT CORRECTNESS RESULTS
        // ----------------------------------

        bool passed =
            (sharedResult == expected) &&
            (shuffleResult == expected);

        std::cout
            << "N=" << n
            << " expected=" << expected
            << " shared=" << sharedResult
            << " shuffle=" << shuffleResult
            << " "
            << (passed ? "PASS" : "FAIL")
            << "\n";


        // ----------------------------------
        // H. WARM-UP
        // ----------------------------------

        for (int i = 0; i < WARMUP; i++) {

            reduceShared<<<blocks, THREADS>>>(
                d_input,
                d_partial,
                n
            );

            finishShared<<<1, THREADS>>>(
                d_partial,
                d_output,
                blocks
            );

            reduceShuffle<<<blocks, THREADS>>>(
                d_input,
                d_partial,
                n
            );

            finishShared<<<1, THREADS>>>(
                d_partial,
                d_output,
                blocks
            );
        }

        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());


        // ----------------------------------
        // I. CREATE CUDA EVENTS
        // ----------------------------------

        cudaEvent_t start, stop;

        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));


        // ----------------------------------
        // J. TIME SHARED-MEMORY METHOD
        // ----------------------------------

        CHECK(cudaEventRecord(start));

        for (int i = 0; i < ITERS; i++) {

            reduceShared<<<blocks, THREADS>>>(
                d_input,
                d_partial,
                n
            );

            finishShared<<<1, THREADS>>>(
                d_partial,
                d_output,
                blocks
            );
        }

        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float sharedMs = 0.0f;

        CHECK(cudaEventElapsedTime(
            &sharedMs,
            start,
            stop
        ));


        // ----------------------------------
        // K. TIME WARP-SHUFFLE METHOD
        // ----------------------------------

        CHECK(cudaEventRecord(start));

        for (int i = 0; i < ITERS; i++) {

            reduceShuffle<<<blocks, THREADS>>>(
                d_input,
                d_partial,
                n
            );

            finishShared<<<1, THREADS>>>(
                d_partial,
                d_output,
                blocks
            );
        }

        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float shuffleMs = 0.0f;

        CHECK(cudaEventElapsedTime(
            &shuffleMs,
            start,
            stop
        ));


        // ----------------------------------
        // L. PRINT TIMINGS
        // ----------------------------------

        float sharedUs =
            sharedMs * 1000.0f / ITERS;

        float shuffleUs =
            shuffleMs * 1000.0f / ITERS;

        std::cout
            << "Shared avg (us): "
            << sharedUs << "\n";

        std::cout
            << "Shuffle avg (us): "
            << shuffleUs << "\n";

        if (shuffleUs > 0.0f) {
            std::cout
                << "Speedup (shared/shuffle): "
                << sharedUs / shuffleUs
                << "x\n";
        }


        // ----------------------------------
        // M. CLEANUP
        // ----------------------------------

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));

        CHECK(cudaFree(d_input));
        CHECK(cudaFree(d_partial));
        CHECK(cudaFree(d_output));
    }

    return 0;
}
