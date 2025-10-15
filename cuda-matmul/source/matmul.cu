#include "json.hpp"

#include <device_launch_parameters.h>

#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
#include <numbers>
#include <random>

using nlohmann::json;

/**
 * LOOK: This is a macro that calls a CUDA function and checks for errors
 */
#define CUDA(call) do {                             \
    cudaError_t e = (call);                         \
    if (e == cudaSuccess) break;                    \
    fprintf(stderr, __FILE__":%d: %s (%d)\n",       \
            __LINE__, cudaGetErrorString(e), e);    \
    exit(1);                                        \
} while (0)

/**
 * LOOK: Use DIMS as a struct to store launch configurations.
 * dimBlock is dimension of block (threads), and dimGrid is dimension of the launch (number of blocks).
 * dim3 is a CUDA provided type, which as 3 components - x, y, z, which are initialized by default to 1.
 */
struct DIMS
{
    dim3 dimBlock;
    dim3 dimGrid;
};

/**
 * Function to divide up a size into part, each of div size. Similar to ceil.
 * Returns the number of parts.
 * Example, divup(10, 3) = 4
 */
__host__ __device__
unsigned divup(unsigned size, unsigned div)
{
    // TODO: implement a 1 line function to return the divup operation.
    // Note: You only need to use addition, subtraction, and division operations.
    return (size + div - 1) / div;
}

const unsigned TILE = 16;

const json MatrixMultiplicationMode = {
    { "BASIC", 0 }, // Square matrices
    { "ADVANCE", 1 }, // Rectangular transposed matrices
    { "FULL", 2 }// Full variation
};

const json KernelMode = {
    { "NAIVE", "naive"},
    { "TILED", "tiled"}
};

struct MatrixSizes {
    unsigned sizeMX = 0;
    unsigned sizeXY = 0;
    unsigned sizeNY = 0;
};

void resultsToTableBasic(const json& results);
void resultsToTableAdvance(const json& results);

__global__ void matrixMultiplicationNaive(float* const matrixP, const float* const matrixM, const float* const matrixN,
                                          const unsigned sizeMX, const unsigned sizeNY, const unsigned sizeXY)
{
    unsigned px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= sizeMX || py >= sizeNY)
        return;

    float dot = 0.0;
    for (int k = 0; k < sizeXY; k++) {
        const float m = matrixM[k * sizeMX + px];
        const float n = matrixN[py * sizeXY + k];
        dot += m * n;
    }

    matrixP[py * sizeMX + px] = dot;
}

__global__ void matrixMultiplicationTiled(float* const matrixP, const float* const matrixM, const float* const matrixN,
                                          const unsigned sizeMX, const unsigned sizeNY, const unsigned sizeXY)
{
    __shared__ float sM[TILE][TILE];
    __shared__ float sN[TILE][TILE];

    unsigned px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned py = blockIdx.y * blockDim.y + threadIdx.y;

    float dot = 0.0f;

    const unsigned tileFactor = divup(sizeXY, TILE);
    for (unsigned t = 0; t < tileFactor; t++) {
        const unsigned tileOffset = t * TILE;

        if (px < sizeMX && (tileOffset + threadIdx.y) < sizeXY) {
            sM[threadIdx.y][threadIdx.x] = matrixM[px + sizeMX * (tileOffset + threadIdx.y)];
        } else {
            sM[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((tileOffset + threadIdx.x) < sizeXY && py < sizeNY) {
            sN[threadIdx.y][threadIdx.x] = matrixN[(tileOffset + threadIdx.x) + (py * sizeXY)];
        }
        else {
            sN[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (unsigned k = 0; k < TILE; k++)
            dot += sN[threadIdx.y][k] * sM[k][threadIdx.x];
        __syncthreads();
    }

    if (px < sizeMX && py < sizeNY)
        matrixP[py * sizeMX + px] = dot;

}

// Identity matrix
std::vector<float> createIdentityMatrix(const unsigned sizeX, const unsigned sizeY)
{
    return std::vector<float>(sizeX * sizeY, 0.0f);
}

// Random matrix
std::vector<float> createRandomMatrix(const unsigned sizeX, const unsigned sizeY)
{
    std::vector<float> m(sizeX * sizeY);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dis(0.0, 10.0); // Random numbers between 0 and 10

    for (float& val : m) {
        val = dis(gen);
    }

    return m;
}

template<typename T>
bool equalsEpsilon(T left, T right, T epsilon = 0.0)
{
    return std::fabs(left - right) <= epsilon;
}

bool compareMatrix(const std::vector<float>& gpuMatrix, const std::vector<float>& cpuMatrix, const unsigned sizeMX, const unsigned sizeNY, const unsigned sizeXY, const float epsilon = 1e-3)
{
    bool error = false;
    const float e = sizeXY * epsilon;
    for (unsigned y = 0; y < sizeNY; y++) {
        for (unsigned x = 0; x < sizeMX; x++) {
            const unsigned index = y * sizeMX + x;
            const float g = gpuMatrix[index];
            const float c = cpuMatrix[index];
            if (!equalsEpsilon(gpuMatrix[index], cpuMatrix[index], e)) {
                std::cerr << "Mismatch Error : GPU = " << g << " and CPU = " << c << " at index " << index << " with epsilon = " << e << " and difference = " << std::fabs(g - c) << std::endl;
                error = true;
                break;
            }
        }
        if (error)
            break;
    }

    return !error;
}

void run(json& results, const MatrixSizes& matrixSizes)
{
    const std::string kernelNaive = KernelMode["NAIVE"];
    const std::string kernelTiled = KernelMode["TILED"];

    const unsigned sizeMX = matrixSizes.sizeMX;
    const unsigned sizeXY = matrixSizes.sizeXY;
    const unsigned sizeNY = matrixSizes.sizeNY;

    results["size"]["mx"].push_back(sizeMX);
    results["size"]["xy"].push_back(sizeXY);
    results["size"]["ny"].push_back(sizeNY);

    std::vector<float> matrixM = createRandomMatrix(sizeMX, sizeXY);
    std::vector<float> matrixN = createRandomMatrix(sizeXY, sizeNY);
    std::vector<float> matrixP = createIdentityMatrix(sizeMX, sizeNY);
    std::vector<float> matrixPNaive = createIdentityMatrix(sizeMX, sizeNY);
    std::vector<float> matrixPTiled = createIdentityMatrix(sizeMX, sizeNY);

    //////////////////////////////////////////////////////////////////
    // CPU Matrix Multiplication
    //////////////////////////////////////////////////////////////////
    auto timerStart = std::chrono::high_resolution_clock::now();
    for (unsigned y = 0; y < sizeNY; y++) {
        for (unsigned x = 0; x < sizeMX; x++) {
            float sum = 0;
            for (unsigned k = 0; k < sizeXY; k++) {
                const float a = matrixM[k * sizeMX + x];
                const float b = matrixN[y * sizeXY + k];
                sum += a * b;
            }
            matrixP[y * sizeMX + x] = sum;
        }
    }
    auto timeEnd = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> cpuElapsedTime = timeEnd - timerStart;
    results["cpuTime"].push_back(cpuElapsedTime.count());

    //////////////////////////////////////////////////////////////////
    // CUDA Setup
    //////////////////////////////////////////////////////////////////
    float *d_matrixM, *d_matrixN, *d_matrixPNaive, *d_matrixPTiled;

    CUDA(cudaMalloc((void **)&d_matrixM, sizeMX * sizeXY * sizeof(float)));
    CUDA(cudaMalloc((void **)&d_matrixN, sizeXY * sizeNY * sizeof(float)));
    CUDA(cudaMemcpy(d_matrixM, matrixM.data(), sizeMX * sizeXY * sizeof(float), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(d_matrixN, matrixN.data(), sizeXY * sizeNY * sizeof(float), cudaMemcpyHostToDevice));

    CUDA(cudaMalloc((void **)&d_matrixPNaive, sizeMX * sizeNY * sizeof(float)));
    CUDA(cudaMalloc((void **)&d_matrixPTiled, sizeMX * sizeNY * sizeof(float)));

    DIMS dims;
    dims.dimBlock = dim3(TILE, TILE, 1);
    dims.dimGrid  = dim3(divup(sizeMX, dims.dimBlock.x),
                         divup(sizeNY, dims.dimBlock.y),
                         1);

    CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //////////////////////////////////////////////////////////////////
    // Naive Matrix Multiplication
    //////////////////////////////////////////////////////////////////
    cudaEventRecord(start, 0);

    matrixMultiplicationNaive<<<dims.dimGrid, dims.dimBlock>>>(d_matrixPNaive, d_matrixM, d_matrixN, sizeMX, sizeNY, sizeXY);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    CUDA(cudaMemcpy(matrixPNaive.data(), d_matrixPNaive, sizeMX * sizeNY * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA(cudaDeviceSynchronize());

    float naiveTime = 0.0f;
    cudaEventElapsedTime(&naiveTime, start, stop);

    bool naiveSuccess = compareMatrix(matrixPNaive, matrixP, sizeMX, sizeNY, sizeXY);

    if (naiveSuccess) {
        //////////////////////////////////////////////////////////////////
        // Benchmark Naive Matrix Multiplication
        //////////////////////////////////////////////////////////////////
        cudaEventRecord(start, 0);

        int iterations = 10;
        for (int i = 0; i < iterations; i++)
            matrixMultiplicationNaive<<<dims.dimGrid, dims.dimBlock>>>(d_matrixPNaive, d_matrixM, d_matrixN, sizeMX, sizeNY, sizeXY);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        CUDA(cudaDeviceSynchronize());

        cudaEventElapsedTime(&naiveTime, start, stop);
        naiveTime /= (1000 * iterations);
        //////////////////////////////////////////////////////////////////
    }

    results["status"][kernelNaive].push_back(naiveSuccess);
    results["gpuTime"][kernelNaive].push_back(naiveTime);

    //////////////////////////////////////////////////////////////////
    // Tiled Matrix Multiplication
    //////////////////////////////////////////////////////////////////
    cudaEventRecord(start, 0);

    matrixMultiplicationTiled<<<dims.dimGrid, dims.dimBlock>>>(d_matrixPTiled, d_matrixM, d_matrixN, sizeMX, sizeNY, sizeXY);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    CUDA(cudaMemcpy(matrixPTiled.data(), d_matrixPTiled, sizeMX * sizeNY * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA(cudaDeviceSynchronize());

    float tiledTime = 0.0f;
    cudaEventElapsedTime(&tiledTime, start, stop);

    bool tiledSuccess = compareMatrix(matrixPTiled, matrixP, sizeMX, sizeNY, sizeXY);

    if (tiledSuccess) {
        //////////////////////////////////////////////////////////////////
        // Benchmark Tiled Matrix Multiplication
        //////////////////////////////////////////////////////////////////
        cudaEventRecord(start, 0);

        int iterations = 10;
        for (int i = 0; i < iterations; i++)
            matrixMultiplicationTiled<<<dims.dimGrid, dims.dimBlock>>>(d_matrixPTiled, d_matrixM, d_matrixN, sizeMX, sizeNY, sizeXY);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        CUDA(cudaDeviceSynchronize());

        cudaEventElapsedTime(&tiledTime, start, stop);
        tiledTime /= (1000 * iterations);
        //////////////////////////////////////////////////////////////////
    }

    results["status"][kernelTiled].push_back(tiledSuccess);
    results["gpuTime"][kernelTiled].push_back(tiledTime);

    //////////////////////////////////////////////////////////////////
    // Clean up
    //////////////////////////////////////////////////////////////////
    CUDA(cudaFree(d_matrixM));
    CUDA(cudaFree(d_matrixN));
    CUDA(cudaFree(d_matrixPNaive));
    CUDA(cudaFree(d_matrixPTiled));
}

int main(int argc, char* argv[])
{
    json results = {
        { "size", {
            { "mx", json::array() } ,
            { "xy", json::array() } ,
            { "ny", json::array() }
        }},
        { "status", {
            { "naive", json::array() },
            { "tiled", json::array() }
        }},
        { "cpuTime", json::array() } ,
        { "gpuTime", {
            { "naive", json::array() },
            { "tiled", json::array() }
        }},
    };

    MatrixSizes matrixSizes;
    
    const unsigned maxSizeX = (1 << 11) + 1;
    const unsigned maxSizeY = (1 << 11) + 1;
    const unsigned maxSizeZ = (1 << 11) + 1;

    const auto matrixMultiplicationMode = MatrixMultiplicationMode["BASIC"];

    if (matrixMultiplicationMode == MatrixMultiplicationMode["BASIC"]) {
        for (float size = 16; size < maxSizeX; size *= std::numbers::sqrt2) { // Test NPOT Size too
            const unsigned truncatedSize = (unsigned) size;
            matrixSizes.sizeMX = truncatedSize;
            matrixSizes.sizeXY = truncatedSize;
            matrixSizes.sizeNY = truncatedSize;
            std::cout << "Running CUDA Matrix Multiplication for size "
                << "[ " << matrixSizes.sizeMX << ", " << matrixSizes.sizeXY << " ]"
                << std::endl;
            run(results, matrixSizes);
        }
    } else if (matrixMultiplicationMode == MatrixMultiplicationMode["ADVANCE"]) {
        for (float sizeY = 16; sizeY < maxSizeY; sizeY *= std::numbers::sqrt2) { // Test NPOT Size too
            const unsigned truncatedSizeY = (unsigned) sizeY;
            matrixSizes.sizeXY = truncatedSizeY;
            for (float sizeX = 16; sizeX < maxSizeX; sizeX *= std::numbers::sqrt2) { // Test NPOT Size too
                const unsigned truncatedSizeX = (unsigned) sizeX;
                matrixSizes.sizeMX = truncatedSizeX;
                matrixSizes.sizeNY = truncatedSizeX;
                std::cout << "Running CUDA Matrix Multiplication for size "
                    << "[ " << matrixSizes.sizeMX << ", " << matrixSizes.sizeXY << " ] x "
                    << "[ " << matrixSizes.sizeXY << ", " << matrixSizes.sizeNY << " ]"
                    << std::endl;
                run(results, matrixSizes);
            }
        }
    } else if (matrixMultiplicationMode == MatrixMultiplicationMode["FULL"]) {
        for (float sizeZ = 16; sizeZ < maxSizeZ; sizeZ *= std::numbers::sqrt2) { // Test NPOT Size too
            matrixSizes.sizeNY = (unsigned)sizeZ;
            for (float sizeY = 16; sizeY < maxSizeY; sizeY *= std::numbers::sqrt2) { // Test NPOT Size too
                matrixSizes.sizeXY = (unsigned)sizeY;
                for (float sizeX = 16; sizeX < maxSizeX; sizeX *= std::numbers::sqrt2) { // Test NPOT Size too
                    matrixSizes.sizeMX = (unsigned)sizeX;
                    std::cout << "Running CUDA Matrix Multiplication for size "
                        << "[ " << matrixSizes.sizeMX << ", " << matrixSizes.sizeXY << " ] x "
                        << "[ " << matrixSizes.sizeXY << ", " << matrixSizes.sizeNY << " ]"
                        << std::endl;
                    run(results, matrixSizes);
                }
            }
        }
    } else {
        std::cerr << "Invalid Matrix Multiplication Mode" << std::endl;
        return -1;
    }

    std::cout << std::endl
        << "************************************" << std::endl
        << "***Matrix Multiplication Complete***" << std::endl
        << "************************************" << std::endl;

    if (matrixMultiplicationMode == MatrixMultiplicationMode["BASIC"]) {
        resultsToTableBasic(results);
    } else {
        resultsToTableAdvance(results);
    }
}

void resultsToTableBasic(const json &results) {
    const std::string kernelNaive = KernelMode["NAIVE"];
    const std::string kernelTiled = KernelMode["TILED"];

    const unsigned resultsLength = results["cpuTime"].size();

    std::vector<std::string> matrixSizes(resultsLength);

    std::vector<float> naiveCPUSpeedUps(resultsLength);
    std::vector<float> tiledCPUSpeedUps(resultsLength);
    std::vector<float> tiledGPUSpeedUps(resultsLength);

    for (unsigned i = 0; i < resultsLength; i++) {
        const float cpuTime = results["cpuTime"][i].get<float>();
        const float gpuNaiveTime = results["gpuTime"][kernelNaive][i].get<float>();
        const float gpuTiledTime = results["gpuTime"][kernelTiled][i].get<float>();

        naiveCPUSpeedUps[i] = cpuTime / gpuNaiveTime;
        tiledCPUSpeedUps[i] = cpuTime / gpuTiledTime;
        tiledGPUSpeedUps[i] = gpuNaiveTime / gpuTiledTime;

        matrixSizes[i] = std::to_string(results["size"]["mx"][i].get<unsigned>()) + " x " + std::to_string(results["size"]["mx"][i].get<unsigned>());
    }

    const json tableData = {
        { "Matrix", matrixSizes },
        { "CPU Time (sec)", results["cpuTime"] },
        { "CUDA Naive MatMul", {
            { "Success", results["status"][kernelNaive] },
            { "Time (sec)", results["gpuTime"][kernelNaive] }
        }},
        { "CUDA Tiled MatMul", {
            { "Success", results["status"][kernelTiled] },
            { "Time (sec)", results["gpuTime"][kernelTiled] }
        }},
        { "Speed Up", {
            { "Naive vs CPU", naiveCPUSpeedUps },
            { "Tiled vs CPU", tiledCPUSpeedUps },
            { "Tiled vs Naive", tiledGPUSpeedUps },
        }}
    };

    // write prettified JSON to another file
    std::ofstream jsonFileOutput("matrix_multiplication_basic.json");
    jsonFileOutput << std::setw(4) << tableData << std::endl;
}

void resultsToTableAdvance(const json &results) {
    const std::string kernelNaive = KernelMode["NAIVE"];
    const std::string kernelTiled = KernelMode["TILED"];

    const unsigned resultsLength = results["cpuTime"].size();

    std::vector<std::string> matrixMSizes(resultsLength);
    std::vector<std::string> matrixNSizes(resultsLength);
    std::vector<std::string> matrixPSizes(resultsLength);

    std::vector<float> naiveCPUSpeedUps(resultsLength);
    std::vector<float> tiledCPUSpeedUps(resultsLength);
    std::vector<float> tiledGPUSpeedUps(resultsLength);

    for (unsigned i = 0; i < resultsLength; i++) {
        const float cpuTime = results["cpuTime"][i].get<float>();
        const float gpuNaiveTime = results["gpuTime"][kernelNaive][i].get<float>();
        const float gpuTiledTime = results["gpuTime"][kernelTiled][i].get<float>();

        naiveCPUSpeedUps[i] = cpuTime / gpuNaiveTime;
        tiledCPUSpeedUps[i] = cpuTime / gpuTiledTime;
        tiledGPUSpeedUps[i] = gpuNaiveTime / gpuTiledTime;

        matrixMSizes[i] = std::to_string(results["size"]["mx"][i].get<unsigned>()) + " x " + std::to_string(results["size"]["xy"][i].get<unsigned>());
        matrixNSizes[i] = std::to_string(results["size"]["xy"][i].get<unsigned>()) + " x " + std::to_string(results["size"]["ny"][i].get<unsigned>());
        matrixPSizes[i] = std::to_string(results["size"]["mx"][i].get<unsigned>()) + " x " + std::to_string(results["size"]["ny"][i].get<unsigned>());
    }

    const json tableData = {
        { "Matrix", {
            { "M", matrixMSizes },
            { "N", matrixNSizes },
            { "P", matrixPSizes },
        }},
        { "CPU Time (sec)", results["cpuTime"] },
        { "CUDA Naive MatMul", {
            { "Success", results["status"][kernelNaive] },
            { "Time (sec)", results["gpuTime"][kernelNaive] }
        }},
        { "CUDA Tiled MatMul", {
            { "Success", results["status"][kernelTiled] },
            { "Time (sec)", results["gpuTime"][kernelTiled] }
        }},
        { "Speed Up", {
            { "Naive vs CPU", naiveCPUSpeedUps },
            { "Tiled vs CPU", tiledCPUSpeedUps },
            { "Tiled vs Naive", tiledGPUSpeedUps },
        }}
    };

    // write prettified JSON to another file
    std::ofstream jsonFileOutput("matrix_multiplication_advance.json");
    jsonFileOutput << std::setw(4) << tableData << std::endl;
}

