const WORKGROUP_SIZE = 16;
const TEN_POWER_THREE = Math.pow(10, 3);
const TEN_POWER_SIX = Math.pow(10, 6);
const TEN_POWER_NINE = Math.pow(10, 9);

function equalsEpsilon(left, right, epsilon) {
    epsilon = (epsilon !== undefined) ? epsilon : 0.0;
    return Math.abs(left - right) <= epsilon;
};

async function initGPUDevice()
{
    if (!navigator.gpu) {
        throw new Error("WebGPU not supported on this browser.");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("No appropriate GPUAdapter found.");
    }

    // Device initialization with performance timers
    const canTimestamp = adapter.features.has('timestamp-query');
    if (!canTimestamp) {
        throw new Error("Timestamps not available. Enable the right flags in your browser.");
    }

    const device = await adapter.requestDevice({
        requiredFeatures: ["timestamp-query"]
    });

    if (!device) {
        throw new Error("Need a browser that supports WebGPU.");
    }

    console.log(device);
    return device;
}

let queryIndex = 0;
function addWebGPUTimestamp(encoder, querySet) {
    encoder.writeTimestamp(querySet, queryIndex);
    queryIndex++;
}

async function webGPUMatrixMultiplication(device, matmulObject) {
    const matrixM = matmulObject.matrixM;
    const matrixN = matmulObject.matrixN;
    const matrixP = matmulObject.matrixP;

    const sizeMX = matmulObject.sizeMX;
    const sizeXY = matmulObject.sizeXY;
    const sizeNY = matmulObject.sizeNY;

    /**
     * Set up X Matrix on CPU and GPU
     */
    // Create buffer on GPU for X
    const matrixMGPUBuffer = device.createBuffer({
        label: "M Matrix GPU Buffer",
        mappedAtCreation: true,
        size: matrixM.byteLength,
        usage: GPUBufferUsage.STORAGE
    });
    // Copy matrixM to GPU
    const matrixMArrayBuffer = matrixMGPUBuffer.getMappedRange();
    new Float32Array(matrixMArrayBuffer).set(matrixM);
    matrixMGPUBuffer.unmap();

    /**
     * Set up Y Matrix on CPU and GPU
     */
    // Create buffer on GPU for Y
    // Alternate method of copying data to GPU
    const matrixNGPUBuffer = device.createBuffer({
        label: "Y Matrix GPU Buffer",
        size: matrixN.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(matrixNGPUBuffer, 0, matrixN);

    /**
     * Set up result P Matrix on GPU (no copy from host)
     */
    // Create buffer on GPU for P
    const matrixPGPUBuffer = device.createBuffer({
        label: "P Matrix GPU Buffer",
        size: matrixP.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    /**
     * Create uniform for sizes
     */
    const uniformBufferSize = 4 * Uint32Array.BYTES_PER_ELEMENT; // lets store size as float too
    const uniformBuffer = device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    });
    const uniformValues = new Uint32Array([sizeMX, sizeXY, sizeNY, 0]);
    device.queue.writeBuffer(uniformBuffer, 0, uniformValues);

    /**
     * Create the compute shader for MatrixMultiplication
     */
    const matmulComputeShaderModule = device.createShaderModule({
        label: "Matrix Multiplication Compute Shader",
        code: `
                struct sizesStruct {
                    sizeMX: u32,
                    sizeXY: u32,
                    sizeNY: u32,
                    extra: u32
                };

                var<workgroup> sM: array<f32, ${WORKGROUP_SIZE} * ${WORKGROUP_SIZE}>;
                var<workgroup> sN: array<f32, ${WORKGROUP_SIZE} * ${WORKGROUP_SIZE}>;

                @group(0) @binding(0) var<uniform> sizes: sizesStruct;

                @group(0) @binding(1) var<storage> matrixM: array<f32>;
                @group(0) @binding(2) var<storage> matrixN: array<f32>;
                @group(0) @binding(3) var<storage, read_write> matrixP: array<f32>;

                @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
                fn computeMain(
                        @builtin(global_invocation_id) globalIdx: vec3u, // blockIdx * blockdim + threadIdx
                        @builtin(workgroup_id) blockIdx: vec3u,
                        @builtin(local_invocation_id) threadIdx: vec3u) {

                    let sizeMX = sizes.sizeMX;
                    let sizeXY = sizes.sizeXY;
                    let sizeNY = sizes.sizeNY;

                    // Cannot do this with workgroupBarrier() - "error: 'workgroupBarrier' must only be called from uniform control flow"
                    //if (globalIdx.x >= sizeMX || globalIdx.y >= sizeNY) { return; }

                    var result = 0.0;
                    var tileFactor = u32(ceil(f32(sizeXY) / ${WORKGROUP_SIZE}));
                    for (var t = u32(0); t < tileFactor; t++) {
                        var tileOffset = t * ${WORKGROUP_SIZE};
                        if (globalIdx.x < sizeMX && (tileOffset + threadIdx.y) < sizeXY) {
                            sM[threadIdx.y * ${WORKGROUP_SIZE} + threadIdx.x] = matrixM[globalIdx.x + sizeMX * (tileOffset + threadIdx.y)];
                        }  else {
                            sM[threadIdx.y * ${WORKGROUP_SIZE} + threadIdx.x] = 0.0;
                        }

                        if ((tileOffset + threadIdx.x) < sizeXY && globalIdx.y < sizeNY) {
                            sN[threadIdx.y * ${WORKGROUP_SIZE} + threadIdx.x] = matrixN[(tileOffset + threadIdx.x) + (globalIdx.y * sizeXY)];
                        } else {
                            sN[threadIdx.y * ${WORKGROUP_SIZE} + threadIdx.x] = 0.0;
                        }

                        workgroupBarrier(); // syncthreads()

                        var elementCount = min(${WORKGROUP_SIZE}, sizeXY - tileOffset);
                        for (var k = u32(0); k < elementCount; k++) {
                            result += sN[threadIdx.y * ${WORKGROUP_SIZE} + k] * sM[k * ${WORKGROUP_SIZE} + threadIdx.x];
                        }

                        workgroupBarrier(); // syncthreads()
                    }

                    if ((globalIdx.x < sizeMX) && (globalIdx.y < sizeNY)) {
                        matrixP[globalIdx.y * sizeMX + globalIdx.x] = result;
                    }
                }
            `
    });

    /**
     * Create a Bind Group Layout.
     * The Bing Group Layout connects the different buffers into the locations in shaders.
     * It also sets up the type and read-write permissions
    */
    const matmulCSBindGroupLayout = device.createBindGroupLayout({
        label: "Matrix Multiplication Compute Shader Bind Group Layout",
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.COMPUTE, // Matrix Sizes - Read-only
            buffer: {}
        }, {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE, // M Matrix Input - Read-only
            buffer: {
                type: "read-only-storage"
            }
        }, {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE, // N Matrix Input - Read only
            buffer: {
                type: "read-only-storage"
            }
        }, {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE, // P Matrix Result - Write
            buffer: {
                type: "storage"
            }
        }]
    });

    /**
     * Create a Bind Group Layout.
     * The Bind Group connects the actual buffers to the locations.
     * This is essentially like the function parameters for calling shaders.
    */
    const matmulCSBindGroup = device.createBindGroup({
            label: "Matrix Multiplication Compute Shader Bind Group",
            layout: matmulCSBindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: uniformBuffer }
            }, {
                binding: 1,
                resource: { buffer: matrixMGPUBuffer }
            }, {
                binding: 2,
                resource: { buffer: matrixNGPUBuffer }
            }, {
                binding: 3,
                resource: { buffer: matrixPGPUBuffer }
            }],
        });

    /**
     * Create a pipeline layout.
     * A pipeline layout is a list of bind group layouts that one or more pipelines use. The order of the bind group layouts in the array needs to correspond with the @group attributes in the shaders. (This means that bindGroupLayout is associated with @group(0).)
    */
    const matmulCSPipelineLayout = device.createPipelineLayout({
        label: "Matrix Multiplication Compute Shader Pipeline Layout",
        bindGroupLayouts: [matmulCSBindGroupLayout],
    });

    /**
     * Create a draw and simulation pipelines.
     * The pipeline connects the shaders and the layouts, which in turn connects the buffers and bind groups.
    */
    const matmulCSPipeline = device.createComputePipeline({
        label: "Matrix Multiplication Compute Shader Pipeline",
        layout: matmulCSPipelineLayout,
        compute: {
            module: matmulComputeShaderModule,
            entryPoint: "computeMain",
        }
    });

    /**
     * In order to do pretty much anything else in WebGPU, you need to provide some commands to the GPU instructing it what to do.
     * To do this, have the device create a GPUCommandEncoder, which provides an interface for recording GPU commands.
     * The commands you want to send to the GPU are related to compute (or rendering), so the next step is to use the encoder to begin a Compute Pass.
     */
    const encoder = device.createCommandEncoder();

    /**
     * Performance timers
     */
    queryIndex = 0;
    const { querySet, performanceResolveBuffer, performanceResultBuffer } = (() => {
        const querySet = device.createQuerySet({
            type: 'timestamp',
            count: 2,
        });
        const performanceResolveBuffer = device.createBuffer({
            size: querySet.count * 8,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        const performanceResultBuffer = device.createBuffer({
            size: performanceResolveBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        return { querySet, performanceResolveBuffer, performanceResultBuffer };
    })();

    /**
     * Compute passes are the actual invocation of compute operations. Each one starts off with a beginComputePass() call,
     * which defines the pipelines, buffers, layouts that are the inputs and outputs.
     * It's important to know that simply making these calls does not cause the GPU to actually do anything. They're just recording commands for the GPU to do later.
     */
    addWebGPUTimestamp(encoder, querySet); // start time
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(matmulCSPipeline);
    computePass.setBindGroup(0, matmulCSBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(sizeMX / WORKGROUP_SIZE), Math.ceil(sizeNY / WORKGROUP_SIZE));
    computePass.end();
    addWebGPUTimestamp(encoder, querySet); // end time

    // Connect the performance timers.
    encoder.resolveQuerySet(querySet, 0, querySet.count, performanceResolveBuffer, 0);
    if (performanceResultBuffer.mapState === 'unmapped') {
        encoder.copyBufferToBuffer(performanceResolveBuffer, 0, performanceResultBuffer, 0, performanceResultBuffer.size);
    }

    /**
     * In order to create a GPUCommandBuffer, call finish() on the command encoder. The command buffer is an opaque handle to the recorded commands.
     * Submit the command buffer to the GPU using the queue of the GPUDevice.
     * The queue performs all GPU commands, ensuring that their execution is well ordered and properly synchronized.
     * The queue's submit() method takes in an array of command buffers, though in this case you only have one.
     * Once you submit a command buffer, it cannot be used again, so there's no need to hold on to it.
     * If you want to submit more commands, you need to build another command buffer. That's why it's fairly common to see those two steps collapsed into one.
     */
    device.queue.submit([encoder.finish()]);

    const gpu = matmulObject.gpu;
    if (performanceResultBuffer.mapState === 'unmapped') {
        performanceResultBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const times = new BigInt64Array(performanceResultBuffer.getMappedRange());
            gpu.elapsedTime = Number(times[1] - times[0]) / TEN_POWER_NINE; // seconds
            gpu.gflops = (matmulObject.ops / TEN_POWER_NINE) / gpu.elapsedTime;
            performanceResultBuffer.unmap();
        });
    }

    /**
     * Copy the result P Matrix to the GPU to check the computation.
     */
    // Get a GPU buffer for reading in an unmapped state.
    const matrixPStagingBuffer = device.createBuffer({
        size: matrixP.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // Create a new encoder as the previous one is no longer usable as a result of calling submit and finish.
    const resultCopyEncoder = device.createCommandEncoder();

    // Encode commands for copying buffer to buffer.
    resultCopyEncoder.copyBufferToBuffer(
        matrixPGPUBuffer /* source buffer */,
        0 /* source offset */,
        matrixPStagingBuffer /* destination buffer */,
        0 /* destination offset */,
        matrixP.byteLength /* size */
    );
    device.queue.submit([resultCopyEncoder.finish()]);

    // Read buffer
    await matrixPStagingBuffer.mapAsync(GPUMapMode.READ);
    const matrixPGPUResult = new Float32Array(matrixPStagingBuffer.getMappedRange());

    //printMatrix('P', matmulObject.matrixP, matmulObject.sizeMX, matmulObject.sizeNY);
    //printMatrix('GPU', matrixPGPUResult, matmulObject.sizeMX, matmulObject.sizeNY);

    /**
     * Compare GPU and CPU results
     */
    let error = false;
    let epsilon = (sizeMX > 1000 || sizeNY > 1000) ? 1e-2 : 1e-3;
    for (let y = 0; y < sizeNY; y++) {
        for (let x = 0; x < sizeMX; x++) {
            const index = y * sizeMX + x;
            if (!equalsEpsilon(matrixPGPUResult[index], matrixP[index], epsilon)) {
                console.log(`Mismatch Error: GPU = ${matrixPGPUResult[index]} and CPU = ${matrixP[index]} at index ${index} with Epsilon = ${epsilon}.`);
                error = true;
                //break;
            }
        }
        if (error)
            break;
    }

    matrixPStagingBuffer.unmap();

    if (!error) {
        console.log('Results match');
    }

    return !error;
}

function cpuMatrixMultiplication(matmulObject) {
    const matrixM = matmulObject.matrixM;
    const matrixN = matmulObject.matrixN;
    const matrixP = matmulObject.matrixP;

    const sizeMX = matmulObject.sizeMX;
    const sizeXY = matmulObject.sizeXY;
    const sizeNY = matmulObject.sizeNY;

    const startTime = performance.now();
    for (let y = 0; y < sizeNY; y++) {
        for (let x = 0; x < sizeMX; x++) {
            let sum = 0;
            for (let k = 0; k < sizeXY; k++) {
                const a = matrixM[k * sizeMX + x];
                const b = matrixN[y * sizeXY + k];
                sum += a * b;
            }
            matrixP[y * sizeMX + x] = sum;
        }
    }
    const endTime = performance.now();

    const cpu = matmulObject.cpu;
    cpu.elapsedTime = (endTime - startTime) / TEN_POWER_THREE; // seconds
    cpu.gflops = (matmulObject.ops / TEN_POWER_NINE) / cpu.elapsedTime;
}

// identity matrix
function createIdentityMatrix(sizeX, sizeY) {
    const m = new Float32Array(sizeX * sizeY);
    m.fill(0);
    return m;
}

// random matrix
function createRandomMatrix(sizeX, sizeY) {
    const nElements = sizeX * sizeY;
    const m = new Float32Array(nElements);
    for (var i = 0; i < nElements; i++) {
        m[i] = Math.trunc(Math.random() * 10);
    }
    return m;
}

function initMatrixMultiplication(matmulObject) {
    // Create X and Y Matrix on CPU
    matmulObject.ops = matmulObject.sizeMX * matmulObject.sizeNY * matmulObject.sizeXY * 2;
    matmulObject.matrixM = createRandomMatrix(matmulObject.sizeMX, matmulObject.sizeXY);
    matmulObject.matrixN = createRandomMatrix(matmulObject.sizeXY, matmulObject.sizeNY);
    matmulObject.matrixP = createIdentityMatrix(matmulObject.sizeMX, matmulObject.sizeNY);
}

function printMatrix(name, matrix, sizeX, sizeY) {
    console.log(`********Matrix ${name}********`);
    console.log(`[Rows, Columns] = [${sizeX}, ${sizeY}]`);
    console.log(matrix);
    let matrixString = '';
    for (let x = 0; x < sizeX; x++) {
        for (let y = 0; y < sizeY; y++) {
            matrixString += `${matrix[y * sizeX + x]}\t`;
        }
        matrixString += '\n';
    }
    console.log(matrixString);
    console.log(`****************************************`);
}

async function matrixMultiplicationNaive() {
    const matmulObject = {
        sizeMX: 0,
        sizeXY: 0,
        sizeNY: 0,
        ops: undefined,
        matrixM: undefined,
        matrixN: undefined,
        matrixP: undefined,
        cpu: {
            elapsedTime: undefined,
            gflops: undefined
        },
        gpu: {
            elapsedTime: undefined,
            gflops: undefined
        }
    };

    const device = await initGPUDevice();

    const maxSize = (1 << 12) + 1;
    for (let size = 16; size < maxSize; size *= Math.SQRT2) { // Test NPOT Size too
        matmulObject.sizeMX = matmulObject.sizeXY = matmulObject.sizeNY = Math.trunc(size);

        initMatrixMultiplication(matmulObject);
        console.log('===========================================================');
        console.log(`Matrix M Size = ${matmulObject.sizeMX} x ${matmulObject.sizeXY}`);
        console.log(`Matrix N Size = ${matmulObject.sizeXY} x ${matmulObject.sizeNY}`);
        console.log(`Matrix P Size = ${matmulObject.sizeMX} x ${matmulObject.sizeXY}`);
        cpuMatrixMultiplication(matmulObject);
        const gpuSuccess = await webGPUMatrixMultiplication(device, matmulObject);
        if (gpuSuccess) {
            console.log(matmulObject.cpu);
            console.log(matmulObject.gpu);
            const timeSpeedUp = matmulObject.cpu.elapsedTime / matmulObject.gpu.elapsedTime;
            console.log(`Speed Up: ${timeSpeedUp.toFixed(3)}x`);
        }
        console.log('===========================================================');
    }
    console.log('***Matrix Multiplication Complete***');
}

await matrixMultiplicationNaive();
