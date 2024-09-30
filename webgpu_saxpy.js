const WORKGROUP_SIZE = 256;
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

async function webGPUSAXPY(device, saxpyObject) {
    const size = saxpyObject.size;
    const aScalar = saxpyObject.aScalar;
    const xVector = saxpyObject.xVector;
    const yVector = saxpyObject.yVector;
    const zVector = saxpyObject.zVector;

    /**
     * Set up X Vector on CPU and GPU
     */
    // Create buffer on GPU for X
    const xVectorGPUBuffer = device.createBuffer({
        label: "X Vector GPU Buffer",
        mappedAtCreation: true,
        size: xVector.byteLength,
        usage: GPUBufferUsage.STORAGE
    });
    // Copy xVector to GPU
    const xVectorArrayBuffer = xVectorGPUBuffer.getMappedRange();
    new Float32Array(xVectorArrayBuffer).set(xVector);
    xVectorGPUBuffer.unmap();

    /**
     * Set up Y Vector on CPU and GPU
     */
    // Create buffer on GPU for Y
    // Alternate method of copying data to GPU
    const yVectorGPUBuffer = device.createBuffer({
        label: "Y Vector GPU Buffer",
        size: yVector.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(yVectorGPUBuffer, 0, yVector);

    /**
     * Set up result Z Vector on GPU (no copy from host)
     */
    // Create buffer on GPU for Z
    const zVectorGPUBuffer = device.createBuffer({
        label: "Z Vector GPU Buffer",
        size: zVector.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    /**
     * Create uniform for A scalar and size
     */
    const uniformBufferSize = 2 * Float32Array.BYTES_PER_ELEMENT; // lets store size as float too
    const uniformBuffer = device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    });
    const uniformValues = new Float32Array([size, aScalar]);
    device.queue.writeBuffer(uniformBuffer, 0, uniformValues);

    /**
     * Create the compute shader for SAXPY
     */
    const saxpyComputeShaderModule = device.createShaderModule({
        label: "SAXPY Compute Shader",
        code: `
                struct scalarsStruct {
                    size: f32,
                    aScalar: f32
                };
                @group(0) @binding(0) var<uniform> scalars: scalarsStruct;

                @group(0) @binding(1) var<storage> xVector: array<f32>;
                @group(0) @binding(2) var<storage> yVector: array<f32>;
                @group(0) @binding(3) var<storage, read_write> zVector: array<f32>;

                @compute @workgroup_size(${WORKGROUP_SIZE})
                fn computeMain(@builtin(global_invocation_id) index: vec3u) {
                    let i = index.x;
                    if (i < u32(scalars.size)) {
                        zVector[i] = (scalars.aScalar * xVector[i]) + yVector[i];
                    }
                }
            `
    });

    /**
     * Create a Bind Group Layout.
     * The Bing Group Layout connects the different buffers into the locations in shaders.
     * It also sets up the type and read-write permissions
    */
    const saxpyCSBindGroupLayout = device.createBindGroupLayout({
        label: "SAXPY Compute Shader Bind Group Layout",
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.COMPUTE, // A Scaler Input - Read-only
            buffer: {}
        }, {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE, // X Vector Input - Read-only
            buffer: {
                type: "read-only-storage"
            }
        }, {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE, // Y Vector Input - Read only
            buffer: {
                type: "read-only-storage"
            }
        }, {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE, // Z Vector Result - Write
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
    const saxpyCSBindGroup = device.createBindGroup({
            label: "SAXPY Compute Shader Bind Group",
            layout: saxpyCSBindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: uniformBuffer }
            }, {
                binding: 1,
                resource: { buffer: xVectorGPUBuffer }
            }, {
                binding: 2,
                resource: { buffer: yVectorGPUBuffer }
            }, {
                binding: 3,
                resource: { buffer: zVectorGPUBuffer }
            }],
        });

    /**
     * Create a pipeline layout.
     * A pipeline layout is a list of bind group layouts that one or more pipelines use. The order of the bind group layouts in the array needs to correspond with the @group attributes in the shaders. (This means that bindGroupLayout is associated with @group(0).)
    */
    const saxpyCSPipelineLayout = device.createPipelineLayout({
        label: "SAXPY Compute Shader Pipeline Layout",
        bindGroupLayouts: [saxpyCSBindGroupLayout],
    });

    /**
     * Create a draw and simulation pipelines.
     * The pipeline connects the shaders and the layouts, which in turn connects the buffers and bind groups.
    */
    const saxpyCSPipeline = device.createComputePipeline({
        label: "SAXPY Compute Shader Pipeline",
        layout: saxpyCSPipelineLayout,
        compute: {
            module: saxpyComputeShaderModule,
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
    computePass.setPipeline(saxpyCSPipeline);
    computePass.setBindGroup(0, saxpyCSBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(size / WORKGROUP_SIZE));
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

    const gpu = saxpyObject.gpu;
    if (performanceResultBuffer.mapState === 'unmapped') {
        performanceResultBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const times = new BigInt64Array(performanceResultBuffer.getMappedRange());
            gpu.elapsedTime = Number(times[1] - times[0]) / TEN_POWER_NINE; // seconds
            gpu.gflops = (saxpyObject.ops / TEN_POWER_NINE) / gpu.elapsedTime;
            performanceResultBuffer.unmap();
        });
    }

    /**
     * Copy the result Z vector to the GPU to check the computation.
     */
    // Get a GPU buffer for reading in an unmapped state.
    const zVectorStagingBuffer = device.createBuffer({
        size: zVector.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // Create a new encoder as the previous one is no longer usable as a result of calling submit and finish.
    const resultCopyEncoder = device.createCommandEncoder();

    // Encode commands for copying buffer to buffer.
    resultCopyEncoder.copyBufferToBuffer(
        zVectorGPUBuffer /* source buffer */,
        0 /* source offset */,
        zVectorStagingBuffer /* destination buffer */,
        0 /* destination offset */,
        zVector.byteLength /* size */
    );
    device.queue.submit([resultCopyEncoder.finish()]);

    // Read buffer
    await zVectorStagingBuffer.mapAsync(GPUMapMode.READ);
    const zVectorGPUResult = new Float32Array(zVectorStagingBuffer.getMappedRange());

    /**
     * Compare GPU and CPU results
     */
    let error = false;
    for (let i = 0; i < size; i++) {
        if (!equalsEpsilon(zVectorGPUResult[i], zVector[i], 1e-6)) {
            console.log (`Mismatch Error: GPU = ${zVectorGPUResult[i]} and CPU = ${zVector[i]} at index ${i}.`);
            error = true;
            break;
        }
    }

    zVectorStagingBuffer.unmap();

    if (!error) {
        console.log('Results match');
    }

    return !error;
}

function cpuSAXPY(saxpyObject) {
    const size = saxpyObject.size;
    const aScalar = saxpyObject.aScalar;
    const xVector = saxpyObject.xVector;
    const yVector = saxpyObject.yVector;
    const zVector = saxpyObject.zVector;

    const startTime = performance.now();
    for (let i = 0; i < size; i++) {
       zVector[i] =  aScalar * xVector[i] + yVector[i];
    }
    const endTime = performance.now();

    const cpu = saxpyObject.cpu;
    cpu.elapsedTime = (endTime - startTime) / TEN_POWER_THREE; // seconds
    cpu.gflops = (saxpyObject.ops / TEN_POWER_NINE) / cpu.elapsedTime;
}

// identity vector
function createIdentityVector(size) {
    const v = new Float32Array(size);
    v.fill(0);
    return v;
}

// random vector
function createRandomVector(size) {
    const v = new Float32Array(size);
    for (var i = 0; i < size; ++i) {
        v[i] = Math.random();
    }
    return v;
}

function initSAXPY(saxpyObject) {
    const size = saxpyObject.size;

    // Create X and Y Vector on CPU
    saxpyObject.ops = size * 2;
    saxpyObject.aScalar = Math.random();
    saxpyObject.xVector = createRandomVector(size);
    saxpyObject.yVector = createRandomVector(size);
    saxpyObject.zVector = createIdentityVector(size);
}

async function saxpy() {
    const saxpyObject = {
        size: 0,
        ops: undefined,
        aScalar: undefined,
        xVector: undefined,
        yVector: undefined,
        zVector: undefined,
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

    const maxSize = (1 << 24) + 1;
    for (let size = 16; size < maxSize; size *= Math.SQRT2) { // Test NPOT Sizes too
        saxpyObject.size = Math.trunc(size);

        initSAXPY(saxpyObject);
        console.log('===========================================================');
        console.log(`Size = ${saxpyObject.size}`)
        cpuSAXPY(saxpyObject);
        const gpuSuccess = await webGPUSAXPY(device, saxpyObject);
        if (gpuSuccess) {
            console.log(saxpyObject.cpu);
            console.log(saxpyObject.gpu);
            const timeSpeedUp = saxpyObject.cpu.elapsedTime / saxpyObject.gpu.elapsedTime;
            console.log(`Speed Up: ${timeSpeedUp.toFixed(3)}x`);
        }
        console.log('===========================================================');
    }
    console.log('***SAXPY Complete***');
}

await saxpy();
