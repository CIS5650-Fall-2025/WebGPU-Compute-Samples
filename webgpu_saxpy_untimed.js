const WORKGROUP_SIZE = 256;

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

    const device = await adapter.requestDevice();
    console.log(device);
    return device;
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
     * Compute passes are the actual invocation of compute operations. Each one starts off with a beginComputePass() call,
     * which defines the pipelines, buffers, layouts that are the inputs and outputs.
     * It's important to know that simply making these calls does not cause the GPU to actually do anything. They're just recording commands for the GPU to do later.
     */
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(saxpyCSPipeline);
    computePass.setBindGroup(0, saxpyCSBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(size / WORKGROUP_SIZE));
    computePass.end();

    /**
     * Copy the result Z vector to the GPU to check the computation.
     */
    // Get a GPU buffer for reading in an unmapped state.
    const zVectorStagingBuffer = device.createBuffer({
        size: zVector.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // Encode commands for copying buffer to buffer.
    encoder.copyBufferToBuffer(
        zVectorGPUBuffer /* source buffer */,
        0 /* source offset */,
        zVectorStagingBuffer /* destination buffer */,
        0 /* destination offset */,
        zVector.byteLength /* size */
    );

    device.queue.submit([encoder.finish()]);

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

    for (let i = 0; i < size; i++) {
       zVector[i] =  aScalar * xVector[i] + yVector[i];
    }
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
        zVector: undefined
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
        console.log('===========================================================');
    }
    console.log('***SAXPY Complete***');
}

await saxpy();
