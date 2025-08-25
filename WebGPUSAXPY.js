import { MathHelpers } from './MathHelpers.js'
import { VectorHelpers } from './VectorHelpers.js'
import { WebGPUHelpers } from './WebGPUHelpers.js'

const regularTable = window.regularTable;
const WORKGROUP_SIZE = WebGPUHelpers.WORKGROUP_SIZE_1D;

// Query index counter for timestamps
let queryIndex = 0;

async function webGPUSAXPY(device, saxpyObject, verbose) {
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
    const { querySet, timestampWrites, performanceResolveBuffer, performanceResultBuffer } = (() => {
        const querySet = device.createQuerySet({
            type: 'timestamp',
            count: 2,
        });
        const timestampWrites = {
            querySet: querySet,
            beginningOfPassWriteIndex: queryIndex,
            endOfPassWriteIndex: queryIndex + 1
        };
        const performanceResolveBuffer = device.createBuffer({
            size: querySet.count * 8,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        const performanceResultBuffer = device.createBuffer({
            size: performanceResolveBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        return { querySet, timestampWrites, performanceResolveBuffer, performanceResultBuffer };
    })();

    /**
     * Compute passes are the actual invocation of compute operations. Each one starts off with a beginComputePass() call,
     * which defines the pipelines, buffers, layouts that are the inputs and outputs.
     * It's important to know that simply making these calls does not cause the GPU to actually do anything. They're just recording commands for the GPU to do later.
     */
    const computePass = encoder.beginComputePass({ timestampWrites });
    computePass.setPipeline(saxpyCSPipeline);
    computePass.setBindGroup(0, saxpyCSBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(size / WORKGROUP_SIZE));
    computePass.end();

    // Incremement the Query Index counter
    queryIndex += 2;

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
            gpu.elapsedTime = Number(times[1] - times[0]) / MathHelpers.TEN_POWER_NINE; // seconds
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
        if (!MathHelpers.equalsEpsilon(zVectorGPUResult[i], zVector[i], 1e-6)) {
            console.log (`Mismatch Error: GPU = ${zVectorGPUResult[i]} and CPU = ${zVector[i]} at index ${i}.`);
            error = true;
            break;
        }
    }

    zVectorStagingBuffer.unmap();

    if (!error && verbose) {
        console.log('Results match');
    }

    return !error;
}

async function run(results, device, saxpyObject, verbose) {
    VectorHelpers.initSAXPY(saxpyObject);
    if (verbose) {
        console.log('===========================================================');
        console.log(`Size = ${saxpyObject.size}`)
    }
    results.size.push(saxpyObject.size);

    VectorHelpers.cpuSAXPY(saxpyObject);
    const gpuSuccess = await webGPUSAXPY(device, saxpyObject, verbose);
    if (gpuSuccess) {
        const timeSpeedUp = saxpyObject.cpu.elapsedTime / saxpyObject.gpu.elapsedTime;
        if (verbose) {
            console.log(`Speed Up: ${timeSpeedUp.toFixed(3)}x`);
        }
    }

    if (verbose) {
        console.log('===========================================================');
    }

    results.status.push(gpuSuccess);
    results.cpuTime.push(saxpyObject.cpu.elapsedTime);
    results.gpuTime.push(saxpyObject.gpu.elapsedTime);
}

async function saxpy() {
    const verbose = false;
    const progressElement = document.getElementById('progress');

    const { adapterInfo, device } = await WebGPUHelpers.initGPUDevice(true);

    document.getElementById('gpu-info').innerText = `GPU: ${adapterInfo.description}`;

    const results = {
        size: [],
        status: [],
        cpuTime: [],
        gpuTime: []
    };

    const saxpyObject = VectorHelpers.getEmptySaxpyObject();

    const maxSize = (1 << 24) + 1;
    for (let size = 1 << 16; size < maxSize; size *= Math.SQRT2) { // Test NPOT Sizes too
        saxpyObject.size = Math.trunc(size);
        progressElement.innerText = `Running WebGPU SAXPY for size ${MathHelpers.formatNumber(saxpyObject.size)}`;
        await run(results, device, saxpyObject, verbose);
    }

    if (verbose) {
        console.log('***SAXPY Complete***');
    }

    progressElement.remove();
    resultsToTable(results);
};

function resultsToTable(results) {
    const speedUps = results.cpuTime.map((value, index) => value / results.gpuTime[index]);

    const tableData = [
        results.size.map(MathHelpers.formatNumber),
        results.status,
        results.cpuTime.map((value) => value.toFixed(6)),
        results.gpuTime.map((value) => value.toFixed(6)),
        speedUps.map((value, index) => (results.status[index] === true) ? (value.toFixed(3) + 'x') : '')
    ];

    const columnHeaders = [['Vector Size'], ['GPU Success'], ['CPU Time (sec)'], ['GPU Time (sec)'], ['Speed Up WebGPU vs CPU)']];

    function dataListener(x0, y0, x1, y1) {
        return {
            num_rows: results.size.length,
            num_columns: tableData.length,
            data: tableData.slice(x0, x1).map((col) => col.slice(y0, y1)),
            column_headers: columnHeaders
        };
    }

    regularTable.setDataListener(dataListener);

    regularTable.addStyleListener(() => {
        for (const td of regularTable.querySelectorAll("td")) {
            const meta = regularTable.getMeta(td);
            if (meta.column_header[0] === columnHeaders[1][0]) {
                td.style.color = meta.value ? 'green' : 'red';
            }
        }
    });

    regularTable.draw();
}

await saxpy();
