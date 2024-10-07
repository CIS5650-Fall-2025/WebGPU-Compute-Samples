export const WebGPUHelpers = {

    WORKGROUP_SIZE_1D: 256,
    WORKGROUP_SIZE_2D: 16,

    initGPUDevice: async (enablePerformanceTimers) => {
        if (!navigator.gpu) {
            throw new Error("WebGPU not supported on this browser.");
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error("No appropriate GPUAdapter found.");
        }

        let device;
        if (enablePerformanceTimers) {
            // Device initialization with performance timers
            const canTimestamp = adapter.features.has('timestamp-query');
            if (!canTimestamp) {
                throw new Error("Timestamps not available. Enable the right flags in your browser.");
            }
            device = await adapter.requestDevice({
                requiredFeatures: ["timestamp-query"]
            });
        } else {
            device = await adapter.requestDevice();
        }

        if (!device) {
            throw new Error("Need a browser that supports WebGPU.");
        }

        return device;
    }
}
