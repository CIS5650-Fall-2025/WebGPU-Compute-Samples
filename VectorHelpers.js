import { MathHelpers } from './MathHelpers.js'

export const VectorHelpers = {
    printVector: (name, vector, size) => {
        console.log(`********Vector ${name} (size: ${size})********`);
        console.log(vector);
        console.log(`****************************************`);
    },

    cpuSAXPY: (saxpyObject) => {
        const size = saxpyObject.size;
        const aScalar = saxpyObject.aScalar;
        const xVector = saxpyObject.xVector;
        const yVector = saxpyObject.yVector;
        const zVector = saxpyObject.zVector;

        const startTime = performance.now();
        for (let i = 0; i < size; i++) {
            zVector[i] = aScalar * xVector[i] + yVector[i];
        }
        const endTime = performance.now();

        const cpu = saxpyObject.cpu;
        cpu.elapsedTime = (endTime - startTime) / MathHelpers.TEN_POWER_THREE; // seconds
    },

    // identity vector
    createIdentityVector: (size) => {
        const v = new Float32Array(size);
        v.fill(0);
        return v;
    },

    // random vector
    createRandomVector: (size) => {
        const v = new Float32Array(size);
        for (var i = 0; i < size; ++i) {
            v[i] = Math.random();
        }
        return v;
    },

    initSAXPY: (saxpyObject) => {
        const size = saxpyObject.size;

        // Create X and Y Vector on CPU
        saxpyObject.aScalar = Math.random();
        saxpyObject.xVector = VectorHelpers.createRandomVector(size);
        saxpyObject.yVector = VectorHelpers.createRandomVector(size);
        saxpyObject.zVector = VectorHelpers.createIdentityVector(size);
    },

    getEmptySaxpyObject: () => {
        return {
            size: 0,
            aScalar: undefined,
            xVector: undefined,
            yVector: undefined,
            zVector: undefined,
            cpu: {
                elapsedTime: undefined,
            },
            gpu: {
                elapsedTime: undefined,
            }
        };
    }
};

