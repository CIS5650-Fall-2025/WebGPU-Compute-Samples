import { MathHelpers } from './MathHelpers.js'

export const MatrixHelpers = {
    mode : {
        BASIC: 0, // Square matrices
        ADVANCE: 1, // Rectangular transposed matrices
        FULL: 2 // Full variation
    },

    kernel : {
        naive: 'naive',
        tiled: 'tiled'
    },

    printMatrix: (name, matrix, sizeX, sizeY) => {
        console.log(`********Matrix ${name}********`);
        console.log(`[Rows, Columns] = [${sizeX}, ${sizeY}]`);
        let matrixString = '';
        for (let x = 0; x < sizeX; x++) {
            for (let y = 0; y < sizeY; y++) {
                matrixString += `${matrix[y * sizeX + x]}\t`;
            }
            matrixString += '\n';
        }
        console.log(matrixString);
        console.log(`****************************************`);
    },

    cpuMatrixMultiplication: (matmulObject) => {
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
        cpu.elapsedTime = (endTime - startTime) / MathHelpers.TEN_POWER_THREE; // seconds
    },

    // identity matrix
    createIdentityMatrix: (sizeX, sizeY) => {
        const m = new Float32Array(sizeX * sizeY);
        m.fill(0);
        return m;
    },

    // random matrix
    createRandomMatrix: (sizeX, sizeY) => {
        const nElements = sizeX * sizeY;
        const m = new Float32Array(nElements);
        for (var i = 0; i < nElements; i++) {
            m[i] = Math.trunc(Math.random() * 10);
        }
        return m;
    },

    initMatrixMultiplication: (matmulObject) => {
        // Create X and Y Matrix on CPU
        matmulObject.matrixM = MatrixHelpers.createRandomMatrix(matmulObject.sizeMX, matmulObject.sizeXY);
        matmulObject.matrixN = MatrixHelpers.createRandomMatrix(matmulObject.sizeXY, matmulObject.sizeNY);
        matmulObject.matrixP = MatrixHelpers.createIdentityMatrix(matmulObject.sizeMX, matmulObject.sizeNY);
    },

    getEmptyMatmulObject: () => {
        return {
            sizeMX: 0,
            sizeXY: 0,
            sizeNY: 0,
            matrixM: undefined,
            matrixN: undefined,
            matrixP: undefined,
            cpu: {
                elapsedTime: undefined,
            },
            gpu: {
                naive: {
                    elapsedTime: undefined,
                    status: false,
                    error: undefined
                },
                tiled: {
                    elapsedTime: undefined,
                    status: false,
                    error: undefined
                }
            }
        };
    },
};
