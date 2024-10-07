export const MathHelpers = {
    TEN_POWER_THREE: Math.pow(10, 3),
    TEN_POWER_SIX: Math.pow(10, 6),
    TEN_POWER_NINE: Math.pow(10, 9),

    equalsEpsilon: (left, right, epsilon) => {
        epsilon = (epsilon !== undefined) ? epsilon : 0.0;
        return Math.abs(left - right) <= epsilon;
    },

    formatNumber: (number) => {
        return number.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    }
};

