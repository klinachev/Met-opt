package methods;

import elements.Vector;
import elements.MyFunction;

public class GradientDescent {
    private final double epsilon;

    public GradientDescent(final double epsilon) {
        this.epsilon = epsilon;
    }

    public void calc(final MyFunction function, Vector x) {
        Vector y, gradientFX = function.applyGradient(x);
        Double fy, fx = null;
        double alpha = 1;
        System.out.println("gradient abs: " + gradientFX.abs());
        while (epsilon < gradientFX.abs()) {
            fx = function.applyFunction(x);
            do {
                y = x.subtract(gradientFX.multiply(alpha));
                fy = function.applyFunction(y);
                if (fy >= fx) {
                    alpha /= 2;
                }
            } while (fy >= fx);
            x = y;
            fx = fy;
            gradientFX = function.applyGradient(x);
            System.out.println(String.format("%s val: %.10f", x, fx));
            System.out.println("gradient abs: " + gradientFX.abs());
        }
        System.out.println(String.format("Result:\n%s val: %.10f", x, fx));
    }
}
