package methods;

import elements.MyFunction;
import elements.Vector;
import methods.Dichotomy;
import methods.result.IterationResult;
import methods.result.Result;

import java.util.function.DoubleUnaryOperator;
import java.util.function.UnaryOperator;


public class FastestDescent {
    private final double epsilon;

    public FastestDescent(final double epsilon) {
        this.epsilon = epsilon;
    }

    public void calc(final MyFunction function, Vector x) {
        Vector gradientFX = function.applyGradient(x);
        Double fx = null;
        System.out.println("gradient abs: " + gradientFX.abs());
        while (epsilon < gradientFX.abs()) {
            Vector finalGradientFX = gradientFX;
            Vector finalX = x;
            DoubleUnaryOperator fAlpha = alpha -> function.applyFunction(finalX.subtract(finalGradientFX.multiply(alpha)));
            Method<? extends IterationResult> method = new Dichotomy(epsilon);
            Result<? extends IterationResult> result = method.calcAllIterations(fAlpha, 0, 100);
            double alpha = result.getPoint();
            x = x.subtract(gradientFX.multiply(alpha));
            fx = function.applyFunction(x);
            gradientFX = function.applyGradient(x);
            System.out.println(String.format("%s val: %.10f", x, fx));
            System.out.println("gradient abs: " + gradientFX.abs());
        }
        System.out.println(String.format("Result:\n%s val: %.10f", x, fx));
    }
}
