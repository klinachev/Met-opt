package methods;

import methods.one_dimentional.BrentsMethod;
import methods.one_dimentional.Dichotomy;
import tools.Vector;

import java.util.function.DoubleUnaryOperator;
import java.util.function.Function;

public class FastestDescent {
    private final double maxAlpha;
    private final Dichotomy method = new Dichotomy(1e-6);

    public FastestDescent(final double maxAlpha) {
        this.maxAlpha = maxAlpha;
    }

    public Vector run(
            final Function<Vector, Double> function,
            final Function<Vector, Vector> grad,
            Vector x,
            final double epsilon) {
        Vector gradientFX = grad.apply(x);
//        Double fx = function.applyFunction(x);
////        Vector lastX = x;
////        double lastFX = 0;
        int sch = 0;
        double alpha;
        while (epsilon < gradientFX.absSqr() && sch < 11_000) {
            sch++;
            final Vector finalGradientFX = gradientFX;
            final Vector finalX = x;
            final DoubleUnaryOperator fAlpha = alpha1 ->
                    function.apply(finalX.subtract(finalGradientFX.multiply(alpha1)));
            alpha = method.calc(fAlpha, 0, maxAlpha);
//            lastX = x;
//            lastFX = fx;
            x = finalX.subtract(finalGradientFX.multiply(alpha));
//            fx = function.applyFunction(x);
            gradientFX = grad.apply(x);
            if (alpha == 0) {
                System.out.println("alpha is 0");
                System.out.println("\niterations count: " + sch);
//                System.out.println(String.format("Result:\n%s \nval: %.10f", x, fx));
                return x;
            }
        }
        System.out.println("\niterations count: " + sch);
//        System.out.println(String.format("Result:\n%s \nval: %.10f", x, function.apply(x)));
        return x;
    }
}
