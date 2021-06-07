package methods.newton;

import methods.one_dimentional.BrentsMethod;
import methods.one_dimentional.Dichotomy;
import tools.Table;
import tools.Vector;

import java.util.function.BiFunction;
import java.util.function.Function;

public class OneDimensionalSearch extends DefaultNewtonMethod {
    protected final Dichotomy method = new Dichotomy(1e-6);

    public OneDimensionalSearch(final BiFunction<Table, Vector, Vector> soleMethod) {
        super(soleMethod);
    }

    protected double evaluateAlpha(
            final Function<Vector, Double> function, final Vector x, final Vector p) {
        return method.calc(alpha1 -> function.apply(x.add(p.multiply(alpha1))), -10, 10);
    }

    public Vector run(
            final Function<Vector, Double> function,
            final Function<Vector, Vector> grad,
            final Function<Vector, Table> hessian,
            Vector x,
            final double epsilon
    ) {
        int iterations = -1;
        Vector deltaX = null;
        printPoint(x);
        while (deltaX == null || deltaX.abs() > epsilon) {
            final Vector p = evaluateP(grad.apply(x), hessian.apply(x));
//            System.out.println("P: " + p);
            final double alpha = evaluateAlpha(function, x, p);
            deltaX = p.multiply(alpha);
            x = x.add(deltaX);
            iterations++;
            printPoint(x);
//            System.out.println(alpha + " X: " + x + " delta x: " + deltaX + " abs: " + deltaX.abs());
        }
        System.out.println("\nIterations count: " + iterations);
        return x;
    }
}
