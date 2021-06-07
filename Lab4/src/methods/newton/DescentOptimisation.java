package methods.newton;

import tools.Table;
import tools.Vector;

import java.util.function.BiFunction;
import java.util.function.Function;

public class DescentOptimisation extends OneDimensionalSearch {
    public DescentOptimisation(final BiFunction<Table, Vector, Vector> soleMethod) {
        super(soleMethod);
    }

    @Override
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
            final Vector gradX = grad.apply(x);
            Vector p = evaluateP(gradX, hessian.apply(x));
            if (p.scalar(gradX) > 0) {
//                System.out.println(" p * gradX > 0 " + x);
                p = gradX.negate();
            }
            iterations++;
            final double alpha = evaluateAlpha(function, x, p);
            deltaX = p.multiply(alpha);
            x = x.add(deltaX);
            printPoint(x);
        }
        System.out.println("\nIterations count: " + iterations);
        return x;
    }
}
