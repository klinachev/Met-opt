package methods.quasinewton;

import methods.one_dimentional.Dichotomy;
import tools.TableImpl;
import tools.Vector;

import java.util.function.Function;

public class BFShMethod {
    protected final Dichotomy method = new Dichotomy(1e-8);

    public static TableImpl vectorsToTable(final Vector v) {
        final int n = v.size();
        final TableImpl T = new TableImpl(n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                T.set(i, j, v.get(i) * v.get(j));
            }
        }
        return T;
    }

    private double calcAlpha(final Function<Vector, Double> function,
                             final Vector x,
                             final Vector way) {
        return method.calc(a -> function.apply(x.add(way.multiply(a))), 0, 1000);
    }

    public Vector run(
            final Function<Vector, Double> function,
            final Function<Vector, Vector> grad,
            Vector x,
            final double epsilon
    ) {
        TableImpl G = new TableImpl(x.size(), 1.);
        Vector w = grad.apply(x).negate();
        final double firstAlpha = calcAlpha(function, x, w);
        Vector deltaX = w.multiply(firstAlpha);
        x = x.add(deltaX);
        while (deltaX.abs() > epsilon) {
            final Vector newW = grad.apply(x).negate();
            final Vector deltaW = newW.subtract(w);
            w = newW;

            final Vector uk = G.multiply(deltaW);
            final double rho = uk.scalar(deltaW);
            final Vector r = uk.multiply(1. / rho).subtract(deltaX.multiply(1. / deltaX.scalar(deltaW)));

            G = G.subtract(vectorsToTable(deltaX).divide(deltaW.scalar(deltaX)))
                    .subtract(vectorsToTable(uk).divide(rho))
                    .add(vectorsToTable(r).multiply(rho));

            final Vector p = G.multiply(w);
            final double alpha = calcAlpha(function, x, p);
            deltaX = p.multiply(alpha);
            x = x.add(deltaX);
//            System.out.println(firstAlpha + " " + deltaX);
//            System.out.println(x);
        }
        return x;
    }
}