package methods.quasinewton;

import methods.one_dimentional.BrentsMethod;
import tools.TableImpl;
import tools.Vector;

import java.util.function.Function;

public class BFShMethod {
    protected final BrentsMethod method;

    public BFShMethod(final BrentsMethod method) {
        this.method = method;
    }

    public static TableImpl vectorsToTable(Vector v) {
        int n = v.size();
        TableImpl T = new TableImpl(n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                T.set(i, j, v.get(i) * v.get(j));
            }
        }
        return T;
    }

    public Vector run(
            final Function<Vector, Double> function,
            final Function<Vector, Vector> grad,
            Vector x,
            final double epsilon
    ) {
        TableImpl G = new TableImpl(x.size(), 1.);
        Vector w = grad.apply(x).negate();
        Vector deltaX = null;
        do {
            Vector p = G.multiply(w);
            final Vector finalX = x;
            Vector finalP = p;
            double alpha = method.calc(a -> function.apply(finalX.add(finalP.multiply(a))), 0, 1e5);
            deltaX = p.multiply(alpha);
            x = x.add(deltaX);
            Vector newW = grad.apply(x).negate();
            Vector deltaW = newW.subtract(w);
            w = newW;
            Vector GdeltaW = G.multiply(deltaW);
            double rho = GdeltaW.scalar(deltaW);
            Vector r = GdeltaW.multiply(1. / rho).subtract(deltaX.multiply(1. / deltaX.scalar(deltaW)));
            G = G.subtract(vectorsToTable(deltaX).divide(deltaW.scalar(deltaX)))
                    .subtract(vectorsToTable(GdeltaW).divide(rho))
                    .add(vectorsToTable(r).multiply(rho));
        } while (deltaX.abs() > epsilon);
        return x;
    }
}