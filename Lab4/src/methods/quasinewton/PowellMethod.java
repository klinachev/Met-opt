package methods.quasinewton;

import methods.one_dimentional.BrentsMethod;
import tools.Vector;
import tools.TableImpl;

import java.util.List;
import java.util.function.Function;

public class PowellMethod {
    protected final BrentsMethod method;

    public PowellMethod(final BrentsMethod method) {
        this.method = method;
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
            Vector deltaXWave = deltaX.add(GdeltaW);
            G = G.subtract(BFShMethod.vectorsToTable(deltaXWave).divide(deltaW.scalar(deltaXWave)));
        } while (deltaX.abs() > epsilon);
        return x;
    }
}