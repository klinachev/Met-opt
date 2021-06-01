package methods.newton;

import tools.Table;
import tools.Vector;

import java.util.function.BiFunction;
import java.util.function.Function;

public class DefaultNewtonMethod {
    protected final BiFunction<Table, Vector, Vector> soleMethod;

    public DefaultNewtonMethod(final BiFunction<Table, Vector, Vector> soleMethod) {
        this.soleMethod = soleMethod;
    }

    protected Vector evaluateP(final Vector gradX, final Table hessian) {
        return soleMethod.apply(hessian, gradX.negate());
    }

    public Vector run(
            final Function<Vector, Vector> grad,
            final Function<Vector, Table> hessian,
            Vector x,
            final double epsilon
    ) {
        Vector deltaX = null;
        while (deltaX == null || deltaX.abs() > epsilon) {
            deltaX = evaluateP(grad.apply(x), hessian.apply(x));
            x = x.add(deltaX);
//            System.out.println(x + " ");
        }
        return x;
    }


}
