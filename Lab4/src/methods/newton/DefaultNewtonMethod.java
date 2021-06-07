package methods.newton;

import tools.Table;
import tools.Vector;

import java.util.Formatter;
import java.util.Locale;
import java.util.function.BiFunction;
import java.util.function.Function;

public class DefaultNewtonMethod {
    protected final BiFunction<Table, Vector, Vector> soleMethod;

    public DefaultNewtonMethod(final BiFunction<Table, Vector, Vector> soleMethod) {
        this.soleMethod = soleMethod;
    }

    public static void printPoint(final Vector v) {
        System.out.format(Locale.US, " (%.4f, %.4f))\nVector((%.4f, %.4f), ", v.get(0), v.get(1), v.get(0), v.get(1));
    }

    protected Vector evaluateP(final Vector gradX, final Table hessian) {
//        System.out.println(hessian);
//        System.out.println(gradX.negate());
        Vector result = soleMethod.apply(hessian, gradX.negate());
//        System.out.println(result);
        return result;
    }

    public Vector run(
            final Function<Vector, Vector> grad,
            final Function<Vector, Table> hessian,
            Vector x,
            final double epsilon
    ) {
        int iterations = -1;
        Vector deltaX = null;
        printPoint(x);
        while (deltaX == null || deltaX.abs() > epsilon) {
            deltaX = evaluateP(grad.apply(x), hessian.apply(x));
            x = x.add(deltaX);
            iterations++;
            printPoint(x);
//            System.out.println(x);
        }
        System.out.println("\nIterations count: " + iterations);
        return x;
    }


}
