import elements.MyFunction;
import elements.Vector;
import methods.ConjugateGradientMethod;
import methods.GradientDescent;
import methods.FastestDescent;

import java.util.List;

public class RunMethod {
    public static void main(final String[] args) {
        final GradientDescent gradientDescent = new GradientDescent(0.01);
        final FastestDescent fastestDescent = new FastestDescent(0.01);
        final ConjugateGradientMethod conjugateGradientMethod = new ConjugateGradientMethod();
        final MyFunction function = new MyFunction(
                list -> {
                    final double first = list.get(0), second = list.get(1);
                    return first * first + 40 * second * second + second;
                },
                List.of(
                        list -> 2 * list.get(0),
                        list -> 80 * list.get(1) + 1.
                )
        );
        final Vector x = new Vector(List.of(1., 1.));
        gradientDescent.calc(function, x);
        System.out.println("-----------------------");
        fastestDescent.calc(function, x);
        System.out.println("-----------------------");
        conjugateGradientMethod.calc(function, x);
    }
}
