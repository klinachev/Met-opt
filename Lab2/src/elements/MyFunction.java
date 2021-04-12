package elements;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class MyFunction {
    Function<List<Double>, Double> function;
    List<Function<List<Double>, Double>> gradient;

    public MyFunction(
            final Function<List<Double>, Double> function,
            final List<Function<List<Double>, Double>> gradient
    ) {
        this.function = function;
        this.gradient = gradient;
    }

    public Double applyFunction(final Vector value) {
        return function.apply(value.getPoint());
    }

    public Vector applyGradient(final Vector vector) {
        return new Vector(
                gradient.stream()
                        .map(function -> function.apply(vector.getPoint()))
                        .collect(Collectors.toList())
        );
    }
}
