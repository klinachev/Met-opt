package methods;

import methods.result.DefaultIterationResult;
import methods.result.Result;

import java.util.ArrayList;
import java.util.List;
import java.util.function.DoubleUnaryOperator;

public class Dichotomy implements Method<DefaultIterationResult> {
    private final double epsilon;
    private final double delta;

    public Dichotomy(double epsilon) {
        this.epsilon = epsilon;
        this.delta = epsilon / 2;
    }

    @Override
    public Result<DefaultIterationResult> calcAllIterations(DoubleUnaryOperator f, double a, double b) {
        double x1, x2, e = (b - a) / 2;
        List<DefaultIterationResult> results = new ArrayList<>();
        while (compare(e, epsilon)) {
            x1 = (a + b - delta) / 2;
            x2 = (a + b + delta) / 2;
            double f1 = f.applyAsDouble(x1);
            double f2 = f.applyAsDouble(x2);
            results.add(new DefaultIterationResult(a, b, x1, x2, f1, f2));
            if (compare(f2, f1)) {
                b = x2;
            } else {
                a = x1;
            }
            e = (b - a) / 2;
        }
        return new Result<>((a + b) / 2, f.applyAsDouble((a + b) / 2), results);
    }

}
