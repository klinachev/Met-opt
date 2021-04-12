package methods;

import methods.result.IterationResult;
import methods.result.Result;

import java.util.function.DoubleUnaryOperator;

public interface Method<R extends IterationResult> {

    Result<R> calcAllIterations(DoubleUnaryOperator f, double a, double b);

    default boolean compare(double x, double y) {
        return x - y >= 0;
    }
}
