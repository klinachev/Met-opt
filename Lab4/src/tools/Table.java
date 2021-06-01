package tools;

import java.util.List;

public interface Table extends List<List<Double>> {
    default void subtractLine(final int from, final int to, final double mull, final int start) {
        for (int i = start; i < size(); ++i) {
            final Double value = get(to, i);
            set(to, i, value - mull * get(from, i));
        }
    }

    default Double get(final int i, final int j) {
        return get(i).get(j);
    }

    default void set(final int i, final int j, final double val) {
        get(i).set(j, val);
    }

    default void swap(final int a, final int b) {
        final List<Double> list = get(a);
        set(a, get(b));
        set(b, list);
    }
}
