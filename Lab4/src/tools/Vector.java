package tools;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class Vector {
    private final List<Double> point;

    public Vector(final List<Double> point) {
        this.point = point;
    }


    public static Vector of(final Double... data) {
        return new Vector(List.of(data));
    }

    public List<Double> getPoint() {
        return point;
    }

    public int size() {
        return point.size();
    }

    public Double get(final int index) {
        return point.get(index);
    }

    public void set(final int i, final double v) {
        point.set(i, v);
    }

    public Vector multiply(final double mul) {
        return new Vector(point.stream()
                .map(value -> value * mul)
                .collect(Collectors.toList())
        );
    }

    public Vector negate() {
        final List<Double> value = new ArrayList<>();
        for (final Double aDouble : point) {
            value.add(-aDouble);
        }
        return new Vector(value);
    }

    public Vector subtract(final Vector vector) {
        final List<Double> value = new ArrayList<>();
        for (int i = 0; i < size(); ++i) {
            value.add(point.get(i) - vector.point.get(i));
        }
        return new Vector(value);
    }

    public Vector add(final Vector vector) {
        final List<Double> value = new ArrayList<>();
        for (int i = 0; i < size(); ++i) {
            value.add(point.get(i) + vector.get(i));
        }
        return new Vector(value);
    }

    public double scalar(final Vector vector) {
        double val = 0;
        for (int i = 0; i < size(); ++i) {
            val += get(i) * vector.get(i);
        }
        return val;
    }

    public double abs() {
        return Math.sqrt(absSqr());
    }

    public double absSqr() {
        return point.stream().reduce(0., (arg1, arg2) -> arg1 + arg2 * arg2);
    }

    @Override
    public String toString() {
        return "Point{" +
                point.stream().map(d -> String.format("%.17f", d)).collect(Collectors.joining("; ")) +
                "}";
    }
}
