package elements;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class Vector {
    private final List<Double> point;

    public Vector(final List<Double> point) {
        this.point = point;
    }

    public List<Double> getPoint() {
        return point;
    }

    public Vector multiply(final double mul) {
        return new Vector(
                point.stream()
                        .map(value -> value * mul)
                        .collect(Collectors.toList())
        );
    }

    public Vector negate() {
        final List<Double> value = new ArrayList<>();
        for (int i = 0; i < point.size(); ++i) {
            value.add(-point.get(i));
        }
        return new Vector(value);
    }

    public Vector subtract(final Vector vector) {
        final List<Double> value = new ArrayList<>();
        for (int i = 0; i < point.size(); ++i) {
            value.add(point.get(i) - vector.point.get(i));
        }
        return new Vector(value);
    }

    public Vector add(final Vector vector) {
        final List<Double> value = new ArrayList<>();
        for (int i = 0; i < point.size(); ++i) {
            value.add(point.get(i) + vector.point.get(i));
        }
        return new Vector(value);
    }


    public double abs() {
        return Math.sqrt(point.stream().reduce(0., (arg1, arg2) -> arg1 + arg2 * arg2));
    }

    public double absSqr() {
        return point.stream().reduce(0., (arg1, arg2) -> arg1 + arg2 * arg2);
    }

    @Override
    public String toString() {
        return "Vector{" +
                "point=" + point +
                '}';
    }
}
