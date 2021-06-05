package methods.gauss;

import tools.Table;
import tools.TableImpl;
import tools.Vector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class GaussMethod {

    private static <T> void swap(final List<T> list, final int a, final int b) {
        final T elem = list.get(a);
        list.set(a, list.get(b));
        list.set(b, elem);
    }

    private static void subtractLines(final Table a, final List<Double> b, final int i) {
        for (int j = i + 1; j < a.size(); ++j) {
            final double t = a.get(j, i) / a.get(i, i);
            b.set(j, b.get(j) - t * b.get(i));
            a.subtractLine(i, j, a.get(j, i) / a.get(i, i), i);
        }
    }

    private static void forwardEliminationWithMax(final Table a, final List<Double> b) {
        for (int i = 0; i < a.size(); ++i) {
            if (a.get(i, i) == 0) {
                int max = i;
                for (int j = i + 1; j < a.size(); ++j) {
                    if (Math.abs(a.get(max, i)) < Math.abs(a.get(j, i))) {
                        max = j;
                    }
                }
                if (a.get(max, i) == 0.) {
                    throw new RuntimeException("no solution");
                }
                swap(a, i, max);
                swap(b, i, max);
            }
            subtractLines(a, b, i);
        }
    }

    private static void forwardElimination(final Table a, final List<Double> b) {
        for (int i = 0; i < a.size(); ++i) {
            if (a.get(i, i) == 0) {
                int j = i + 1;
                while (j < a.size() && (a.get(j, i) == 0)) {
                    j++;
                }
                if (j == a.size()) {
                    throw new RuntimeException("no solution");
                }
                swap(a, i, j);
                swap(b, i, j);
            }
            subtractLines(a, b, i);
        }
    }

    static void backSubstitution(final Table a, final List<Double> b) {
        b.set(a.size() - 1, b.get(b.size() - 1) / a.get(a.size() - 1, a.size() - 1));
        for (int k = a.size() - 2; k >= 0; --k) {
            double value = 0;
            for (int j = k + 1; j < a.size(); ++j) {
                value += a.get(k, j) * b.get(j);
            }
            b.set(k, (b.get(k) - value) / a.get(k, k));
        }
    }

    public static List<Double> run(final Table a, final List<Double> b) {
        try {

//        System.out.println("a1 " + a);
//        System.out.println("b1 " + b);
            forwardElimination(a, b);
//        System.out.println("a2 " + a);
//        System.out.println("b2 " + b);
            backSubstitution(a, b);
//        System.out.println("result " + list);
            return b;
        } catch (final Throwable throwable) {
            System.out.println(a);
            System.out.println(b);
            throw throwable;
        }
    }


    public static Vector calc(final Table a, final Vector b) {
        return new Vector(run(new TableImpl(a), new ArrayList<>(b.getPoint())));
    }

}
