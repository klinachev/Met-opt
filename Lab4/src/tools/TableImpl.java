package tools;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class TableImpl extends ArrayList<List<Double>> implements Table {
    public TableImpl(final int n) {
        super(Stream.generate(() -> new ArrayList<>(Collections.nCopies(n, 0.)))
                .limit(n).collect(Collectors.toList()));
    }

    public TableImpl(final int n, final double x) {
        super(Stream.generate(() -> new ArrayList<>(Collections.nCopies(n, 0.)))
                .limit(n).collect(Collectors.toList()));
        for (int i = 0; i < n; i++) {
            set(i, i, x);
        }
    }

    public TableImpl(final List<List<Double>> table) {
        this(table.size());
        for (int i = 0; i < size(); ++i) {
            for (int j = 0; j < size(); ++j) {
                set(i, j, table.get(i).get(j));
            }
        }
    }

    public Vector multiply(final Vector x) {
        Vector b = new Vector(new ArrayList<>(Collections.nCopies(x.size(), 0.)));
        for (int i = 0; i < x.size(); i++) {
            for (int j = 0; j < x.size(); j++) {
                b.set(i, b.get(i) + get(i, j) * x.get(j));
            }
        }
        return b;
    }

    public TableImpl add(final TableImpl B) {
        TableImpl C = new TableImpl(size());
        for (int i = 0; i < size(); i++) {
            for (int j = 0; j < size(); j++) {
                C.set(i, j, get(i, j) + B.get(i, j));
            }
        }
        return C;
    }

    public TableImpl subtract(final TableImpl B) {
        TableImpl C = new TableImpl(size());
        for (int i = 0; i < size(); i++) {
            for (int j = 0; j < size(); j++) {
                C.set(i, j, get(i, j) - B.get(i, j));
            }
        }
        return C;
    }

    public TableImpl multiply(final double x) {
        TableImpl C = new TableImpl(size());
        for (int i = 0; i < size(); i++) {
            for (int j = 0; j < size(); j++) {
                C.set(i, j, get(i, j) * x);
            }
        }
        return C;
    }

    public TableImpl divide(final double d) {
        TableImpl C = new TableImpl(size());
        for (int i = 0; i < size(); i++) {
            for (int j = 0; j < size(); j++) {
                C.set(i, j, get(i, j) / d);
            }
        }
        return C;
    }

    public TableImpl transpose() {
        TableImpl T = new TableImpl(size());
        for (int i = 0; i < size(); i++) {
            for (int j = 0; j < size(); j++) {
                T.set(i, j, get(j, i));
            }
        }
        return T;
    }
}