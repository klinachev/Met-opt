package methods.result;

public class DefaultIterationResult implements IterationResult {
    private final double a;
    private final double b;
    private final double x1;
    private final double x2;

    private final double fx1;
    private final double fx2;

    public DefaultIterationResult(double a, double b, double x1, double x2, double fx1, double fx2) {
        this.a = a;
        this.b = b;
        this.x1 = x1;
        this.x2 = x2;
        this.fx1 = fx1;
        this.fx2 = fx2;
    }

    public double getA() {
        return a;
    }

    public double getB() {
        return b;
    }

    public double getX1() {
        return x1;
    }

    public double getX2() {
        return x2;
    }

    public double getFx1() {
        return fx1;
    }

    public double getFx2() {
        return fx2;
    }

    public String tableArgs() {
        return "A & B & X1 & X2 & F(X1) & F(X2) & B - A";
    }

    public String asTable() {
        return String.format("%.7f & %.7f & %.7f & %.7f & %.7f & %.7f & %.7f",
                getA(),
                getB(),
                getX1(),
                getX2(),
                getFx1(),
                getFx2(),
                getB() - getA());
    }

    public String toString() {
        return String.format("%.16f %.16f %.16f %.16f %.16f %.16f",
                getA(),
                getB(),
                getX1(),
                getX2(),
                getFx1(),
                getFx2());
    }
}
