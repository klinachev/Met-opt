package methods.result;


public class ParabolaIterationResult implements IterationResult {
    private final double x1;
    private final double x2;
    private final double x3;

    private final double fx1;
    private final double fx2;
    private final double fx3;

    private final double minX;
    private final double fMinX;

    public ParabolaIterationResult(double x1, double x2, double x3, double minX,
                                   double fx1, double fx2, double fx3, double fMinX) {
        this.x1 = x1;
        this.x2 = x2;
        this.x3 = x3;
        this.fx1 = fx1;
        this.fx2 = fx2;
        this.fx3 = fx3;
        this.minX = minX;
        this.fMinX = fMinX;
    }

    public double getX1() {
        return x1;
    }

    public double getX2() {
        return x2;
    }

    public double getX3() {
        return x3;
    }

    public double getFx1() {
        return fx1;
    }

    public double getFx2() {
        return fx2;
    }

    public double getFx3() {
        return fx3;
    }

    public double getMinX() {
        return minX;
    }

    public double getfMinX() {
        return fMinX;
    }

    public String tableArgs() {
        return "X1 & X2 & X3 & F(X1) & F(X2) & F(X3) & Minimal X & F(Minimal) & X3 - X1";
    }

    public String asTable() {
        return String.format("%.7f & %.7f & %.7f & %.7f & %.7f & %.7f & %.7f & %.7f & %.7f",
                getX1(),
                getX2(),
                getX3(),
                getFx1(),
                getFx2(),
                getFx3(),
                getMinX(),
                getfMinX(),
                getX3() - getX1()
        );
    }

    public String toString() {
        return String.format("%.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f",
                getX1(),
                getX2(),
                getX3(),
                getFx1(),
                getFx2(),
                getFx3(),
                getMinX(),
                getfMinX()
        );
    }
}