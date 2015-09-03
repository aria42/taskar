package optimize

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

var xSquared = NewGradientFn(1, func(xs []float64) (val float64, grad []float64) {
	val = xs[0] * xs[0]
	grad = []float64{2 * xs[0]}
	return
})
var quarticFn = NewGradientFn(2, func(xs []float64) (float64, []float64) {
	x, y := xs[0], xs[1]
	val := math.Pow(x-1.0, 4.0) + math.Pow(y+2.0, 4.0)
	grad := []float64{4 * math.Pow(x-1.0, 3.0), 4 * math.Pow(y+2.0, 3.0)}
	return val, grad
})

func TestCachingGradientFn(t *testing.T) {
	Convey("NewCachingGradientFn should cache", t, func() {
		numCalls := 0
		f := NewCachingGradientFn(1, NewGradientFn(1, func(xs []float64) (float64, []float64) {
			numCalls++
			return xSquared.EvalAt(xs)
		}))
		f.EvalAt([]float64{2})
		So(numCalls, ShouldEqual, 1)
		f.EvalAt([]float64{2})
		So(numCalls, ShouldEqual, 1)
		f.EvalAt([]float64{4})
		So(numCalls, ShouldEqual, 2)
		f.EvalAt([]float64{4})
		So(numCalls, ShouldEqual, 2)
	})
}

func TestLineSearch(t *testing.T) {
	Convey("LineSearch should minimize along a direction", t, func() {
		ls := NewLineSearcher(0.5)
		stepLen, fnVal := ls.LineSearch(xSquared, []float64{1.0}, []float64{-1.0})
		So(stepLen, ShouldEqual, 1.0)
		So(fnVal, ShouldEqual, 0.0)
	})
	Convey("LineSearch should do nothing at a local minimum", t, func() {
		ls := NewLineSearcher(0.5)
		stepLen, fnVal := ls.LineSearch(xSquared, []float64{0.0}, []float64{1.0})
		So(stepLen, ShouldEqual, 0.0)
		So(fnVal, ShouldEqual, 0.0)
	})
}

func TestGradientDescent(t *testing.T) {
	Convey("Gradient descent should minimize a simple function", t, func() {
		o := new(NewtonOpts)
		o.InitGuess = []float64{1.0}
		xmin := NewGradientDescent(o).Minimize(xSquared)
		So(xmin[0], ShouldEqual, 0.0)
	})
	// (x-2)^4 + (y+3)^4
	Convey("Gradient descent should minimize a 2-variable function", t, func() {
		o := new(NewtonOpts)
		o.InitGuess = []float64{0.0, 0.0}
		o.Tolerance = 1.0e-5
		xmin := NewGradientDescent(o).Minimize(quarticFn)
		minfx, _ := quarticFn.EvalAt(xmin)
		So(math.Abs(minfx-0.0), ShouldBeLessThan, 1.0e-4)
	})
}

func TestLBFGS(t *testing.T) {
	// x^2
	Convey("LBFG should minimize a simple function", t, func() {
		o := new(NewtonOpts)
		o.InitGuess = []float64{1.0}
		xmin := NewLBFGS(o, 2).Minimize(xSquared)
		So(xmin[0], ShouldEqual, 0.0)
	})
	// (x - 2) ^ 4 + (y + 3) ^ 4
	Convey("LBFG should minimize a 2-variable function", t, func() {
		o := new(NewtonOpts)
		o.InitGuess = []float64{0.0, 0.0}
		o.Tolerance = 1.0e-5
		xmin := NewLBFGS(o, 2).Minimize(quarticFn)
		minfx, _ := quarticFn.EvalAt(xmin)
		So(math.Abs(minfx-0.0), ShouldBeLessThan, 1.0e-4)
	})

}
