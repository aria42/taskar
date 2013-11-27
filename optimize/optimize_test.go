package optimize

import (
	. "github.com/smartystreets/goconvey/convey"
	"testing"
)

var (
	xSquared = NewGradientFn(1, func(xs []float64) (val float64, grad []float64) {
		val = xs[0] * xs[0]
		grad = []float64{2 * xs[0]}
		return
	})
)

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
