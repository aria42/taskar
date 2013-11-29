package optimize

import (
	"fmt"
	"github.com/aria42/taskar/vector"
	"math"
)

type invHessianMutiply func(x []float64) []float64

func newtonStep(f GradientFn, x []float64, l LineSearcher, hinv invHessianMutiply) []float64 {
	_, grad := f.EvalAt(x)
	dir := hinv(grad)
	vector.ScaleInPlace(dir, -1.0)
	// fmt.Printf("newtonStep: x: %v grad: %v dir: %v\n", x, grad, dir)
	alpha, _ := l.LineSearch(f, x, dir)
	return vector.Add(x, dir, 1.0, alpha)
}

type updateInvHessianGuess func(x []float64) invHessianMutiply

func newtonMinimize(f GradientFn, update updateInvHessianGuess, opts *NewtonOpts) []float64 {
	var x []float64
	if opts.InitGuess != nil {
		x = opts.InitGuess
	} else {
		x = make([]float64, f.Dimension(), f.Dimension())
	}
	for iter := 0; opts.MaxIters == 0 || iter < opts.MaxIters; iter++ {
		fx, _ := f.EvalAt(x)
		var alpha float64
		if iter == 0 {
			alpha = opts.initAlpha
		} else {
			alpha = opts.alpha
		}
		l := NewLineSearcher(alpha)
		// fmt.Printf("newtonIter: iter %d, fx: %.5f\n", iter, fx)
		xnew := newtonStep(f, x, l, update(x))
		fxnew, _ := f.EvalAt(xnew)
		if fxnew > fx {
			panic(fmt.Sprintf("newtonStep did not minimize: %.5f -> %.5f", fx, fxnew))
		}
		var reldiff float64
		if fx == fxnew {
			reldiff = 0.0
		} else {
			reldiff = math.Abs(fx-fxnew) / math.Abs(fxnew)
		}
		// fmt.Printf("fx: %.5f fxnew: %.5f reldirr: %0.5f\n", fx, fxnew, reldiff)
		if reldiff <= opts.Tolerance {
			break
		}
		x = xnew
	}
	return x
}

type gradientDescent struct {
	NewtonOpts
}

func (g *gradientDescent) Minimize(f GradientFn) []float64 {
	// gradient descent always uses direction as is
	updateFn := func(x []float64) invHessianMutiply {
		return func(dir []float64) []float64 {
			result := make([]float64, len(dir))
			copy(result, dir)
			return result
		}
	}
	return newtonMinimize(f, updateFn, &g.NewtonOpts)
}

type lbfgs struct {
	NewtonOpts
	maxHistory int
}

func (l *lbfgs) Minimize(f GradientFn) []float64 {
	// gradient descent always uses direction as is
	updateFn := func(x []float64) invHessianMutiply {
		return func(dir []float64) []float64 {
			result := make([]float64, len(dir))
			copy(result, dir)
			return result
		}
	}
	return newtonMinimize(f, updateFn, &l.NewtonOpts)
}
