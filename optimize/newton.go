package optimize

import (
	"container/list"
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
		// fmt.Printf("newtonIter: iter %d, fx: %.5f\n", iter, fx)
		xnew := newtonStep(f, x, NewLineSearcher(alpha), update(x))
		fxnew, gradnew := f.EvalAt(xnew)
		// fmt.Printf("newtonStep: iter %d, fx: %.5f fxnew: %.5f\n", iter, fx, fxnew)
		if fxnew > fx {
			panic(fmt.Sprintf("newtonStep did not minimize: %.5f -> %.5f", fx, fxnew))
		}
		var reldiff float64
		if fx == fxnew {
			reldiff = 0.0
		} else {
			reldiff = math.Abs(fx-fxnew) / math.Abs(fxnew)
		}
		// fmt.Printf("gradnew %v norm %.5f\n", gradnew, vector.L2(gradnew))
		x = xnew
		fmt.Printf("Iteration %d began with %v, ended with value %v\n", iter, fx, fxnew)
		if reldiff <= opts.Tolerance || vector.L2(gradnew) <= opts.Tolerance {
			break
		}
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

type historyEntry struct {
	xdelta    []float64
	graddelta []float64
}

func (l *lbfgs) Minimize(f GradientFn) []float64 {
	history := list.New()
	var lastx []float64
	updateFn := func(x []float64) invHessianMutiply {
		gamma := 1.0
		if lastx != nil {
			xdelta := vector.Add(x, lastx, 1.0, -1.0)
			_, lastgrad := f.EvalAt(lastx)
			_, grad := f.EvalAt(x)
			graddelta := vector.Add(grad, lastgrad, 1.0, -1.0)
			history.PushBack(&historyEntry{xdelta, graddelta})

			curvature := vector.DotProd(xdelta, graddelta)
			gamma = curvature / vector.L2(graddelta)
		}
		lastx = x
		return func(dir []float64) []float64 {
			result := make([]float64, len(dir))
			copy(result, dir)
			alphas := make([]float64, history.Len())
			idx := 0
			// forward history pass
			for e := history.Front(); e != nil; e = e.Next() {
				entry := e.Value.(*historyEntry)
				curvature := vector.DotProd(entry.xdelta, entry.graddelta)
				alpha := vector.DotProd(entry.xdelta, result) / curvature
				vector.AddInPlace(result, entry.graddelta, -alpha)
				alphas[idx] = alpha
				idx++
			}
			vector.ScaleInPlace(result, gamma)
			// backward pass
			idx = len(alphas) - 1
			for e := history.Back(); e != nil; e = e.Prev() {
				entry := e.Value.(*historyEntry)
				rho := vector.DotProd(entry.graddelta, result) / vector.DotProd(entry.xdelta, entry.graddelta)
				vector.AddInPlace(result, entry.xdelta, alphas[idx]-rho)
				idx--
			}
			vector.ScaleInPlace(result, gamma)
			return result
		}
	}
	return newtonMinimize(f, updateFn, &l.NewtonOpts)
}
