package optimize

import (
	"container/list"
	"fmt"
	"math"

	"../vector"
)

// A function which acts like f(x) = H^{-1}x, where
// H is the hessian of a function
type invHessianMutiply func(x []float64) []float64

func newtonStep(f GradientFn, x []float64, l LineSearcher, hinv invHessianMutiply) []float64 {
	_, grad := f.EvalAt(x)
	// Newton direction is -H^{-1}grad(f_x)
	dir := hinv(grad)
	vector.ScaleInPlace(dir, -1.0)
	alpha, _ := l.LineSearch(f, x, dir)
	return vector.Add(x, dir, 1.0, alpha)
}

// Given a point of history return a new implicit inverse hessian
// multiply. This strategy defines the quasi-newton method
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
		xnew := newtonStep(f, x, NewLineSearcher(opts.alpha), update(x))
		fxnew, gradnew := f.EvalAt(xnew)
		if fxnew > fx {
			panic(fmt.Sprintf("newtonStep did not minimize: %.5f -> %.5f", fx, fxnew))
		}
		var reldiff float64
		if fx == fxnew {
			reldiff = 0.0
		} else {
			reldiff = math.Abs(fx-fxnew) / math.Abs(fxnew)
		}
		x = xnew
		// fmt.Printf("Iteration %d: began with %v, ended with value %v\n", iter, fx, fxnew)
		// fmt.Printf("Iteration %d: at x=%v\n", iter, x)
		// fmt.Printf("Iteration %d: gradient with %v and  relDiff %v\n", iter, vector.L2(gradnew), reldiff)
		// fmt.Println()
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

func (entry *historyEntry) curvature() float64 {
	curvature := vector.DotProd(entry.xdelta, entry.graddelta)
	if curvature < 0.0 {
		panic(fmt.Sprintf("Negative Curvature: %0.5f", curvature))
	}
	return curvature
}

func (l *lbfgs) Minimize(f GradientFn) []float64 {
	history := list.New()
	var lastx []float64
	updateFn := func(x []float64) invHessianMutiply {
		var gamma float64
		if lastx != nil {
			// xdelta = x - lastx
			xdelta := vector.Add(x, lastx, 1.0, -1.0)
			_, grad := f.EvalAt(x)
			_, lastgrad := f.EvalAt(lastx)
			graddelta := vector.Add(grad, lastgrad, 1.0, -1.0)
			entry := &historyEntry{xdelta, graddelta}
			gamma = entry.curvature() / vector.DotProd(graddelta, graddelta)

			// Update history
			history.PushFront(entry)
			if history.Len() > l.maxHistory {
				history.Remove(history.Back())
			}
		} else {
			gamma = 1.0
		}
		lastx = x
		return func(dir []float64) []float64 {
			result := make([]float64, len(dir))
			copy(result, dir)
			// forward history pass
			for e := history.Front(); e != nil; e = e.Next() {
				entry := e.Value.(*historyEntry)
				alpha := vector.DotProd(entry.xdelta, result) / entry.curvature()
				vector.AddInPlace(result, entry.graddelta, -alpha)
			}
			vector.ScaleInPlace(result, 1.0/gamma)
			// backward pass
			for e := history.Back(); e != nil; e = e.Prev() {
				entry := e.Value.(*historyEntry)
				curvature := entry.curvature()
				alpha := vector.DotProd(entry.xdelta, result) / curvature
				beta := vector.DotProd(entry.graddelta, result) / curvature
				vector.AddInPlace(result, entry.xdelta, alpha-beta)
			}
			return result
		}
	}
	return newtonMinimize(f, updateFn, &l.NewtonOpts)
}
