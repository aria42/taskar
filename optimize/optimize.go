// Standard unconstraiend numerical optimization
package optimize

import (
	"container/list"
)

type GradientFn interface {
	// return pair of f(x) and gradient of f at x
	EvalAt(x []float64) (fx float64, grad []float64)
	Dimension() int64
}

type LineSearcher interface {
	// approximately minimize_stepLen f(x + stepLen * dir) and
	// return pair of minStepLen and f(x + minStepLen * dir)
	LineSearch(f GradientFn, xs, dir []float64) (stepLength, fnVal float64)
}

type Minimizer interface {
	// Minimize a gradient fn. Only well-defined
	// when f is strongly convex
	Minimize(f GradientFn) (xmin []float64)
}

// Make a GradientFn from a dimention int and a fn from
// []float64 => (float64, []float64)
func NewGradientFn(dim int64, f gradientFnType) GradientFn {
	return &gradientFn{f, dim}
}

// Build a caching Gradient fn from an existing GradientFn and a max
// history to store
func NewCachingGradientFn(maxToCache int, fn GradientFn) GradientFn {
	return &cachingGradientFn{fn, list.New(), maxToCache}
}

// Return standard backtracking line-searcher
func NewLineSearcher(alpha float64) LineSearcher {
	return &lineSearchParams{alpha, 0.01, 1.0e-10}
}

// Options for Newtown opitimiziation of a GradientFn
type NewtonOpts struct {
	InitGuess []float64
	MaxIters  int
	Tolerance float64
	alpha     float64
}

// Create standard gradient descent minimizer
func NewGradientDescent(opts *NewtonOpts) Minimizer {
	opts.alpha = 0.5
	return &gradientDescent{*opts}
}

// Create limited-memory Quasi-Newton minimizer form NewtonOpts
// and a `maxHistory` parameter for how much history to use
// to approximate inverse hessian multiplicaiton
func NewLBFGS(opts *NewtonOpts, maxHistory int) Minimizer {
	opts.alpha = 0.5
	return &lbfgs{*opts, maxHistory}
}
