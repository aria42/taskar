package optimize

import (
	"container/list"
)

type GradientFn interface {
	EvalAt(x []float64) (fx float64, grad []float64)
	Dimension() int64
}

type LineSearcher interface {
	LineSearch(f GradientFn, xs, dir []float64) (stepLength, fnVal float64)
}

type Minimizer interface {
	Minimize(f GradientFn) (xmin []float64)
}

func NewGradientFn(dim int64, f gradientFnType) GradientFn {
	return &gradientFn{f, dim}
}

func NewCachingGradientFn(maxToCache int, fn GradientFn) GradientFn {
	return &cachingGradientFn{fn, list.New(), maxToCache}
}

func NewLineSearcher(alpha float64) LineSearcher {
	return &lineSearchParams{alpha, 0.0001, 0.0001}
}

type NewtonOpts struct {
	InitGuess []float64
	MaxIters  int
	Tolerance float64
	initAlpha float64
	alpha     float64
}

func NewGradientDescent(opts *NewtonOpts) Minimizer {
	opts.initAlpha = 0.5
	opts.alpha = 0.1
	return &gradientDescent{*opts}
}

func NewLBFGS(opts *NewtonOpts, maxHistory int) Minimizer {
	return &lbfgs{*opts, maxHistory}
}
