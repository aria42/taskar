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
	Minimize(f GradientFn, initGuess []float64) (xmin []float64, fmin []float64)
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
