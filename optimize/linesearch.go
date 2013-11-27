package optimize

import (
	"github.com/aria42/taskar/vector"
)

type LineSearcher interface {
	LineSearch(f GradientFn, xs, dir []float64) (stepLength, fnVal float64)
}

type lineSearchParams struct {
	// Line step factor in (0,1)
	alpha float64
	// Termination criterion (0,0.5)
	beta float64
	// alpha termination
	stepLenThresh float64
}

func NewLineSearcher(alpha float64) LineSearcher {
	return &lineSearchParams{alpha, 0.0001, 0.0001}
}

func (params *lineSearchParams) LineSearch(f GradientFn, xs, dir []float64) (float64, float64) {
	f0, grad := f.EvalAt(xs)
	if vector.L2(grad) < params.beta {
		// already at minimum
		return 0.0, f0
	}
	delta := params.beta * vector.DotProd(grad, dir)
	stepLen := 1.0
	for stepLen > params.stepLenThresh {
		stepX := vector.Add(xs, dir, 1.0, stepLen)
		fnVal, _ := f.EvalAt(stepX)
		if fnVal < f0+stepLen*delta {
			return stepLen, fnVal
		}
		stepLen *= params.alpha
	}
	panic("Step-size underflow")
}
