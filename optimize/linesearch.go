package optimize

import (
	// "fmt"
	"github.com/aria42/taskar/vector"
)

type lineSearchParams struct {
	// Line step factor in (0,1)
	alpha float64
	// Termination criterion (0,0.5)
	beta float64
	// alpha termination
	stepLenThresh float64
}

func (params *lineSearchParams) LineSearch(f GradientFn, xs, dir []float64) (float64, float64) {
	f0, grad := f.EvalAt(xs)
	// fmt.Printf("L2(grad): %v\n", vector.L2(grad))
	if vector.L2(grad) < params.stepLenThresh {
		// already at minimum
		// fmt.Printf("lineSearch: Already at minimum %v\n", vector.L2(grad))
		return 0.0, f0
	}
	delta := params.beta * vector.DotProd(grad, dir)
	stepLen := 1.0

	for stepLen > params.stepLenThresh {
		stepX := vector.Add(xs, dir, 1.0, stepLen)
		fnVal, _ := f.EvalAt(stepX)
		// fmt.Printf("lineSearch: %v, curVal: %.5f trgVal: %.5f\n", params.alpha, fnVal, f0+stepLen*delta)
		if fnVal <= f0+stepLen*delta {
			return stepLen, fnVal
		}
		stepLen *= params.alpha
	}
	panic("Step-size underflow")
}
