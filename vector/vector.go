package vector

import (
	"fmt"
	"math"
)

func checkEqualLen(xs, ys []float64) {
	if len(xs) != len(ys) {
		panic(fmt.Sprintf("Lengths not equal: %d %d", len(xs), len(ys)))
	}
}

func DotProd(xs, ys []float64) (sum float64) {
	checkEqualLen(xs, ys)
	for idx, x := range xs {
		sum += x * ys[idx]
	}
	return sum
}

func Add(xs, ys []float64, alpha, beta float64) []float64 {
	checkEqualLen(xs, ys)
	result := make([]float64, len(xs))
	for idx, val := range xs {
		result[idx] = alpha*val + beta*ys[idx]
	}
	return result
}

func L2(xs []float64) (magnitude float64) {
	for _, val := range xs {
		magnitude += val * val
	}
	return
}

func L1(xs []float64) (magnitude float64) {
	for _, val := range xs {
		magnitude += math.Abs(val)
	}
	return
}
