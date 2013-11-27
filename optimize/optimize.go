package optimize

import (
	"container/list"
	"reflect"
)

type GradientFn interface {
	EvalAt(x []float64) (fx float64, grad []float64)
	Dimension() int64
}

type gradientFnType func(xs []float64) (float64, []float64)

type gradientFn struct {
	f   gradientFnType
	dim int64
}

func NewGradientFn(dim int64, f gradientFnType) GradientFn {
	return &gradientFn{f, dim}
}

func (g *gradientFn) EvalAt(xs []float64) (float64, []float64) {
	return g.f(xs)
}

func (g *gradientFn) Dimension() int64 {
	return g.dim
}

type cachingGradientFn struct {
	fn         GradientFn
	entries    *list.List
	maxToCache int
}

type cacheEntry struct {
	val   float64
	grad  []float64
	input []float64
}

func (cdf *cachingGradientFn) EvalAt(xs []float64) (float64, []float64) {
	var c *cacheEntry
	for e := cdf.entries.Front(); e != nil; e = e.Next() {
		entry := e.Value.(*cacheEntry)
		if reflect.DeepEqual(entry.input, xs) {
			c = entry
			break
		}
	}
	if c != nil {
		return c.val, c.grad
	} else {
		val, grad := cdf.fn.EvalAt(xs)
		newEntry := &cacheEntry{val, grad, xs}
		cdf.entries.PushFront(newEntry)
		if cdf.entries.Len() > cdf.maxToCache {
			cdf.entries.Remove(cdf.entries.Back())
		}
		return val, grad
	}
}

func (c *cachingGradientFn) Dimension() int64 {
	return c.fn.Dimension()
}

func NewCachingGradientFn(maxToCache int, fn GradientFn) GradientFn {
	return &cachingGradientFn{fn, list.New(), maxToCache}
}
