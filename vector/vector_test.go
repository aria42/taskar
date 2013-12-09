package vector

import (
	. "github.com/smartystreets/goconvey/convey"
	"testing"
)

func TestDotProd(t *testing.T) {
	Convey("Dot product should work", t, func() {
		x := []float64{1.0, 2.0, 3.0}
		So(DotProd(x, x), ShouldEqual, 14.0)
	})
}

func TestL2(t *testing.T) {
	Convey("L2 should work", t, func() {
		x := []float64{1.0, 2.0, 3.0}
		So(L2(x), ShouldEqual, 14.0)
	})
}

func TestScaling(t *testing.T) {
	Convey("ScaleInPlace", t, func() {
		x := []float64{1.0, 2.0, 3.0}
		ScaleInPlace(x, 2.0)
		So(x, ShouldResemble, []float64{2.0, 4.0, 6.0})
	})
	Convey("Scale", t, func() {
		x := []float64{1.0, 2.0, 3.0}
		So(Scale(x, 2.0), ShouldResemble, []float64{2.0, 4.0, 6.0})
		So(x, ShouldResemble, []float64{1.0, 2.0, 3.0})
	})
}
