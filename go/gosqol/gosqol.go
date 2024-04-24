// Copyright 2024 The Zimtohrli Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package gosqol provides a Go wrapper around zimtohrli::ViSQOL.
package gosqol

/*
#cgo LDFLAGS: ${SRCDIR}/gosqol.a -lz -lm -lstdc++
#cgo CFLAGS: -O3
#include "gosqol.h"
*/
import "C"
import (
	"runtime"
)

// Gosqol is a Go wrapper around zimtohrli::ViSQOL.
type Gosqol struct {
	visqol C.ViSQOL
}

// New returns a new Gosqol.
func New() *Gosqol {
	result := &Gosqol{
		visqol: C.CreateViSQOL(),
	}
	runtime.SetFinalizer(result, func(g *Gosqol) {
		C.FreeViSQOL(g.visqol)
	})
	return result
}

func (g *Gosqol) MOS(sampleRate float64, reference []float32, degraded []float32) float64 {
	return float64(C.MOS(g.visqol, C.float(sampleRate), (*C.float)(&reference[0]), C.int(len(reference)), (*C.float)(&degraded[0]), C.int(len(degraded))))
}
