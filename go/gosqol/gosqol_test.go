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

package gosqol

import (
	"math"
	"testing"
)

func TestGosqol(t *testing.T) {
	sampleRate := 48000.0
	g := New()
	for _, tc := range []struct {
		freqA   float64
		freqB   float64
		wantMOS float64
	}{
		{
			freqA:   5000,
			freqB:   5000,
			wantMOS: 4.7321014404296875,
		},
		{
			freqA:   5000,
			freqB:   10000,
			wantMOS: 1.5407887697219849,
		},
	} {
		soundA := make([]float32, int(sampleRate))
		for index := 0; index < len(soundA); index++ {
			soundA[index] = float32(math.Sin(2 * math.Pi * tc.freqA * float64(index) / sampleRate))
		}
		soundB := make([]float32, int(sampleRate))
		for index := 0; index < len(soundB); index++ {
			soundB[index] = float32(math.Sin(2 * math.Pi * tc.freqB * float64(index) / sampleRate))
		}
		if mos := g.MOS(sampleRate, soundA, soundB); math.Abs(mos-tc.wantMOS) > 1e-3 {
			t.Errorf("got mos %v, wanted mos %v", mos, tc.wantMOS)
		}
	}
}
