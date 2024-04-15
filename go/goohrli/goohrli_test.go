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

package goohrli

import (
	"math"
	"testing"
)

func TestGettersSetters(t *testing.T) {
	g := New(48000.0, 4.0)

	timeNormOrder := g.GetTimeNormOrder()
	if timeNormOrder != 4 {
		t.Errorf("TimeNormOrder = %v, want %v", timeNormOrder, 4)
	}
	g.SetTimeNormOrder(timeNormOrder * 2)
	if g.GetTimeNormOrder() != timeNormOrder*2 {
		t.Errorf("TimeNormOrder = %v, want %v", g.GetTimeNormOrder(), timeNormOrder*2)
	}

	freqNormOrder := g.GetFreqNormOrder()
	if freqNormOrder != 4 {
		t.Errorf("FreqNormOrder = %v, want %v", freqNormOrder, 4)
	}
	g.SetFreqNormOrder(freqNormOrder * 2)
	if g.GetFreqNormOrder() != freqNormOrder*2 {
		t.Errorf("FreqNormOrder = %v, want %v", g.GetFreqNormOrder(), freqNormOrder*2)
	}
}

func TestGoohrli(t *testing.T) {
	for _, tc := range []struct {
		freqA    float64
		freqB    float64
		distance float64
	}{
		{
			freqA:    5000,
			freqB:    5000,
			distance: 0,
		},
		{
			freqA:    5000,
			freqB:    5010,
			distance: 1.4678096771240234,
		},
		{
			freqA:    5000,
			freqB:    10000,
			distance: 55.7608528137207,
		},
	} {
		sampleRate := 48000.0
		frequencyResolution := 4.0
		g := New(sampleRate, frequencyResolution)
		soundA := make([]float32, int(sampleRate))
		for index := 0; index < len(soundA); index++ {
			soundA[index] = float32(math.Sin(2 * math.Pi * tc.freqA * float64(index) / sampleRate))
		}
		analysisA := g.Analyze(soundA)
		soundB := make([]float32, int(sampleRate))
		for index := 0; index < len(soundB); index++ {
			soundB[index] = float32(math.Sin(2 * math.Pi * tc.freqB * float64(index) / sampleRate))
		}
		analysisB := g.Analyze(soundB)
		analysisDistance := float64(g.AnalysisDistance(analysisA, analysisB))
		if math.Abs(analysisDistance-tc.distance) > 1e-3 {
			t.Errorf("Distance = %v, want %v", analysisDistance, tc.distance)
		}
		distance := float64(g.Distance(soundA, soundB))
		if math.Abs(distance-tc.distance) > 1e-3 {
			t.Errorf("Distance = %v, want %v", distance, tc.distance)
		}
	}
}
