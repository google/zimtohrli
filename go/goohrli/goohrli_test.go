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
	"reflect"
	"testing"
)

func TestMeasureAndNormalize(t *testing.T) {
	signal := []float32{1, 2, -1, -2}
	measurements := Measure(signal)
	if measurements.MaxAbsAmplitude != 2 {
		t.Errorf("MaxAbsAmplitude = %v, want %v", measurements.MaxAbsAmplitude, 2)
	}
	wantEnergyDBFS := float32(20 * math.Log10(2.5))
	if math.Abs(float64(measurements.EnergyDBFS-float32(wantEnergyDBFS))) > 1e-4 {
		t.Errorf("EnergyDBFS = %v, want %v", measurements.EnergyDBFS, wantEnergyDBFS)
	}
	NormalizeAmplitude(1, signal)
	wantNormalizedSignal := []float32{0.5, 1, -0.5, -1}
	if !reflect.DeepEqual(signal, wantNormalizedSignal) {
		t.Errorf("NormalizeAmplitude produced %+v, want %+v", signal, wantNormalizedSignal)
	}
}

func TestMOSFromZimtohrli(t *testing.T) {
	for _, tc := range []struct {
		zimtDistance float64
		wantMOS      float64
	}{
		{
			zimtDistance: 0,
			wantMOS:      5.0,
		},
		{
			zimtDistance: 0.1,
			wantMOS:      3.9802114963531494,
		},
		{
			zimtDistance: 0.5,
			wantMOS:      1.9183233976364136,
		},
		{
			zimtDistance: 0.7,
			wantMOS:      1.5097649097442627,
		},
		{
			zimtDistance: 1.0,
			wantMOS:      1.210829496383667,
		},
	} {
		if mos := MOSFromZimtohrli(tc.zimtDistance); math.Abs(mos-tc.wantMOS) > 1e-2 {
			t.Errorf("MOSFromZimtohrli(%v) = %v, want %v", tc.zimtDistance, mos, tc.wantMOS)
		}
	}
}

func TestGettersSetters(t *testing.T) {
	g := New(48000.0, 4.0)

	nsimStepWindow := g.GetNSIMStepWindow()
	g.SetNSIMStepWindow(nsimStepWindow * 2)
	if g.GetNSIMStepWindow() != nsimStepWindow*2 {
		t.Errorf("NSIMStepWindow = %v, want %v", g.GetNSIMStepWindow(), nsimStepWindow*2)
	}

	nsimChannelWindow := g.GetNSIMChannelWindow()
	g.SetNSIMChannelWindow(nsimChannelWindow * 2)
	if g.GetNSIMChannelWindow() != nsimChannelWindow*2 {
		t.Errorf("NSIMChannelWindow = %v, want %v", g.GetNSIMChannelWindow(), nsimChannelWindow*2)
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
			distance: 0.0001035928726196289,
		},
		{
			freqA:    5000,
			freqB:    10000,
			distance: 0.23286527395248413,
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

func TestViSQOL(t *testing.T) {
	sampleRate := 48000.0
	g := NewViSQOL()
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
		mos, err := g.MOS(sampleRate, soundA, soundB)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(mos-tc.wantMOS) > 1e-3 {
			t.Errorf("got mos %v, wanted mos %v", mos, tc.wantMOS)
		}
	}
}
