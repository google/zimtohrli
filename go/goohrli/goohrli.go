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

// Package goohrli provides a Go wrapper around zimtohrli::Zimtohrli.
package goohrli

/*
#cgo LDFLAGS: -lc++ -lm -lasound -lFLAC -logg -lvorbis -lvorbisenc -lopus ${SRCDIR}/goohrli.a
#include "goohrli.h"
*/
import "C"
import (
	"runtime"
	"time"
)

// EnergyAndMaxAbsAmplitude is holds the energy and maximum absolute amplitude of a measurement.
type EnergyAndMaxAbsAmplitude struct {
	EnergyDBFS      float32
	MaxAbsAmplitude float32
}

// Measure returns the energy in dB FS and maximum absolute amplitude of the signal.
func Measure(signal []float32) EnergyAndMaxAbsAmplitude {
	measurements := C.Measure((*C.float)(&signal[0]), C.int(len(signal)))
	return EnergyAndMaxAbsAmplitude{
		EnergyDBFS:      float32(measurements.EnergyDBFS),
		MaxAbsAmplitude: float32(measurements.MaxAbsAmplitude),
	}
}

// NormalizeAmplitude normalizes the amplitudes of the signal so that it has the provided max
// amplitude, and returns the new energ in dB FS, and the new maximum absolute amplitude.
func NormalizeAmplitude(maxAbsAmplitude float32, signal []float32) EnergyAndMaxAbsAmplitude {
	measurements := C.NormalizeAmplitudes(C.float(maxAbsAmplitude), (*C.float)(&signal[0]), C.int(len(signal)))
	return EnergyAndMaxAbsAmplitude{
		EnergyDBFS:      float32(measurements.EnergyDBFS),
		MaxAbsAmplitude: float32(measurements.MaxAbsAmplitude),
	}
}

// Goohrli is a Go wrapper around zimtohrli::Zimtohrli.
type Goohrli struct {
	zimtohrli           C.Zimtohrli
	sampleRate          float64
	frequencyResolution float64

	// PerceptualSampleRate is the sample rate of returned analyses. Defaults to 100, i.e. 10ms is the
	// smallest time difference human hearing is expected to notice.
	PerceptualSampleRate float32

	// UnwarpWindow is the duration of a window when unwarping the timeline using dynamic time warp.
	UnwarpWindow time.Duration
}

// New returns a new Goohrli for the given parameters.
//
// sampleRate is the sample rate of input audio.
//
// frequencyResolution is the width of the lowest frequency channel, i.e. the expected frequency
// resolution of human hearing.
func New(sampleRate float64, frequencyResolution float64) *Goohrli {
	result := &Goohrli{
		zimtohrli:            C.CreateZimtohrli(C.float(sampleRate), C.float(frequencyResolution)),
		sampleRate:           sampleRate,
		frequencyResolution:  frequencyResolution,
		PerceptualSampleRate: 100.0,
		UnwarpWindow:         2 * time.Second,
	}
	runtime.SetFinalizer(result, func(g *Goohrli) {
		C.FreeZimtohrli(g.zimtohrli)
	})
	return result
}

// SampleRate returns the expected sample rate of input audio.
func (g *Goohrli) SampleRate() float64 { return g.sampleRate }

// FrequencyResolution returns the configured frequency resolution.
func (g *Goohrli) FrequencyResolution() float64 { return g.frequencyResolution }

// Analysis is a Go wrapper around zimthrli::Analysis.
type Analysis struct {
	analysis C.Analysis
}

// Analyze returns an analysis of the signal.
func (g *Goohrli) Analyze(signal []float32) *Analysis {
	result := &Analysis{
		analysis: C.Analyze(g.zimtohrli, C.float(g.PerceptualSampleRate), (*C.float)(&signal[0]), C.int(len(signal))),
	}
	runtime.SetFinalizer(result, func(a *Analysis) {
		C.FreeAnalysis(a.analysis)
	})
	return result
}

// AnalysisDistance returns the Zimtohrli distance between two analyses.
func (g *Goohrli) AnalysisDistance(analysisA *Analysis, analysisB *Analysis) float32 {
	return float32(C.AnalysisDistance(g.zimtohrli, analysisA.analysis, analysisB.analysis, C.int(float64(g.PerceptualSampleRate)*g.UnwarpWindow.Seconds())))
}

// Distance returns the Zimtohrli distance between two signals.
func (g *Goohrli) Distance(signalA []float32, signalB []float32) float32 {
	analysisA := C.Analyze(g.zimtohrli, C.float(g.PerceptualSampleRate), (*C.float)(&signalA[0]), C.int(len(signalA)))
	defer C.FreeAnalysis(analysisA)
	analysisB := C.Analyze(g.zimtohrli, C.float(g.PerceptualSampleRate), (*C.float)(&signalB[0]), C.int(len(signalB)))
	defer C.FreeAnalysis(analysisB)
	return float32(C.AnalysisDistance(g.zimtohrli, analysisA, analysisB, C.int(g.PerceptualSampleRate)))
}

// GetTimeNormOrder returns the order of the norm across time steps when computing Zimtohrli distance.
func (g *Goohrli) GetTimeNormOrder() float32 {
	return float32(C.GetTimeNormOrder(g.zimtohrli))
}

// SetTimeNormOrder sets the order of the norm across time steps when computing Zimtohrli distance.
func (g *Goohrli) SetTimeNormOrder(f float32) {
	C.SetTimeNormOrder(g.zimtohrli, C.float(f))
}

// GetFreqNormOrder returns the order of the norm across frequency channels when computing Zimtohrli distance.
func (g *Goohrli) GetFreqNormOrder() float32 {
	return float32(C.GetFreqNormOrder(g.zimtohrli))
}

// SetFreqNormOrder sets the order of the norm across frequency channels when computing Zimtohrli distance.
func (g *Goohrli) SetFreqNormOrder(f float32) {
	C.SetFreqNormOrder(g.zimtohrli, C.float(f))
}
