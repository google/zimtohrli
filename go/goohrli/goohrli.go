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
#cgo LDFLAGS: ${SRCDIR}/goohrli.a -lz -lopus -lFLAC -lvorbis -lvorbisenc -logg -lasound -lm -lstdc++
#cgo CFLAGS: -O3
#include "goohrli.h"
*/
import "C"
import (
	"fmt"
	"math"
	"runtime"
	"time"

	"github.com/google/zimtohrli/go/audio"
)

// DefaultFrequencyResolution returns the default frequency resolution corresponding to the minimum width (at the low frequency end) of the Zimtohrli filter bank.
func DefaultFrequencyResolution() float64 {
	return float64(C.DefaultFrequencyResolution())
}

// DefaultPerceptualSampleRate returns the default perceptual sample rate corresponding to the human hearing sensitivity to timing changes.
func DefaultPerceptualSampleRate() float64 {
	return float64(C.DefaultPerceptualSampleRate())
}

// DefaultNSIMStepWindow returns the default NSIM step window.
func DefaultNSIMStepWindow() int {
	return int(C.DefaultNSIMStepWindow())
}

func DefaultNSIMChannelWindow() int {
	return int(C.DefaultNSIMChannelWindow())
}

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
	measurements := C.NormalizeAmplitude(C.float(maxAbsAmplitude), (*C.float)(&signal[0]), C.int(len(signal)))
	return EnergyAndMaxAbsAmplitude{
		EnergyDBFS:      float32(measurements.EnergyDBFS),
		MaxAbsAmplitude: float32(measurements.MaxAbsAmplitude),
	}
}

// MOSFromZimtohrli returns an approximate mean opinion score for a given zimtohrli distance.
func MOSFromZimtohrli(zimtohrliDistance float64) float64 {
	return float64(C.MOSFromZimtohrli(C.float(zimtohrliDistance)))
}

// Goohrli is a Go wrapper around zimtohrli::Zimtohrli.
type Goohrli struct {
	zimtohrli           C.Zimtohrli
	sampleRate          float64
	frequencyResolution float64

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
		zimtohrli:           C.CreateZimtohrli(C.float(sampleRate), C.float(frequencyResolution)),
		sampleRate:          sampleRate,
		frequencyResolution: frequencyResolution,
		UnwarpWindow:        2 * time.Second,
	}
	runtime.SetFinalizer(result, func(g *Goohrli) {
		C.FreeZimtohrli(g.zimtohrli)
	})
	return result
}

func (g *Goohrli) String() string {
	return fmt.Sprintf("%+v", map[string]any{
		"SampleRate":           g.sampleRate,
		"FrequencyResolution":  g.frequencyResolution,
		"UnwarpWindow":         g.UnwarpWindow,
		"PerceptualSampleRate": g.GetPerceptualSampleRate(),
		"NSIMStepWindow":       g.GetNSIMStepWindow(),
		"NSIMChannelWindow":    g.GetNSIMChannelWindow(),
	})
}

// SampleRate returns the expected sample rate of input audio.
func (g *Goohrli) SampleRate() float64 { return g.sampleRate }

// FrequencyResolution returns the configured frequency resolution.
func (g *Goohrli) FrequencyResolution() float64 { return g.frequencyResolution }

// NormalizedAudioDistance returns the distance between the audio files after normalizing their amplitudes for the same max amplitude.
func (g *Goohrli) NormalizedAudioDistance(audioA, audioB *audio.Audio) (float64, error) {
	sumOfSquares := 0.0
	if g.sampleRate != audioA.Rate || g.sampleRate != audioB.Rate {
		return 0, fmt.Errorf("one of the audio files doesn't have the expected sample rate %v: %v, %v", g.sampleRate, audioA.Rate, audioB.Rate)
	}
	if len(audioA.Samples) != len(audioB.Samples) {
		return 0, fmt.Errorf("the audio files don't have the same number of channels: %v, %v", len(audioA.Samples), len(audioB.Samples))
	}
	if len(audioA.Samples) == 0 {
		return 0, fmt.Errorf("the audio files don't have any channels")
	}
	for channelIndex := range audioA.Samples {
		measurement := Measure(audioA.Samples[channelIndex])
		NormalizeAmplitude(measurement.MaxAbsAmplitude, audioB.Samples[channelIndex])
		dist := float64(g.Distance(audioA.Samples[channelIndex], audioB.Samples[channelIndex]))
		if math.IsNaN(dist) {
			return 0, fmt.Errorf("%v.Distance(...) returned %v", g, dist)
		}
		sumOfSquares += dist * dist
	}
	result := math.Sqrt(sumOfSquares / float64(len(audioA.Samples)))
	if math.IsNaN(result) {
		return 0, fmt.Errorf("math.Sqrt(%v / %v) is %v", sumOfSquares, len(audioA.Samples), result)
	}
	return result, nil
}

// Analysis is a Go wrapper around zimthrli::Analysis.
type Analysis struct {
	analysis C.Analysis
}

// Analyze returns an analysis of the signal.
func (g *Goohrli) Analyze(signal []float32) *Analysis {
	result := &Analysis{
		analysis: C.Analyze(g.zimtohrli, (*C.float)(&signal[0]), C.int(len(signal))),
	}
	runtime.SetFinalizer(result, func(a *Analysis) {
		C.FreeAnalysis(a.analysis)
	})
	return result
}

// AnalysisDistance returns the Zimtohrli distance between two analyses.
func (g *Goohrli) AnalysisDistance(analysisA *Analysis, analysisB *Analysis) float32 {
	return float32(C.AnalysisDistance(g.zimtohrli, analysisA.analysis, analysisB.analysis, C.int(float64(g.GetPerceptualSampleRate())*g.UnwarpWindow.Seconds())))
}

// Distance returns the Zimtohrli distance between two signals.
func (g *Goohrli) Distance(signalA []float32, signalB []float32) float64 {
	analysisA := C.Analyze(g.zimtohrli, (*C.float)(&signalA[0]), C.int(len(signalA)))
	defer C.FreeAnalysis(analysisA)
	analysisB := C.Analyze(g.zimtohrli, (*C.float)(&signalB[0]), C.int(len(signalB)))
	defer C.FreeAnalysis(analysisB)
	return float64(C.AnalysisDistance(g.zimtohrli, analysisA, analysisB, C.int(float64(g.GetPerceptualSampleRate())*g.UnwarpWindow.Seconds())))
}

// GetNSIMStepWIndow returns the window in perceptual_sample_rate time steps when compting the NSIM.
func (g *Goohrli) GetNSIMStepWindow() int {
	return int(C.GetNSIMStepWindow(g.zimtohrli))
}

// SetNSIMStepWindow sets the window in perceptual_sample_rate time steps when compting the NSIM.
func (g *Goohrli) SetNSIMStepWindow(s int) {
	C.SetNSIMStepWindow(g.zimtohrli, C.int(s))
}

// GetNSIMChannelWindow returns the window in channels when computing the NSIM.
func (g *Goohrli) GetNSIMChannelWindow() int {
	return int(C.GetNSIMChannelWindow(g.zimtohrli))
}

// SetNSIMChannelWindow sets the window in channels when computing the NSIM.
func (g *Goohrli) SetNSIMChannelWindow(s int) {
	C.SetNSIMChannelWindow(g.zimtohrli, C.int(s))
}

// GetPerceptualSampleRate returns the perceptual sample rate used, corresponding to human hearing sensitivity to differences in timing.
func (g *Goohrli) GetPerceptualSampleRate() float64 {
	return float64(C.GetPerceptualSampleRate(g.zimtohrli))
}

// SetPerceptualSampleRate sets the perceptual sample rate used.
func (g *Goohrli) SetPerceptualSampleRate(f float64) {
	C.SetPerceptualSampleRate(g.zimtohrli, C.float(f))
}

// ViSQOL is a Go wrapper around zimtohrli::ViSQOL.
type ViSQOL struct {
	visqol C.ViSQOL
}

// NewViSQOL returns a new Gosqol.
func NewViSQOL() *ViSQOL {
	result := &ViSQOL{
		visqol: C.CreateViSQOL(),
	}
	runtime.SetFinalizer(result, func(g *ViSQOL) {
		C.FreeViSQOL(g.visqol)
	})
	return result
}

// MOS returns the ViSQOL mean opinion score of the degraded samples comapred to the reference samples.
func (v *ViSQOL) MOS(sampleRate float64, reference []float32, degraded []float32) (float64, error) {
	result := C.MOS(v.visqol, C.float(sampleRate), (*C.float)(&reference[0]), C.int(len(reference)), (*C.float)(&degraded[0]), C.int(len(degraded)))
	if result.Status != 0 {
		return 0, fmt.Errorf("calling ViSQOL returned status %v", result.Status)
	}
	return float64(result.MOS), nil
}

// AudioMOS returns the ViSQOL mean opinion score of the degraded audio compared to the reference audio.
func (v *ViSQOL) AudioMOS(reference, degraded *audio.Audio) (float64, error) {
	sumOfSquares := 0.0
	if reference.Rate != degraded.Rate {
		return 0, fmt.Errorf("the audio files don't have the same sample rate: %v, %v", reference.Rate, degraded.Rate)
	}
	if len(reference.Samples) != len(degraded.Samples) {
		return 0, fmt.Errorf("the audio files don't have the same number of channels: %v, %v", len(reference.Samples), len(degraded.Samples))
	}
	for channelIndex := range reference.Samples {
		mos, err := v.MOS(reference.Rate, reference.Samples[channelIndex], degraded.Samples[channelIndex])
		if err != nil {
			return 0, err
		}
		sumOfSquares += mos * mos
	}
	return math.Sqrt(sumOfSquares / float64(len(reference.Samples))), nil
}
