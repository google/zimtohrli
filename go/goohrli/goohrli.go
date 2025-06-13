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
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"runtime"
	"time"

	"github.com/google/zimtohrli/go/audio"
)

// MOSFromZimtohrli returns an approximate mean opinion score for a given zimtohrli distance.
func MOSFromZimtohrli(zimtohrliDistance float64) float64 {
	return float64(C.MOSFromZimtohrli(C.float(zimtohrliDistance)))
}

// Goohrli is a Go wrapper around zimtohrli::Zimtohrli.
type Goohrli struct {
	zimtohrli C.Zimtohrli
}

// New returns a new Goohrli for the given parameters.
func New(params Parameters) *Goohrli {
	result := &Goohrli{
		zimtohrli: C.CreateZimtohrli(cFromGoParameters(params)),
	}
	runtime.SetFinalizer(result, func(g *Goohrli) {
		C.FreeZimtohrli(g.zimtohrli)
	})
	return result
}

// Duration wraps a time.Duration to provide specialized JSON marshal/unmarshal methods.
type Duration struct {
	time.Duration
}

// MarshalJSON implements json.Marshaler.
func (d Duration) MarshalJSON() ([]byte, error) {
	return json.Marshal(d.Duration.String())
}

// UnmarshalJSON implements json.Unmarshaler.
func (d *Duration) UnmarshalJSON(b []byte) error {
	timeString := ""
	if err := json.Unmarshal(b, &timeString); err != nil {
		return err
	}
	timeDuration, err := time.ParseDuration(timeString)
	if err != nil {
		return err
	}
	d.Duration = timeDuration
	return nil
}

const (
	numLoudnessAFParams = 10
	numLoudnessLUParams = 16
	numLoudnessTFParams = 13
)

// Parameters contains the parameters used by a Goohrli instance.
type Parameters struct {
	PerceptualSampleRate float64
	FullScaleSineDB      float64
	NSIMStepWindow       int
	NSIMChannelWindow    int
}

var durationType = reflect.TypeOf(time.Second)

func SampleRate() float64 {
	return float64(C.SampleRate())
}

// Update assumes the argument is JSON and updates the parameters with the fields present in the provided JSON object.
func (p *Parameters) Update(b []byte) error {
	updateMap := map[string]any{}
	if err := json.Unmarshal(b, &updateMap); err != nil {
		return err
	}
	val := reflect.ValueOf(p).Elem()
	for k, v := range updateMap {
		fieldVal := val.FieldByName(k)
		if fieldVal.IsZero() {
			return fmt.Errorf("provided unknown field %q", k)
		}
		switch fieldVal.Kind() {
		case reflect.Float64:
			fieldVal.SetFloat(v.(float64))
		case reflect.Bool:
			fieldVal.SetBool(v.(bool))
		case reflect.Int:
			fieldVal.SetInt(int64(v.(float64)))
		case reflect.Array:
			b, err := json.Marshal(v)
			if err != nil {
				return err
			}
			aryPtr := reflect.New(reflect.ArrayOf(fieldVal.Type().Len(), fieldVal.Type().Elem()))
			if err := json.Unmarshal(b, aryPtr.Interface()); err != nil {
				return err
			}
			fieldVal.Set(aryPtr.Elem())
		default:
			if fieldVal.Type() == durationType {
				d, err := time.ParseDuration(v.(string))
				if err != nil {
					return fmt.Errorf("unable to parse duration field %q", v)
				}
				fieldVal.Set(reflect.ValueOf(d))
			}
		}
	}
	return nil
}

func cFromGoParameters(params Parameters) C.ZimtohrliParameters {
	var cParams C.ZimtohrliParameters
	cParams.PerceptualSampleRate = C.float(params.PerceptualSampleRate)
	cParams.FullScaleSineDB = C.float(params.FullScaleSineDB)
	cParams.NSIMStepWindow = C.int(params.NSIMStepWindow)
	cParams.NSIMChannelWindow = C.int(params.NSIMChannelWindow)
	return cParams
}

func goFromCParameters(cParams C.ZimtohrliParameters) Parameters {
	result := Parameters{
		PerceptualSampleRate: float64(cParams.PerceptualSampleRate),
		FullScaleSineDB:      float64(cParams.FullScaleSineDB),
		NSIMStepWindow:       int(cParams.NSIMStepWindow),
		NSIMChannelWindow:    int(cParams.NSIMChannelWindow),
	}
	return result
}

// DefaultParameters returns the default Zimtohrli parameters.
func DefaultParameters() Parameters {
	return goFromCParameters(C.DefaultZimtohrliParameters())
}

// Parameters returns the parameters controlling the behavior of this instance.
func (g *Goohrli) Parameters() Parameters {
	return goFromCParameters(C.GetZimtohrliParameters(g.zimtohrli))
}

// Set updates the parameters controlling the behavior of this instance.
//
// SampleRate, FrequencyResolution, and Filter*-parameters can't be updated and will be ignored in this method.
func (g *Goohrli) Set(params Parameters) {
	C.SetZimtohrliParameters(g.zimtohrli, cFromGoParameters(params))
}

func (g *Goohrli) String() string {
	return fmt.Sprintf("%+v", g.Parameters())
}

// NormalizedAudioDistance returns the distance between the audio files after normalizing their amplitudes for the same max amplitude.
func (g *Goohrli) NormalizedAudioDistance(audioA, audioB *audio.Audio) (float64, error) {
	sumOfSquares := 0.0
	if int(audioA.Rate) != int(C.SampleRate()) || int(audioB.Rate) != int(C.SampleRate()) {
		return 0, fmt.Errorf("one of the audio files doesn't have the expected sample rate %v: %v, %v", C.SampleRate(), audioA.Rate, audioB.Rate)
	}
	if len(audioA.Samples) != len(audioB.Samples) {
		return 0, fmt.Errorf("the audio files don't have the same number of channels: %v, %v", len(audioA.Samples), len(audioB.Samples))
	}
	if len(audioA.Samples) == 0 {
		return 0, fmt.Errorf("the audio files don't have any channels")
	}
	for channelIndex := range audioA.Samples {
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

// Spec is a Go wrapper around zimthrli::Analysis.
type Spec struct {
	spec C.GoSpectrogram
}

// Spectrogram returns a spectrogram of the signal.
func (g *Goohrli) Analyze(signal []float32) *Spec {
	result := &Spec{
		spec: C.Analyze(g.zimtohrli, (*C.float)(&signal[0]), C.int(len(signal))),
	}
	runtime.SetFinalizer(result, func(a *Spec) {
		C.FreeSpec(a.spec)
	})
	return result
}

// SpecDistance returns the Zimtohrli distance between two analyses.
func (g *Goohrli) SpecDistance(specA *Spec, specB *Spec) float32 {
	return float32(C.Distance(g.zimtohrli, specA.spec, specB.spec))
}

// Distance returns the Zimtohrli distance between two signals.
func (g *Goohrli) Distance(signalA []float32, signalB []float32) float64 {
	specA := C.Analyze(g.zimtohrli, (*C.float)(&signalA[0]), C.int(len(signalA)))
	defer C.FreeSpec(specA)
	specB := C.Analyze(g.zimtohrli, (*C.float)(&signalB[0]), C.int(len(signalB)))
	defer C.FreeSpec(specB)
	return float64(C.Distance(g.zimtohrli, specA, specB))
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
