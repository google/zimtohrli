// Copyright 2025 The Zimtohrli Authors. All Rights Reserved.
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
	"encoding/json"
	"log"
	"math"
	"reflect"
	"testing"
	"time"
)

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
			zimtDistance: 0.001,
			wantMOS:      4.8008866310119629,
		},
		{
			zimtDistance: 0.01,
			wantMOS:      3.4005415439605713,
		},
		{
			zimtDistance: 0.02,
			wantMOS:      2.4406499862670898,
		},
		{
			zimtDistance: 0.03,
			wantMOS:      1.8645849227905273,
		},
		{
			zimtDistance: 0.04,
			wantMOS:      1.5188679695129395,
		},
	} {
		if mos := MOSFromZimtohrli(tc.zimtDistance); math.Abs(mos-tc.wantMOS) > 1e-2 {
			t.Errorf("MOSFromZimtohrli(%v) = %v, want %v", tc.zimtDistance, mos, tc.wantMOS)
		}
	}
}

func TestParams(t *testing.T) {
	g := New(DefaultParameters())

	params := g.Parameters()
	params.FullScaleSineDB *= 0.5
	params.NSIMChannelWindow *= 2
	params.NSIMStepWindow *= 2
	params.PerceptualSampleRate *= 0.5

	g.Set(params)
	newParams := g.Parameters()
	if !reflect.DeepEqual(newParams, params) {
		t.Errorf("Expected updated parameters to be %+v, got %+v", params, newParams)
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
			distance: 0,
		},
		{
			freqA:    5000,
			freqB:    10000,
			distance: 0.02702277898788452,
		},
	} {
		params := DefaultParameters()
		g := New(params)
		soundA := make([]float32, int(SampleRate()))
		for index := 0; index < len(soundA); index++ {
			soundA[index] = float32(math.Sin(2 * math.Pi * tc.freqA * float64(index) / SampleRate()))
		}
		analysisA := g.Analyze(soundA)
		soundB := make([]float32, int(SampleRate()))
		for index := 0; index < len(soundB); index++ {
			soundB[index] = float32(math.Sin(2 * math.Pi * tc.freqB * float64(index) / SampleRate()))
		}
		analysisB := g.Analyze(soundB)
		analysisDistance := float64(g.SpecDistance(analysisA, analysisB))
		if d := rdiff(analysisDistance, tc.distance); d > 0.1 {
			t.Errorf("Distance = %v, want %v", analysisDistance, tc.distance)
		}
		distance := float64(g.Distance(soundA, soundB))
		if d := rdiff(distance, tc.distance); d > 0.1 {
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

var goohrliDurationType = reflect.TypeOf(Duration{})

func populate(s any) {
	counter := 1
	val := reflect.ValueOf(s).Elem()
	typ := val.Type()
	for _, field := range reflect.VisibleFields(typ) {
		switch field.Type.Kind() {
		case reflect.Float64:
			val.FieldByIndex(field.Index).SetFloat(float64(counter))
		case reflect.Int:
			val.FieldByIndex(field.Index).SetInt(int64(counter & 0xffff))
		case reflect.Bool:
			val.FieldByIndex(field.Index).SetBool(true)
		case reflect.Array:
			if field.Type.Elem().Kind() == reflect.Float64 {
				for i := 0; i < field.Type.Len(); i++ {
					val.FieldByIndex(field.Index).Index(i).SetFloat(float64(counter))
					counter++
				}
			} else {
				log.Panicf("Unsupported array type %v", field.Type.Elem().Kind())
			}
		default:
			if field.Type == goohrliDurationType {
				val.FieldByIndex(field.Index).Set(reflect.ValueOf(Duration{Duration: time.Duration(counter) * time.Minute}))
			} else {
				log.Panicf("Unsupported field %v", field)
			}
		}
		counter++
	}
}

func rdiff(a, b float64) float64 {
	return math.Abs(float64(a-b) / (0.5 * (a + b)))
}

func checkNear(a, b any, rtol float64, t *testing.T) {
	t.Helper()
	aVal := reflect.ValueOf(a)
	bVal := reflect.ValueOf(b)
	if aVal.Type() != bVal.Type() {
		t.Fatalf("%v and %v not same type", a, b)
	}
	for _, field := range reflect.VisibleFields(aVal.Type()) {
		switch field.Type.Kind() {
		case reflect.Float64:
			aFloat := aVal.FieldByIndex(field.Index).Float()
			bFloat := bVal.FieldByIndex(field.Index).Float()
			if d := rdiff(aFloat, bFloat); d > rtol {
				t.Errorf("%v: %v is more than %v off from %v", field, aFloat, rtol, bFloat)
			}
		case reflect.Int:
			aInt := aVal.FieldByIndex(field.Index).Int()
			bInt := bVal.FieldByIndex(field.Index).Int()
			if aInt != bInt {
				t.Errorf("%v: %v != %v", field, aInt, bInt)
			}
		case reflect.Bool:
			aBool := aVal.FieldByIndex(field.Index).Bool()
			bBool := bVal.FieldByIndex(field.Index).Bool()
			if aBool != bBool {
				t.Errorf("%v: %v != %v", field, aBool, bBool)
			}
		case reflect.Array:
			if field.Type.Elem().Kind() == reflect.Float64 {
				for i := 0; i < aVal.FieldByIndex(field.Index).Len(); i++ {
					aFloat := aVal.FieldByIndex(field.Index).Index(i).Float()
					bFloat := bVal.FieldByIndex(field.Index).Index(i).Float()
					if d := rdiff(aFloat, bFloat); d > rtol {
						t.Errorf("%v[%v]: %v is more than %v off from %v", field, i, aFloat, rtol, bFloat)
					}
				}
			} else {
				log.Panicf("Unsupported array type %v", field.Type.Elem())
			}
		default:
			if field.Type == goohrliDurationType {
				aDur := aVal.FieldByIndex(field.Index).Interface().(Duration).Duration
				bDur := bVal.FieldByIndex(field.Index).Interface().(Duration).Duration
				if d := rdiff(float64(aDur), float64(bDur)); d > rtol {
					t.Errorf("%v: %v is more than %v off from %v", field, aDur, rtol, bDur)
				}
			} else {
				log.Panicf("Unsupported field %v", field)
			}
		}
	}
}

func TestParamConversion(t *testing.T) {
	params := Parameters{}
	populate(&params)
	cParams := cFromGoParameters(params)
	reconvertedParams := goFromCParameters(cParams)
	checkNear(reconvertedParams, params, 1e-6, t)
}

func TestParamUpdate(t *testing.T) {
	params := DefaultParameters()
	js, err := json.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	m := map[string]any{}
	if err := json.Unmarshal(js, &m); err != nil {
		t.Fatal(err)
	}
	for k, v := range m {
		js, err = json.Marshal(map[string]any{k: v})
		if err != nil {
			t.Fatal(err)
		}
		if err := params.Update(js); err != nil {
			t.Error(err)
		}
	}
}
