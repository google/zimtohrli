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

// compare is a Go version of compare.cc.
package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/google/zimtohrli/go/goohrli"
	"github.com/youpy/go-wav"
)

func readWAV(path string) ([][]float32, float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	r := wav.NewReader(f)
	format, err := r.Format()
	if err != nil {
		return nil, 0, err
	}
	samples := []wav.Sample{}
	var buf []wav.Sample
	for buf, err = r.ReadSamples(32768); err == nil; buf, err = r.ReadSamples(32768) {
		samples = append(samples, buf...)
	}
	if err != io.EOF {
		return nil, 0, err
	}
	result := make([][]float32, format.NumChannels)
	for _, sample := range samples {
		for channelIndex := 0; channelIndex < int(format.NumChannels); channelIndex++ {
			result[channelIndex] = append(result[channelIndex], float32(sample.Values[channelIndex])/float32(int(1)<<(format.BitsPerSample-1)))
		}
	}
	return result, float64(format.SampleRate), nil
}

func main() {
	pathA := flag.String("path_a", "", "Path to WAV file with signal A.")
	pathB := flag.String("path_b", "", "Path to WAV file with signal B.")
	frequencyResolution := flag.Float64("frequency_resolution", 1.0, "Band width of smallest filter, i.e. expected frequency resolution of human hearing.")
	flag.Parse()

	if *pathA == "" || *pathB == "" {
		flag.Usage()
		os.Exit(1)
	}

	signalA, sampleRateA, err := readWAV(*pathA)
	if err != nil {
		log.Panic(err)
	}
	signalB, sampleRateB, err := readWAV(*pathB)
	if err != nil {
		log.Panic(err)
	}

	if sampleRateA != sampleRateB {
		log.Panic(fmt.Errorf("sample rate of %q is %v, and sample rate of %q is %v", *pathA, sampleRateA, *pathB, sampleRateB))
	}

	if len(signalA) != len(signalB) {
		log.Panic(fmt.Errorf("%q has %v channels, and %q has %v channels", *pathA, len(signalA), *pathB, len(signalB)))
	}

	g := goohrli.New(sampleRateA, *frequencyResolution)
	for channelIndex := 0; channelIndex < len(signalA); channelIndex++ {
		fmt.Println(g.Distance(signalA[channelIndex], signalB[channelIndex]))
	}
}
