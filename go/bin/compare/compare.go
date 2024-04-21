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
	"log"
	"os"

	"github.com/google/zimtohrli/go/aio"
	"github.com/google/zimtohrli/go/goohrli"
)

func main() {
	pathA := flag.String("path_a", "", "Path to ffmpeg-decodable file with signal A.")
	pathB := flag.String("path_b", "", "Path to ffmpeg-decodable file with signal B.")
	frequencyResolution := flag.Float64("frequency_resolution", 5.0, "Band width of smallest filter, i.e. expected frequency resolution of human hearing.")
	flag.Parse()

	if *pathA == "" || *pathB == "" {
		flag.Usage()
		os.Exit(1)
	}

	signalA, err := aio.Load(*pathA)
	if err != nil {
		log.Panic(err)
	}
	signalB, err := aio.Load(*pathB)
	if err != nil {
		log.Panic(err)
	}

	if signalA.Rate != signalB.Rate {
		log.Panic(fmt.Errorf("sample rate of %q is %v, and sample rate of %q is %v", *pathA, signalA.Rate, *pathB, signalB.Rate))
	}

	if len(signalA.Samples) != len(signalB.Samples) {
		log.Panic(fmt.Errorf("%q has %v channels, and %q has %v channels", *pathA, len(signalA.Samples), *pathB, len(signalB.Samples)))
	}

	g := goohrli.New(signalA.Rate, *frequencyResolution)
	for channelIndex := range signalA.Samples {
		measurement := goohrli.Measure(signalA.Samples[channelIndex])
		goohrli.NormalizeAmplitude(measurement.MaxAbsAmplitude, signalB.Samples[channelIndex])
		fmt.Println(g.Distance(signalA.Samples[channelIndex], signalB.Samples[channelIndex]))
	}
}
