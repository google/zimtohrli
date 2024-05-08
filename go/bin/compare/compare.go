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
	"github.com/google/zimtohrli/go/pipe"
)

func main() {
	pathA := flag.String("path_a", "", "Path to ffmpeg-decodable file with signal A.")
	pathB := flag.String("path_b", "", "Path to ffmpeg-decodable file with signal B.")
	visqol := flag.Bool("visqol", false, "Whether to measure using ViSQOL.")
	pipeMetric := flag.String("pipe_metric", "", "Path to a binary that serves metrics via stdin/stdout pipe. Install some of the via 'install_python_metrics.py'.")
	zimtohrli := flag.Bool("zimtohrli", true, "Whether to measure using Zimtohrli.")
	outputZimtohrliDistance := flag.Bool("output_zimtohrli_distance", false, "Whether to output the raw Zimtohrli distance instead of a mapped mean opinion score.")
	zimtohrliPerceptualSampleRate := flag.Float64("zimtohrli_perceptual_sample_rate", goohrli.DefaultPerceptualSampleRate(), "Sample rate of Zimtohrli spectrograms.")
	zimtohrliFrequencyResolution := flag.Float64("zimtohrli_frequency_resolution", goohrli.DefaultFrequencyResolution(), "Smallest bandwidth of Zimtohrli spectrograms.")
	perChannel := flag.Bool("per_channel", false, "Whether to output the produced metric per channel instead of a single value for all channels.")
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

	if *pipeMetric != "" {
		metric, err := pipe.StartMetric(*pipeMetric)
		if err != nil {
			log.Panic(err)
		}
		defer metric.Close()
		scoreType, err := metric.ScoreType()
		if err != nil {
			log.Panic(err)
		}
		score, err := metric.Measure(signalA, signalB)
		if err != nil {
			log.Panic(err)
		}
		fmt.Printf("%v=%v\n", scoreType, score)
	}

	if *visqol {
		v := goohrli.NewViSQOL()
		if *perChannel {
			for channelIndex := range signalA.Samples {
				mos, err := v.MOS(signalA.Rate, signalA.Samples[channelIndex], signalB.Samples[channelIndex])
				if err != nil {
					log.Panic(err)
				}
				fmt.Printf("ViSQOL#%v=%v\n", channelIndex, mos)
			}
		} else {
			mos, err := v.AudioMOS(signalA, signalB)
			if err != nil {
				log.Panic(err)
			}
			fmt.Printf("ViSQOL=%v\n", mos)
		}
	}

	if *zimtohrli {
		getMetric := func(f float64) float64 {
			if *outputZimtohrliDistance {
				return f
			}
			return goohrli.MOSFromZimtohrli(f)
		}

		g := goohrli.New(signalA.Rate, *zimtohrliFrequencyResolution)
		g.SetPerceptualSampleRate(*zimtohrliPerceptualSampleRate)
		if *perChannel {
			for channelIndex := range signalA.Samples {
				measurement := goohrli.Measure(signalA.Samples[channelIndex])
				goohrli.NormalizeAmplitude(measurement.MaxAbsAmplitude, signalB.Samples[channelIndex])
				fmt.Printf("Zimtohrli#%v=%v\n", channelIndex, getMetric(g.Distance(signalA.Samples[channelIndex], signalB.Samples[channelIndex])))
			}
		} else {
			dist, err := g.NormalizedAudioDistance(signalA, signalB)
			if err != nil {
				log.Panic(err)
			}
			fmt.Printf("Zimtohrli=%v\n", getMetric(dist))
		}
	}
}
