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

// score handles listening test datasets.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"sort"

	"github.com/google/zimtohrli/go/data"
	"github.com/google/zimtohrli/go/goohrli"
	"github.com/google/zimtohrli/go/pipe"
	"github.com/google/zimtohrli/go/progress"
	"github.com/google/zimtohrli/go/worker"
)

const (
	sampleRate = 48000
)

func main() {
	details := flag.String("details", "", "Path to database directory with a study to show the details from.")
	calculate := flag.String("calculate", "", "Path to a database directory with a study to calculate metrics for.")
	force := flag.Bool("force", false, "Whether to recalculate scores that already exist.")
	calculateZimtohrli := flag.Bool("calculate_zimtohrli", false, "Whether to calculate Zimtohrli scores.")
	calculateViSQOL := flag.Bool("calculate_visqol", false, "Whether to calculate ViSQOL scores.")
	calculatePipeMetric := flag.String("calculate_pipe", "", "Path to a binary that serves metrics via stdin/stdout pipe. Install some of the via 'install_python_metrics.py'.")
	zimtohrliFrequencyResolution := flag.Float64("zimtohrli_frequency_resolution", goohrli.DefaultFrequencyResolution(), "Smallest bandwidth of the Zimtohrli filterbank.")
	zimtohrliPerceptualSampleRate := flag.Float64("zimtohrli_perceptual_sample_rate", goohrli.DefaultPerceptualSampleRate(), "Sample rate of the Zimtohrli spectrograms.")
	correlate := flag.String("correlate", "", "Path to a database directory with a study to correlate scores for.")
	leaderboard := flag.String("leaderboard", "", "Glob to directories with databases to compute leaderboard for.")
	accuracy := flag.String("accuracy", "", "Path to a database directory with a study to provide JND accuracy for.")
	workers := flag.Int("workers", runtime.NumCPU(), "Number of concurrent workers for tasks.")
	failFast := flag.Bool("fail_fast", false, "Whether to panic immediately on any error.")
	flag.Parse()

	if *details == "" && *calculate == "" && *correlate == "" && *accuracy == "" && *leaderboard == "" {
		flag.Usage()
		os.Exit(1)
	}

	if *leaderboard != "" {
		databases, err := filepath.Glob(*leaderboard)
		if err != nil {
			log.Fatal(err)
		}
		studies := make(data.Studies, len(databases))
		for index, path := range databases {
			if studies[index], err = data.OpenStudy(path); err != nil {
				log.Fatal(err)
			}
		}
		board, err := studies.Leaderboard()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(board)
	}

	if *details != "" {
		study, err := data.OpenStudy(*details)
		if err != nil {
			log.Fatal(err)
		}
		defer study.Close()
		refs := []*data.Reference{}
		if err := study.ViewEachReference(func(ref *data.Reference) error {
			refs = append(refs, ref)
			return nil
		}); err != nil {
			log.Fatal(err)
		}
		b, err := json.MarshalIndent(refs, "", "  ")
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%s\n", b)
	}

	if *calculate != "" {
		study, err := data.OpenStudy(*calculate)
		if err != nil {
			log.Fatal(err)
		}
		defer study.Close()
		bar := progress.New("Calculating")
		pool := &worker.Pool[any]{
			Workers:  *workers,
			OnChange: bar.Update,
			FailFast: *failFast,
		}
		measurements := map[data.ScoreType]data.Measurement{}
		if *calculateZimtohrli {
			z := goohrli.New(sampleRate, *zimtohrliFrequencyResolution)
			z.SetPerceptualSampleRate(float32(*zimtohrliPerceptualSampleRate))
			measurements[data.Zimtohrli] = z.NormalizedAudioDistance
		}
		if *calculateViSQOL {
			v := goohrli.NewViSQOL()
			measurements[data.ViSQOL] = v.AudioMOS
		}
		if *calculatePipeMetric != "" {
			pool, err := pipe.NewMeterPool(*calculatePipeMetric)
			if err != nil {
				log.Fatal(err)
			}
			defer pool.Close()
			measurements[pool.ScoreType] = pool.Measure
		}
		if len(measurements) == 0 {
			log.Print("No metrics to calculate, provide one of the -calculate_XXX flags!")
			os.Exit(2)
		}
		sortedTypes := sort.StringSlice{}
		for scoreType := range measurements {
			sortedTypes = append(sortedTypes, string(scoreType))
		}
		sort.Sort(sortedTypes)
		log.Printf("*** Calculating %+v (force=%v)", sortedTypes, *force)
		if err := study.Calculate(measurements, pool, *force); err != nil {
			log.Fatal(err)
		}
		bar.Finish()
	}

	if *correlate != "" {
		study, err := data.OpenStudy(*correlate)
		if err != nil {
			log.Fatal(err)
		}
		defer study.Close()
		corrTable, err := study.Correlate()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(corrTable)
	}

	if *accuracy != "" {
		study, err := data.OpenStudy(*accuracy)
		if err != nil {
			log.Fatal(err)
		}
		defer study.Close()
		accuracy, err := study.Accuracy()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(accuracy)
	}
}
