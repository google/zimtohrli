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
	"reflect"
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
	details := flag.String("details", "", "Glob to directories with databases to show the details of.")
	calculate := flag.String("calculate", "", "Glob to directories with databases to calculate metrics for.")
	force := flag.Bool("force", false, "Whether to recalculate scores that already exist.")
	calculateZimtohrli := flag.Bool("calculate_zimtohrli", false, "Whether to calculate Zimtohrli scores.")
	zimtohrliScoreType := flag.String("zimtohrli_score_type", string(data.Zimtohrli), "Score type name to use when storing Zimtohrli scores in a dataset.")
	calculateViSQOL := flag.Bool("calculate_visqol", false, "Whether to calculate ViSQOL scores.")
	calculatePipeMetric := flag.String("calculate_pipe", "", "Path to a binary that serves metrics via stdin/stdout pipe. Install some of the via 'install_python_metrics.py'.")
	zimtohrliParameters := goohrli.DefaultParameters(48000)
	b, err := json.Marshal(zimtohrliParameters)
	if err != nil {
		log.Panic(err)
	}
	zimtohrliParametersJSON := flag.String("zimtohrli_parameters", string(b), "Zimtohrli model parameters. Sample rate will be set to the sample rate of the measured audio files.")
	correlate := flag.String("correlate", "", "Glob to directories with databases to correlate scores for.")
	leaderboard := flag.String("leaderboard", "", "Glob to directories with databases to compute leaderboard for.")
	report := flag.String("report", "", "Glob to directories with databases to generate a report for.")
	accuracy := flag.String("accuracy", "", "Glob to directories with databases to provide JND accuracy for.")
	optimize := flag.String("optimize", "", "Glob to directories with databases to optimize for.")
	optimizeLogfile := flag.String("optimize_logfile", "", "File to write optimization events to.")
	optimizeStartStep := flag.Float64("optimize_start_step", 1, "Start step for the simulated annealing.")
	optimizeNumSteps := flag.Float64("optimize_num_steps", 1000, "Number of steps for the simulated annealing.")
	workers := flag.Int("workers", runtime.NumCPU(), "Number of concurrent workers for tasks.")
	failFast := flag.Bool("fail_fast", false, "Whether to panic immediately on any error.")
	flag.Parse()

	if *details == "" && *calculate == "" && *correlate == "" && *accuracy == "" && *leaderboard == "" && *report == "" && *optimize == "" {
		flag.Usage()
		os.Exit(1)
	}

	if err := zimtohrliParameters.Update([]byte(*zimtohrliParametersJSON)); err != nil {
		log.Panic(err)
	}

	if *optimize != "" {
		bundles, err := data.OpenBundles(*optimize)
		if err != nil {
			log.Fatal(err)
		}
		optimizeLog := func(ev data.OptimizationEvent) {}
		if *optimizeLogfile != "" {
			f, err := os.OpenFile(*optimizeLogfile, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0644)
			if err != nil {
				log.Fatal(err)
			}
			optimizeLog = func(ev data.OptimizationEvent) {
				b, _ := json.Marshal(ev)
				f.WriteString(string(b) + "\n")
				f.Sync()
			}
		}
		err = bundles.Optimize(*optimizeStartStep, *optimizeNumSteps, optimizeLog)
		if err != nil {
			log.Fatal(err)
		}
	}

	if *calculate != "" {
		studies, err := data.OpenStudies(*calculate)
		if err != nil {
			log.Fatal(err)
		}
		defer studies.Close()
		for _, study := range studies {
			measurements := map[data.ScoreType]data.Measurement{}
			if *calculateZimtohrli {
				if !reflect.DeepEqual(zimtohrliParameters, goohrli.DefaultParameters(zimtohrliParameters.SampleRate)) {
					log.Printf("Using %+v", zimtohrliParameters)
				}
				zimtohrliParameters.SampleRate = sampleRate
				z := goohrli.New(zimtohrliParameters)
				measurements[data.ScoreType(*zimtohrliScoreType)] = z.NormalizedAudioDistance
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
			bundle, err := study.ToBundle()
			if err != nil {
				log.Fatal(err)
			}
			log.Printf("*** Calculating %+v (force=%v) for %v", sortedTypes, *force, bundle.Dir)
			bar := progress.New("Calculating")
			pool := &worker.Pool[any]{
				Workers:  *workers,
				OnChange: bar.Update,
				FailFast: *failFast,
			}
			if err := bundle.Calculate(measurements, pool, *force); err != nil {
				log.Printf("%#v", err)
				log.Fatal(err)
			}
			if err := study.Put(bundle.References); err != nil {
				log.Fatal(err)
			}
			bar.Finish()
		}
	}

	if *correlate != "" {
		bundles, err := data.OpenBundles(*correlate)
		if err != nil {
			log.Fatal(err)
		}
		for _, bundle := range bundles {
			if bundle.IsJND() {
				fmt.Printf("Not computing correlation for JND dataset %q\n\n", bundle.Dir)
			} else {
				corrTable, err := bundle.Correlate()
				if err != nil {
					log.Fatal(err)
				}
				fmt.Printf("## %v\n\n", bundle.Dir)
				fmt.Println(corrTable)
			}
		}
	}

	if *accuracy != "" {
		bundles, err := data.OpenBundles(*accuracy)
		if err != nil {
			log.Fatal(err)
		}
		for _, bundle := range bundles {
			if bundle.IsJND() {
				accuracy, err := bundle.JNDAccuracy()
				if err != nil {
					log.Fatal(err)
				}
				fmt.Printf("## %v\n", bundle.Dir)
				fmt.Println(accuracy)
			} else {
				fmt.Printf("Not computing accuracy for non-JND dataset %q\n\n", bundle.Dir)
			}
		}
	}

	if *report != "" {
		bundles, err := data.OpenBundles(*report)
		if err != nil {
			log.Fatal(err)
		}
		report, err := bundles.Report()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(report)
	}

	if *leaderboard != "" {
		bundles, err := data.OpenBundles(*leaderboard)
		if err != nil {
			log.Fatal(err)
		}
		board, err := bundles.Leaderboard()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(board)
	}

	if *details != "" {
		bundles, err := data.OpenBundles(*details)
		if err != nil {
			log.Fatal(err)
		}
		b, err := json.MarshalIndent(bundles, "", "  ")
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%s\n", b)
	}
}
