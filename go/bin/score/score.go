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
	zimtohrliNSIMStepWindow := flag.Int("zimtohrli_nsim_step_window", goohrli.DefaultNSIMStepWindow(), "Window size in perceptual sample rate steps when computing NSIM.")
	zimtohrliNSIMChannelWindow := flag.Int("zimtohrli_nsim_channel_window", goohrli.DefaultNSIMChannelWindow(), "Window size in channels when computing NSIM.")
	correlate := flag.String("correlate", "", "Path to a database directory with a study to correlate scores for.")
	leaderboard := flag.String("leaderboard", "", "Glob to directories with databases to compute leaderboard for.")
	report := flag.String("report", "", "Glob to directories with databases to generate a report for.")
	accuracy := flag.String("accuracy", "", "Path to a database directory with a study to provide JND accuracy for.")
	optimize := flag.String("optimize", "", "Glob to directories with databases to optimize for.")
	optimizeLogfile := flag.String("optimize_logfile", "", "File to write optimization events to.")
	optimizeStartStep := flag.Float64("optimize_start_step", 1, "Start step for the simulated annealing.")
	optimizeNumSteps := flag.Float64("optimize_num_steps", 500, "Number of steps for the simulated annealing.")
	workers := flag.Int("workers", runtime.NumCPU(), "Number of concurrent workers for tasks.")
	failFast := flag.Bool("fail_fast", false, "Whether to panic immediately on any error.")
	flag.Parse()

	if *details == "" && *calculate == "" && *correlate == "" && *accuracy == "" && *leaderboard == "" && *report == "" && *optimize == "" {
		flag.Usage()
		os.Exit(1)
	}

	if *optimize != "" {
		studies, err := data.OpenStudies(*optimize)
		if err != nil {
			log.Fatal(err)
		}
		defer studies.Close()
		if len(studies) == 0 {
			log.Fatal(fmt.Errorf("no studies found in %q", *optimize))
		}
		bundles, err := studies.ToBundles()
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

	if *report != "" {
		studies, err := data.OpenStudies(*report)
		if err != nil {
			log.Fatal(err)
		}
		defer studies.Close()
		if len(studies) == 0 {
			log.Fatal(fmt.Errorf("no studies found in %q", *report))
		}
		bundles, err := studies.ToBundles()
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
		studies, err := data.OpenStudies(*leaderboard)
		if err != nil {
			log.Fatal(err)
		}
		defer studies.Close()
		if len(studies) == 0 {
			log.Fatal(fmt.Errorf("no studies found in %q", *leaderboard))
		}
		bundles, err := studies.ToBundles()
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
		study, err := data.OpenStudy(*details)
		if err != nil {
			log.Fatal(err)
		}
		defer study.Close()
		bundle, err := study.ToBundle()
		if err != nil {
			log.Fatal(err)
		}
		b, err := json.MarshalIndent(bundle, "", "  ")
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
			z.SetPerceptualSampleRate(*zimtohrliPerceptualSampleRate)
			z.SetNSIMStepWindow(*zimtohrliNSIMStepWindow)
			z.SetNSIMChannelWindow(*zimtohrliNSIMChannelWindow)
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
		log.Printf("*** Calculating %+v for %q (force=%v)", sortedTypes, *calculate, *force)
		bundle, err := study.ToBundle()
		if err != nil {
			log.Fatal(err)
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

	if *correlate != "" {
		study, err := data.OpenStudy(*correlate)
		if err != nil {
			log.Fatal(err)
		}
		defer study.Close()
		bundle, err := study.ToBundle()
		if err != nil {
			log.Fatal(err)
		}
		corrTable, err := bundle.Correlate()
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
		bundle, err := study.ToBundle()
		if err != nil {
			log.Fatal(err)
		}
		accuracy, err := bundle.Accuracy()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(accuracy)
	}
}
