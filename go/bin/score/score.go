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
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"

	"github.com/google/zimtohrli/go/aio"
	"github.com/google/zimtohrli/go/data"
	"github.com/google/zimtohrli/go/goohrli"
	"github.com/google/zimtohrli/go/pipe"
	"github.com/google/zimtohrli/go/progress"
	"github.com/google/zimtohrli/go/worker"
)

func main() {
	details := flag.String("details", "", "Glob to directories with databases to show the details of.")
	calculate := flag.String("calculate", "", "Glob to directories with databases to calculate metrics for.")
	force := flag.Bool("force", false, "Whether to recalculate scores that already exist.")
	calculateZimtohrli := flag.Bool("calculate_zimtohrli", false, "Whether to calculate Zimtohrli scores.")
	zimtohrliScoreType := flag.String("zimtohrli_score_type", string(data.Zimtohrli), "Score type name to use when storing Zimtohrli scores in a dataset.")
	calculateViSQOL := flag.Bool("calculate_visqol", false, "Whether to calculate ViSQOL scores.")
	calculatePipeMetric := flag.String("calculate_pipe", "", "Path to a binary that serves metrics via stdin/stdout pipe. Install some of the via 'install_python_metrics.py'.")
	zimtohrliParameters := goohrli.DefaultParameters(aio.DefaultSampleRate)
	b, err := json.Marshal(zimtohrliParameters)
	if err != nil {
		log.Panic(err)
	}
	zimtohrliParametersJSON := flag.String("zimtohrli_parameters", string(b), "Zimtohrli model parameters. Sample rate will be set to the sample rate of the measured audio files.")
	correlate := flag.String("correlate", "", "Glob to directories with databases to correlate scores for.")
	leaderboard := flag.String("leaderboard", "", "Glob to directories with databases to compute leaderboard for.")
	report := flag.String("report", "", "Glob to directories with databases to generate a report for.")
	accuracy := flag.String("accuracy", "", "Glob to directories with databases to provide JND accuracy for.")
	mse := flag.String("mse", "", "Glob to directories with databases to provide mean-square-error when predicting MOS or JND for.")
	optimizedMSE := flag.String("optimized_mse", "", "Glob to directories with databases to provide mean-square-error when predicting MOS or JND for after having optimized the MOS mapping (as in `-optimize_mapping`).")
	optimize := flag.String("optimize", "", "Glob to directories with databases to optimize for.")
	optimizeLogfile := flag.String("optimize_logfile", "", "File to write optimization events to.")
	workers := flag.Int("workers", runtime.NumCPU(), "Number of concurrent workers for tasks.")
	failFast := flag.Bool("fail_fast", false, "Whether to panic immediately on any error.")
	optimizeMapping := flag.String("optimize_mapping", "", "Glob to directories with databases to optimize the MOS mapping for.")
	sample := flag.String("sample", "", "Glob to directories with databases to sample metadata and audio from.")
	sampleMinMOS := flag.Float64("sample_min_mos", 0, "Discard evaluations with lower MOS than this when sampling.")
	sampleDestination := flag.String("sample_destination", "", "Path to directory to put the sampled databases into.")
	sampleFraction := flag.Float64("sample_fraction", 1.0, "Fraction of references to copy from the source databases.")
	sampleSeed := flag.Int64("sample_seed", 0, "Seed when sampling a random fraction of references.")
	flag.Parse()

	if *details == "" && *calculate == "" && *correlate == "" && *accuracy == "" && *leaderboard == "" && *report == "" && *optimize == "" && *optimizeMapping == "" && *mse == "" && *optimizedMSE == "" && *sample == "" {
		flag.Usage()
		os.Exit(1)
	}

	if err := zimtohrliParameters.Update([]byte(*zimtohrliParametersJSON)); err != nil {
		log.Panic(err)
	}
	if zimtohrliParameters.SampleRate != aio.DefaultSampleRate {
		log.Fatalf("Zimtohrli sample rates != %v not supported by this tool, since it loads all data set audio at %vHz.", aio.DefaultSampleRate, aio.DefaultSampleRate)
	}

	if *sample != "" {
		if *sampleDestination == "" {
			log.Fatal("`-sample_destination` required for sample operation")
		}
		bundles, err := data.OpenBundles(*sample)
		if err != nil {
			log.Fatal(err)
		}
		rng := rand.New(rand.NewSource(*sampleSeed))
		for _, bundle := range bundles {
			dest, err := data.OpenStudy(filepath.Join(*sampleDestination, filepath.Base(bundle.Dir)))
			if err != nil {
				log.Fatal(err)
			}
			scaler, err := bundle.MOSScaler()
			if err != nil {
				log.Fatal(err)
			}
			func() {
				defer dest.Close()
				if *sampleFraction == 1.0 {
					bar := progress.New(fmt.Sprintf("Copying %q", filepath.Base(bundle.Dir)))
					if err := dest.Copy(bundle.Dir, bundle.References, *sampleMinMOS, scaler, bar.Update); err != nil {
						log.Fatal(err)
					}
					bar.Finish()
				} else {
					numRefs := len(bundle.References)
					numWanted := int(*sampleFraction * float64(numRefs))
					toCopy := []*data.Reference{}
					bar := progress.New(fmt.Sprintf("Copying %v of %q", *sampleFraction, filepath.Base(bundle.Dir)))
					for _, index := range rng.Perm(numRefs) {
						ref := bundle.References[index]
						if ref.HasMOSAbove(*sampleMinMOS, scaler) {
							toCopy = append(toCopy, bundle.References[index])
						}
						if len(toCopy) >= numWanted {
							break
						}
					}
					if err := dest.Copy(bundle.Dir, toCopy, *sampleMinMOS, scaler, bar.Update); err != nil {
						log.Fatal(err)
					}
					bar.Finish()
				}
			}()
		}
	}

	if *optimize != "" {
		bundles, err := data.OpenBundles(*optimize)
		if err != nil {
			log.Fatal(err)
		}
		recorder := &data.Recorder{}
		if *optimizeLogfile != "" {
			f, err := os.OpenFile(*optimizeLogfile, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0644)
			if err != nil {
				log.Fatal(err)
			}
			recorder.Output = f
		}
		if err = bundles.Optimize(recorder); err != nil {
			log.Fatal(err)
		}
	}

	if *optimizeMapping != "" {
		bundles, err := data.OpenBundles(*optimizeMapping)
		if err != nil {
			log.Fatal(err)
		}
		result, err := bundles.OptimizeMapping()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%+v\n", result)
	}

	makeZimtohrli := func() *goohrli.Goohrli {
		if !reflect.DeepEqual(zimtohrliParameters, goohrli.DefaultParameters(zimtohrliParameters.SampleRate)) {
			log.Printf("Using %+v", zimtohrliParameters)
		}
		z := goohrli.New(zimtohrliParameters)
		return z
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
				z := makeZimtohrli()
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
		corrTable, err := bundles.Correlate()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("## %v\n\n", *correlate)
		fmt.Println(corrTable)
	}

	if *accuracy != "" {
		bundles, err := data.OpenBundles(*accuracy)
		if err != nil {
			log.Fatal(err)
		}
		result, err := bundles.JNDAccuracy()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("## %v\n\n", *accuracy)
		fmt.Println(result)
	}

	if *optimizedMSE != "" {
		bundles, err := data.OpenBundles(*optimizedMSE)
		if err != nil {
			log.Fatal(err)
		}
		res, err := bundles.OptimizedZimtohrliMSE()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("## %v\n\n", *optimizedMSE)
		fmt.Printf("Error for MOS datasets is `human-MOS - Zimtohrli-predicted-MOS`. Error for JND datasets is `distance from correct side of threshold`.\n\n")
		fmt.Printf("MSE after optimizing mapping: %.15f\n\n", res)
	}

	if *mse != "" {
		bundles, err := data.OpenBundles(*mse)
		if err != nil {
			log.Fatal(err)
		}
		z := makeZimtohrli()
		res, err := bundles.ZimtohrliMSE(z, true)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("## %v\n\n", *mse)
		fmt.Print("Error for MOS datasets is `human-MOS - Zimtohrli-predicted-MOS`. Error for JND datasets is `distance from correct side of threshold`.\n\n")
		fmt.Printf("MSE: %.15f\n\n", res)
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
		board, err := bundles.Leaderboard(15)
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
