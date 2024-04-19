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

// Package runner provides utility functions to run the dataset tool.
package runner

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"sync"

	"github.com/dgryski/go-onlinestats"
	"github.com/google/zimtohrli/go/dataset"
	"github.com/google/zimtohrli/go/dataset/coresvnet"
	"github.com/google/zimtohrli/go/goohrli"
	"github.com/google/zimtohrli/go/progress"
	"github.com/google/zimtohrli/go/worker"
	"gonum.org/v1/gonum/optimize"
)

// Setup contains the arguments for a tool run. The fields are easily populated by flags.
type Setup struct {
	Show                         *bool                       `flag:"Whether to show the dataset metadata."`
	Dataset                      *string                     `flag:"The name of the dataset to fetch metadata for. One of %+v."`
	Calculate                    *string                     `flag:"The destination file to put metric calculation results in."`
	Correlate                    *string                     `flag:"The source file to correlate metric calculation results."`
	CopyToDirectory              *string                     `flag:"The destination directory to copy the dataset to."`
	OptimizeDatasets             *string                     `flag:"The path to a directory with local datasets to optimize with."`
	OptimizeValidationFraction   *float64                    `flag:"The fraction of optimization datasets to use for validation."`
	OptimizeMinibatchFraction    *float64                    `flag:"The fraction of training data to use per minibatch."`
	ZimtohrliCompare             *bool                       `flag:"Whether to compare the data using Zimtohrli."`
	ZimtohrliFrequencyResolution *float64                    `flag:"The frequency resolution to use for Zimtohrli compare."`
	LocalDatasetPath             *string                     `flag:"The path to a local dataset to use if Dataset is 'local'."`
	MaxWorkers                   *int                        `flag:"The maximum number of workers to use for metric calculation."`
	ProgressDirectory            *string                     `flag:"The path to a directory to write intermediate data to enable continuing at a later time."`
	ExtraDatasets                map[string]DatasetGenerator `flag:"The extra datasets to use."`
}

// SetupFlagString returns the value of the flag tag in the Setup struct, which is intended for flag documentation.
func SetupFlagString(field string) string {
	structField, found := reflect.TypeOf(Setup{}).FieldByName(field)
	if !found {
		return "unknown field"
	}
	return structField.Tag.Get("flag")
}

// DatasetNames returns the names of all available datasets.
func (s Setup) DatasetNames() []string {
	res := []string{}
	for name := range s.Datasets() {
		res = append(res, name)
	}
	return res
}

// DatasetGenerator is a function that returns a dataset.Dataset.
type DatasetGenerator func() (*dataset.Dataset, error)

// Datasets returns a map of dataset names to dataset generators.
func (s Setup) Datasets() map[string]DatasetGenerator {
	result := map[string]DatasetGenerator{
		"coresv.net": coresvnet.Fetch,
		"local": func() (*dataset.Dataset, error) {
			return dataset.LoadLocal(*s.LocalDatasetPath)
		},
	}
	if len(s.ExtraDatasets) > 0 {
		for name, dataset := range s.ExtraDatasets {
			result[name] = dataset
		}
	}
	return result
}

func safeString(s *string) string {
	if *s == "" {
		return ""
	}
	return *s
}

func safeBool(b *bool) bool {
	if *b == false {
		return false
	}
	return *b
}

func safeFloat64(f *float64) float64 {
	if *f == 0.0 {
		return 0.0
	}
	return *f
}

func safeInt(i *int) int {
	if *i == 0 {
		return 0
	}
	return *i
}

type comparison struct {
	pathA          string
	pathB          string
	referenceScore float64
}

type comparisonSlice []*comparison

func (c comparisonSlice) Len() int {
	return len(c)
}

func (c comparisonSlice) Less(i, j int) bool {
	if c[i].referenceScore != c[j].referenceScore {
		return c[i].referenceScore < c[j].referenceScore
	}
	if c[i].pathA != c[j].pathA {
		return c[i].pathA < c[j].pathA
	}
	return c[i].pathB < c[j].pathB
}

func (c comparisonSlice) Swap(i, j int) {
	c[i], c[j] = c[j], c[i]
}

type prepareResult struct {
	comparison *comparison
	dataset    string
}

func (s Setup) getOptimizeComparisons() (map[string]comparisonSlice, error) {
	datasetDirs, err := filepath.Glob(filepath.Join(safeString(s.OptimizeDatasets), "*"))
	if err != nil {
		return nil, err
	}
	bar := progress.New(fmt.Sprintf("Preparing %v datasets", len(datasetDirs)))
	defer fmt.Println()
	pool := &worker.Pool[prepareResult]{
		Workers: safeInt(s.MaxWorkers),
		OnComplete: func(submitted, completed int) {
			bar.Update(submitted, completed)
		},
	}
	for _, loopDatasetDir := range datasetDirs {
		datasetDir := loopDatasetDir
		data, err := dataset.LoadLocal(datasetDir)
		if err != nil {
			return nil, err
		}
		for _, loopRef := range data.References {
			ref := loopRef
			pool.Submit(func(f func(prepareResult)) error {
				referencePath, _, err := ref.Provider(nil)
				if err != nil {
					return err
				}
				for _, dist := range ref.Distortions {
					distPath, _, err := dist.Provider(nil)
					if err != nil {
						return err
					}
					f(prepareResult{
						comparison: &comparison{
							pathA:          referencePath,
							pathB:          distPath,
							referenceScore: dist.Score,
						},
						dataset: datasetDir,
					})
				}
				return nil
			})
		}
	}
	if err := pool.Error(); err != nil {
		return nil, err
	}
	results := map[string]comparisonSlice{}
	for result := range pool.Results() {
		results[result.dataset] = append(results[result.dataset], result.comparison)
	}
	for _, comps := range results {
		sort.Sort(comps)
	}

	return results, nil
}

type compareResult struct {
	dataset   string
	refScore  float64
	zimtScore float64
}

func (s Setup) optimize() error {
	rng := rand.New(rand.NewSource(0))
	comparisons, err := s.getOptimizeComparisons()
	if err != nil {
		return err
	}
	defer func() {
		for _, comps := range comparisons {
			for _, comp := range comps {
				os.Remove(comp.pathA)
				os.Remove(comp.pathB)
			}
		}
	}()

	validationComparisons := map[string]comparisonSlice{}
	trainingComparisons := map[string]comparisonSlice{}
	for name, comps := range comparisons {
		numValidationComparisons := int(safeFloat64(s.OptimizeValidationFraction) * float64(len(comps)))
		for consecutiveIndex, randomIndex := range rng.Perm(len(comps)) {
			if consecutiveIndex < numValidationComparisons {
				validationComparisons[name] = append(validationComparisons[name], comps[randomIndex])
			} else {
				trainingComparisons[name] = append(trainingComparisons[name], comps[randomIndex])
			}
		}
		log.Printf("%v has %v training comparisons and %v validation comparisons", filepath.Base(name), len(trainingComparisons[name]), len(validationComparisons[name]))
	}
	comparisons = nil

	zimt := &dataset.Zimtohrli{Goohrli: goohrli.New(48000, 4)}
	calculate := func(x []float64, comparisons map[string]comparisonSlice) (float64, error) {
		zimt.Goohrli.SetFreqNormOrder(zimt.Goohrli.GetFreqNormOrder() * float32(x[0]))
		zimt.Goohrli.SetTimeNormOrder(zimt.Goohrli.GetTimeNormOrder() * float32(x[1]))
		results := map[string][2][]float64{}
		bar := progress.New(fmt.Sprintf("Calculating for %v datasets", len(comparisons)))
		if err := func() error {
			defer fmt.Println()
			pool := &worker.Pool[compareResult]{
				Workers: safeInt(s.MaxWorkers),
				OnComplete: func(submitted, completed int) {
					bar.Update(submitted, completed)
				},
			}

			for loopDataset, loopComps := range comparisons {
				dataset := loopDataset
				comps := loopComps
				for _, loopComparison := range comps {
					comparison := loopComparison
					pool.Submit(func(f func(compareResult)) error {
						scores, err := zimt.Distances(comparison.pathA, comparison.pathB)
						if err != nil {
							return err
						}
						if len(scores) == 1 {
							f(compareResult{
								dataset:   dataset,
								zimtScore: scores[0],
								refScore:  comparison.referenceScore,
							})
						} else {
							return fmt.Errorf("Zimtohrli returned %+v instead of a single score", scores)
						}
						return nil
					})
				}
			}
			if err := pool.Error(); err != nil {
				return err
			}
			for result := range pool.Results() {
				scores, found := results[result.dataset]
				if !found {
					scores = [2][]float64{}
				}
				scores[0] = append(scores[0], result.refScore)
				scores[1] = append(scores[1], result.zimtScore)
				results[result.dataset] = scores
			}
			return nil
		}(); err != nil {
			return 0, err
		}
		spearmanSum := 0.0
		for dataset, scores := range results {
			spearman, _ := onlinestats.Spearman(scores[0], scores[1])
			log.Printf("Spearman for %v at %+v: %v", filepath.Base(dataset), x, spearman)
			spearmanSum += math.Abs(spearman)
		}
		return spearmanSum / float64(len(comparisons)), nil
	}

	statusLock := sync.RWMutex{}
	var statusErr error
	res, err := optimize.Minimize(optimize.Problem{
		Func: func(x []float64) float64 {
			minibatchComparisons := map[string]comparisonSlice{}
			total := 0
			for name, comps := range trainingComparisons {
				numMinibatchComparisons := int(safeFloat64(s.OptimizeMinibatchFraction) * float64(len(comps)))
				for _, index := range rng.Perm(len(comps))[:numMinibatchComparisons] {
					minibatchComparisons[name] = append(minibatchComparisons[name], comps[index])
					total++
				}
			}
			meanSpearman, err := calculate(x, minibatchComparisons)
			if err != nil {
				func() {
					statusLock.Lock()
					defer statusLock.Unlock()
					statusErr = err
				}()
				return 0
			}
			loss := math.Pow(1.0-math.Abs(meanSpearman), 2)
			log.Printf("Train minibatch mean Spearman for %+v: %v, loss=%v", x, meanSpearman, loss)
			return loss
		},
		Status: func() (optimize.Status, error) {
			statusLock.RLock()
			defer statusLock.RUnlock()
			return optimize.NotTerminated, statusErr
		},
	}, []float64{1.0, 1.0}, &optimize.Settings{
		Recorder: logRecorder{
			validate: func(loc *optimize.Location) (float64, error) {
				return calculate(loc.X, validationComparisons)
			},
		},
	}, nil)
	if err != nil {
		return err
	}
	log.Printf("%+v", res)
	return nil
}

type logRecorder struct {
	validate func(*optimize.Location) (float64, error)
}

func (l logRecorder) Init() error {
	return nil
}

func (l logRecorder) Record(loc *optimize.Location, op optimize.Operation, stats *optimize.Stats) error {
	if op == optimize.MajorIteration {
		spearman, err := l.validate(loc)
		if err != nil {
			return err
		}
		log.Printf("Validation set Spearman for %+v: %v", loc.X, spearman)
	}
	return nil
}

// Run runs the tool.
func (s Setup) Run() error {
	if safeString(s.OptimizeDatasets) != "" {
		return s.optimize()
	}

	if safeString(s.Dataset) == "" {
		return fmt.Errorf("No dataset provided")
	}
	if !safeBool(s.Show) && safeString(s.Calculate) == "" && safeString(s.Correlate) == "" && safeString(s.CopyToDirectory) == "" {
		return fmt.Errorf("No command (show, calculate, correlate) provided")
	}

	data, err := s.Datasets()[safeString(s.Dataset)]()
	if err != nil {
		return err
	}

	if safeBool(s.Show) {
		b, err := json.MarshalIndent(data, "", "  ")
		if err != nil {
			return err
		}
		fmt.Println(string(b))
	}

	if copyToDirectory := safeString(s.CopyToDirectory); copyToDirectory != "" {
		if err := os.MkdirAll(copyToDirectory, 0755); err != nil {
			return err
		}
		copyIndex := 0
		copyFile := func(provider dataset.PathProvider) (string, error) {
			existingPath, _, err := provider(nil)
			if err != nil {
				return "", err
			}
			newFilename := fmt.Sprintf("sample_%v.wav", copyIndex)
			copyIndex++
			data, err := os.ReadFile(existingPath)
			if err != nil {
				return "", err
			}
			return newFilename, os.WriteFile(filepath.Join(copyToDirectory, newFilename), data, 0644)
		}
		total := 0
		for _, ref := range data.References {
			total += 1 + len(ref.Distortions)
		}
		bar := progress.New("Copying dataset files")
		for _, ref := range data.References {
			newName, err := copyFile(ref.Provider)
			if err != nil {
				return err
			}
			ref.Name = newName
			bar.AddCompleted(1)
			for _, dist := range ref.Distortions {
				newName, err = copyFile(dist.Provider)
				if err != nil {
					return err
				}
				dist.Name = newName
				bar.AddCompleted(1)

			}
		}
		fmt.Println()
		jsonOut, err := os.Create(filepath.Join(copyToDirectory, "dataset.json"))
		if err != nil {
			return err
		}
		if err = json.NewEncoder(jsonOut).Encode(data); err != nil {
			return err
		}
	}

	if calculate := safeString(s.Calculate); calculate != "" {
		if err := os.MkdirAll(filepath.Dir(calculate), 0755); err != nil {
			return err
		}
		metrics := []dataset.MetricRunner{}
		if safeBool(s.ZimtohrliCompare) {
			metrics = append(metrics, &dataset.Zimtohrli{Goohrli: goohrli.New(48000, 4)})
		}
		if len(metrics) == 0 {
			fmt.Fprintln(os.Stderr, "No metrics to correlate. Please provide metric binaries.")
			os.Exit(-2)
		}
		bar := progress.New(fmt.Sprintf("Calculating for %v references", len(data.References)))
		result, err := data.Calculate(metrics, func(submitted, completed int) {
			bar.Update(submitted, completed)
		}, safeInt(s.MaxWorkers), safeString(s.ProgressDirectory))
		if err != nil {
			return err
		}
		defer fmt.Println()
		b, err := json.MarshalIndent(result, "", "  ")
		if err != nil {
			return err
		}
		if err := ioutil.WriteFile(calculate, b, 0644); err != nil {
			return err
		}
	}

	if correlate := safeString(s.Correlate); correlate != "" {
		metricDatasets := map[dataset.ScoreType]*dataset.Dataset{}
		inFile, err := os.Open(correlate)
		if err != nil {
			return err
		}
		defer inFile.Close()
		if err := json.NewDecoder(inFile).Decode(&metricDatasets); err != nil {
			return err
		}
		corr, err := data.Correlate(metricDatasets)
		if err != nil {
			return err
		}
		b, err := json.MarshalIndent(corr, "", "  ")
		if err != nil {
			return err
		}
		fmt.Println(string(b))
	}
	return nil
}
