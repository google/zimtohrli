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

// Package data contains structs and methods common for listening test datasets.
package data

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/dgryski/go-onlinestats"
	"github.com/google/zimtohrli/go/aio"
	"github.com/google/zimtohrli/go/audio"
	"github.com/google/zimtohrli/go/goohrli"
	"github.com/google/zimtohrli/go/progress"
	"github.com/google/zimtohrli/go/worker"
	"gonum.org/v1/gonum/optimize"

	_ "github.com/mattn/go-sqlite3" // To open sqlite3-databases.
)

const (
	// MOS is mean opinion score from human evaluators.
	MOS ScoreType = "MOS"
	// Zimtohrli is the Zimtohrli distance.
	Zimtohrli ScoreType = "Zimtohrli"
	// JND is 1 if the evaluator detected a difference and 0 if not.
	JND ScoreType = "JND"
	// ViSQOL is the ViSQOL MOS.
	ViSQOL = "ViSQOL"
)

// ScoreType represents a type of score, such as MOS or Zimtohrli.
type ScoreType string

// Better returns 1 if higher is better for the score type, or -1 if lower is better.
func (s ScoreType) Better() int {
	switch s {
	case MOS:
		return 1
	case Zimtohrli:
		return -1
	case JND:
		return -1
	case ViSQOL:
		return 1
	default:
		return 0
	}
}

// ScoreTypes is a slice of ScoreType.
type ScoreTypes []ScoreType

func (s ScoreTypes) Len() int {
	return len(s)
}

func (s ScoreTypes) Less(i, j int) bool {
	return string(s[i]) < string(s[j])
}

func (s ScoreTypes) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

// Study contains data from a study.
type Study struct {
	dir string
	db  *sql.DB
}

// ReferenceBundle is a plain data type containing a bunch of references, typicall the content of a study.
type ReferenceBundle struct {
	// Dir is the directory of the source study.
	Dir string
	// References are all the reference sounds of the bundle.
	References []*Reference
	// ScoreTypes are the number of scores of each type in the bundle.
	ScoreTypes map[ScoreType]int
	// ScoreTypeLimits are the upper/lower limits of each score type in the bundle.
	ScoreTypeLimits map[ScoreType][2]*float64

	// mosScaler returns the MOS score provided scaled to 1-5. Useful for datasets where the MOS is scaled up to 0-100.
	mosScaler func(float64) float64
}

// ReferenceBundles is a slice of ReferenceBundle.
type ReferenceBundles []*ReferenceBundle

// IsJND returns if this bundle is one with just-noticeable-difference evaluations, and only those.
func (r *ReferenceBundle) IsJND() bool {
	_, res := r.ScoreTypes[JND]
	return res
}

// ScaledMOS returns the MOS score scaled to 1-5.
func (r *ReferenceBundle) ScaledMOS(mos float64) (float64, error) {
	if r.mosScaler == nil {
		if math.Abs(*r.ScoreTypeLimits[MOS][0]-1) < 0.2 && math.Abs(*r.ScoreTypeLimits[MOS][1]-5) < 0.2 {
			r.mosScaler = func(mos float64) float64 {
				return mos
			}
		} else if math.Abs(*r.ScoreTypeLimits[MOS][0]) < 0.2 && math.Abs(*r.ScoreTypeLimits[MOS][1]-100) < 0.2 {
			r.mosScaler = func(mos float64) float64 {
				return 1 + 0.04*mos
			}
		} else {
			return 0, fmt.Errorf("minimum MOS %v and maximum MOS %v are confusing", *r.ScoreTypeLimits[MOS][0], *r.ScoreTypeLimits[MOS][1])
		}
	}
	return r.mosScaler(mos), nil
}

// SortedTypes returns the score types of a bundle, alphabetically ordered.
func (r *ReferenceBundle) SortedTypes() ScoreTypes {
	sorted := ScoreTypes{}
	for scoreType := range r.ScoreTypes {
		sorted = append(sorted, scoreType)
	}
	sort.Sort(sorted)
	return sorted
}

// Add adds a reference to a bundle.
func (r *ReferenceBundle) Add(ref *Reference) {
	for _, dist := range ref.Distortions {
		for scoreType, value := range dist.Scores {
			r.ScoreTypes[scoreType]++
			if r.ScoreTypeLimits[scoreType][0] == nil || *r.ScoreTypeLimits[scoreType][0] > value {
				valueCopy := value
				limits := r.ScoreTypeLimits[scoreType]
				limits[0] = &valueCopy
				r.ScoreTypeLimits[scoreType] = limits
			}
			if r.ScoreTypeLimits[scoreType][1] == nil || *r.ScoreTypeLimits[scoreType][1] < value {
				valueCopy := value
				limits := r.ScoreTypeLimits[scoreType]
				limits[1] = &valueCopy
				r.ScoreTypeLimits[scoreType] = limits
			}
		}
	}
	r.References = append(r.References, ref)
}

// ToBundle returns a reference bundle for this study.
func (s *Study) ToBundle() (*ReferenceBundle, error) {
	result := &ReferenceBundle{
		Dir:             s.dir,
		ScoreTypes:      map[ScoreType]int{},
		ScoreTypeLimits: map[ScoreType][2]*float64{},
	}
	if err := s.ViewEachReference(func(ref *Reference) error {
		result.Add(ref)
		return nil
	}); err != nil {
		return nil, err
	}
	max := 0.0
	var maxType *ScoreType
	min := 0.0
	var minType *ScoreType
	for loopScoreType, count := range result.ScoreTypes {
		scoreType := loopScoreType
		fCount := float64(count)
		if maxType == nil || fCount > max {
			max = fCount
			maxType = &scoreType
		}
		if minType == nil || fCount < min {
			min = fCount
			minType = &scoreType
		}
	}
	if maxType == nil || minType == nil {
		return nil, fmt.Errorf("%q has no score types?", s.dir)
	}
	if (max - min) > max*0.01 {
		log.Printf("%q has %v scores and %q has %v scores in %q, more than 5%% missing scores", *minType, min, *maxType, max, s.dir)
	}
	return result, nil
}

// CorrelationScore contains the scorrelation score between two score types.
type CorrelationScore struct {
	ScoreTypeA ScoreType
	ScoreTypeB ScoreType
	Score      float64
}

// CorrelationRow is correlations between a single score type and all score types.
type CorrelationRow []CorrelationScore

func (c CorrelationRow) Len() int {
	return len(c)
}

func (c CorrelationRow) Less(i, j int) bool {
	return c[i].Score > c[j].Score
}

func (c CorrelationRow) Swap(i, j int) {
	c[i], c[j] = c[j], c[i]
}

// CorrelationTable contains the pairwise correlations between a set of score types.
type CorrelationTable []CorrelationRow

func (c CorrelationTable) String() string {
	listResult := Table{Row{"Score type", "Spearman correlation"}, nil}
	tableResult := Table{}
	header := Row{""}
	for _, score := range c[0] {
		header = append(header, string(score.ScoreTypeB))
	}
	tableResult = append(tableResult, header)
	tableResult = append(tableResult, nil)
	for _, scores := range c {
		row := Row{string(scores[0].ScoreTypeA)}
		for _, score := range scores {
			row = append(row, fmt.Sprintf("%.2f", score.Score))
		}
		tableResult = append(tableResult, row)
		if scores[0].ScoreTypeA == MOS {
			sort.Sort(scores)
			for _, score := range scores {
				if score.ScoreTypeB != MOS {
					listResult = append(listResult, Row{string(score.ScoreTypeB), fmt.Sprintf("%.2f", score.Score)})
				}
			}
		}
	}
	return fmt.Sprintf("### Spearman correlation table for all score types\n\n%s\n### Score type MOS Spearman correlation in order\n\n%s", tableResult.String(), listResult.String())
}

// Correlation returns the Spearman correlation between score type A and B.
func (r *ReferenceBundle) Correlation(typeA, typeB ScoreType) (float64, error) {
	scoresA := []float64{}
	scoresB := []float64{}
	appender := func(scores *[]float64, typ ScoreType, dist *Distortion) error {
		score := dist.Scores[typ]
		if typ == MOS {
			var err error
			if score, err = r.ScaledMOS(score); err != nil {
				return err
			}
		}
		*scores = append(*scores, score)
		return nil
	}
	for _, ref := range r.References {
		for _, dist := range ref.Distortions {
			if err := appender(&scoresA, typeA, dist); err != nil {
				return 0, err
			}
			if err := appender(&scoresB, typeB, dist); err != nil {
				return 0, err
			}
		}
	}
	if len(scoresA) != len(scoresB) {
		return 0, fmt.Errorf("not the same number of %q and %q: %v vs %v", typeA, typeB, len(scoresA), len(scoresB))
	}
	res, _ := onlinestats.Spearman(scoresA, scoresB)
	return math.Abs(res), nil
}

// Correlate returns a table of all scores in the bundle Spearman correlated to each other.
func (r *ReferenceBundle) Correlate() (CorrelationTable, error) {
	if r.IsJND() {
		return nil, fmt.Errorf("cannot correlate JND references")
	}
	result := CorrelationTable{}
	for _, typeA := range r.SortedTypes() {
		row := []CorrelationScore{}
		for _, typeB := range r.SortedTypes() {
			corr, err := r.Correlation(typeA, typeB)
			if err != nil {
				return nil, err
			}
			row = append(row, CorrelationScore{
				ScoreTypeA: typeA,
				ScoreTypeB: typeB,
				Score:      corr,
			})
		}
		result = append(result, row)
	}
	return result, nil
}

// Correlate returns a table of all scores in the bundles Spearman correlated to each other.
func (r ReferenceBundles) Correlate() (CorrelationTable, error) {
	merged := &ReferenceBundle{
		ScoreTypes:      map[ScoreType]int{},
		ScoreTypeLimits: map[ScoreType][2]*float64{},
	}
	for _, bundle := range r {
		if !bundle.IsJND() {
			for _, ref := range bundle.References {
				merged.Add(ref)
			}
		}
	}
	return merged.Correlate()
}

// JNDAccuracyScore contains the accuracy for a metric when used to predict audible differences, and the threshold when that accuracy was achieved.
type JNDAccuracyScore struct {
	ScoreType ScoreType
	Threshold float64
	Accuracy  float64
}

// JNDAccuracyScores contains the accuracy scores for multiple score types.
type JNDAccuracyScores []JNDAccuracyScore

func (a JNDAccuracyScores) String() string {
	table := Table{Row{"Score type", "Accuracy", "Threshold"}}
	table = append(table, nil)
	for _, score := range a {
		table = append(table, Row{string(score.ScoreType), fmt.Sprintf("%.2f", score.Accuracy), fmt.Sprintf("%.2v", score.Threshold)})
	}
	return fmt.Sprintf("### Maximal audibility classification accuracy and threshold per score type\n\n%s", table.String())
}

func (a JNDAccuracyScores) Len() int {
	return len(a)
}

func (a JNDAccuracyScores) Less(i, j int) bool {
	return a[i].Accuracy > a[j].Accuracy
}

func (a JNDAccuracyScores) Swap(i, j int) {
	a[i], a[j] = a[j], a[i]
}

func abs(i int) int {
	if i < 0 {
		return -1
	}
	return i
}

func ternarySearch(f func(int) float64, left, right int) int {
	for abs(right-left) > 2 {
		third := (right - left) / 3
		leftThird := left + third
		rightThird := right - third
		if f(leftThird) < f(rightThird) {
			left = leftThird
		} else {
			right = rightThird
		}
	}
	return (left + right) / 2
}

// JNDAccuracyAndThreshold returns the treshold for the score type that provides the highest accuracy at
// predicting the JND score (whether a human observer was able to detect the distortion), and the
// accuracy it provided.
func (r *ReferenceBundle) JNDAccuracyAndThreshold(scoreType ScoreType) (float64, float64, error) {
	if !r.IsJND() {
		return 0, 0, fmt.Errorf("cannot compute JND accuracy on non-JND references")
	}
	audible := sort.Float64Slice{}
	inaudible := sort.Float64Slice{}
	allMap := map[float64]struct{}{}
	for _, ref := range r.References {
		for _, dist := range ref.Distortions {
			jnd, found := dist.Scores[JND]
			if !found {
				return 0, 0, fmt.Errorf("%+v doesn't have a JND score", ref)
			}
			score := dist.Scores[scoreType]
			allMap[score] = struct{}{}
			switch jnd {
			case 0:
				inaudible = append(inaudible, score)
			case 1:
				audible = append(audible, score)
			default:
				return 0, 0, fmt.Errorf("%+v JND isn't 0 or 1", ref)
			}
		}
	}
	sort.Sort(audible)
	sort.Sort(inaudible)
	all := sort.Float64Slice{}
	for score := range allMap {
		all = append(all, score)
	}
	sort.Sort(all)
	accuracy := func(index int) float64 {
		threshold := all[index]
		audibleBelowThreshold := sort.SearchFloat64s(audible, threshold)
		inaudibleBelowThreshold := sort.SearchFloat64s(inaudible, threshold)
		correctAudible, correctInaudible := 0, 0
		if scoreType.Better() > 0 {
			correctAudible = audibleBelowThreshold
			correctInaudible = len(inaudible) - inaudibleBelowThreshold
		} else {
			correctAudible = len(audible) - audibleBelowThreshold
			correctInaudible = inaudibleBelowThreshold
		}
		return float64(correctAudible+correctInaudible) / float64(len(audible)+len(inaudible))
	}
	bestAccuracyThresholdIndex := ternarySearch(accuracy, 0, len(all)-1)
	return accuracy(bestAccuracyThresholdIndex), all[bestAccuracyThresholdIndex], nil
}

// JNDAccuracy returns the accuracy of each score type when used to predict audible differences.
func (r *ReferenceBundle) JNDAccuracy() (JNDAccuracyScores, error) {
	result := JNDAccuracyScores{}
	for scoreType := range r.ScoreTypes {
		if scoreType != JND {
			accuracy, threshold, err := r.JNDAccuracyAndThreshold(scoreType)
			if err != nil {
				return nil, err
			}
			result = append(result, JNDAccuracyScore{
				ScoreType: scoreType,
				Threshold: threshold,
				Accuracy:  accuracy,
			})
		}
	}
	sort.Sort(result)
	return result, nil
}

// JNDAccuracy returns the accuracy of each score type when used to predict audible differences.
func (r ReferenceBundles) JNDAccuracy() (JNDAccuracyScores, error) {
	merged := &ReferenceBundle{
		ScoreTypes:      map[ScoreType]int{},
		ScoreTypeLimits: map[ScoreType][2]*float64{},
	}
	for _, bundle := range r {
		if bundle.IsJND() {
			for _, ref := range bundle.References {
				merged.Add(ref)
			}
		}
	}
	return merged.JNDAccuracy()
}

// OptimizedZimtohrliMSE optimizes the MOS mapping and returns the ZimtohrliMSE using the optimized mapping.
func (r ReferenceBundles) OptimizedZimtohrliMSE() (float64, error) {
	optResult, err := r.OptimizeMapping()
	if err != nil {
		return 0, err
	}
	params := goohrli.DefaultParameters(aio.DefaultSampleRate)
	copy(params.MOSMapperParams[:], optResult.ParamsAfter)
	z := goohrli.New(params)
	return r.ZimtohrliMSE(z, true)
}

// ZimtohrliMSE returns the mean square of the ZimtohrliMSE of the bundles.
func (r ReferenceBundles) ZimtohrliMSE(z *goohrli.Goohrli, includeJND bool) (float64, error) {
	sumOfSquares := 0.0
	count := 0
	for _, bundle := range r {
		if includeJND || !bundle.IsJND() {
			mse, err := bundle.ZimtohrliMSE(z)
			if err != nil {
				return 0, err
			}
			sumOfSquares += mse * mse
			count += 1
		}
	}
	return sumOfSquares / float64(count), nil
}

// ZimtohrliMSE returns the precision when predicting the MOS score or JND difference.
func (r *ReferenceBundle) ZimtohrliMSE(z *goohrli.Goohrli) (float64, error) {
	if r.IsJND() {
		_, threshold, err := r.JNDAccuracyAndThreshold(Zimtohrli)
		if err != nil {
			return 0, err
		}
		sumOfSquares := 0.0
		count := 0
		for _, ref := range r.References {
			for _, dist := range ref.Distortions {
				jnd, found := dist.Scores[JND]
				if !found {
					return 0, fmt.Errorf("%+v doesn't have a JND score", ref)
				}
				zimt, found := dist.Scores[Zimtohrli]
				if !found {
					return 0, fmt.Errorf("%+v doesn't have a Zimtohrli score", ref)
				}
				switch jnd {
				case 0:
					if zimt >= threshold {
						delta := zimt - threshold
						sumOfSquares += delta * delta
					}
				case 1:
					if zimt < threshold {
						delta := zimt - threshold
						sumOfSquares += delta * delta
					}
				default:
					return 0, fmt.Errorf("%+v JND isn't 0 or 1", ref)
				}
				count++
			}
		}
		return sumOfSquares / float64(count), nil
	} else {
		sumOfSquares := 0.0
		count := 0
		for _, ref := range r.References {
			for _, dist := range ref.Distortions {
				mos, found := dist.Scores[MOS]
				if !found {
					return 0, fmt.Errorf("%+v doesn't have a MOS score", ref)
				}
				zimt, found := dist.Scores[Zimtohrli]
				if !found {
					return 0, fmt.Errorf("%+v doesn't have a Zimtohrli score", ref)
				}
				scaledMOS, err := r.ScaledMOS(mos)
				if err != nil {
					return 0, err
				}
				delta := scaledMOS - z.MOSFromZimtohrli(zimt)
				sumOfSquares += delta * delta
				count++
			}
		}
		return sumOfSquares / float64(count), nil
	}
}

// Studies is a slice of studies.
type Studies []*Study

// ToBundles returns reference bundles with the content of the studies.
func (s Studies) ToBundles() (ReferenceBundles, error) {
	result := make(ReferenceBundles, len(s))
	var err error
	for index, study := range s {
		if result[index], err = study.ToBundle(); err != nil {
			return nil, err
		}
	}
	return result, nil
}

// CalculateZimtohrliMSE calculates Zimtohrli scores for all examples in the bundles, optimizes the MOS mapping, and returns the resulting MSE.
func (r ReferenceBundles) CalculateZimtohrliMSE(z *goohrli.Goohrli) (float64, error) {
	for _, bundle := range r {
		bar := progress.New(fmt.Sprintf("Calculating for %v", filepath.Base(bundle.Dir)))
		pool := &worker.Pool[any]{
			Workers:  runtime.NumCPU(),
			OnChange: bar.Update,
		}
		if err := bundle.Calculate(map[ScoreType]Measurement{Zimtohrli: z.NormalizedAudioDistance}, pool, true); err != nil {
			return 0, err
		}
		bar.Finish()
	}
	return r.OptimizedZimtohrliMSE()
}

// MappingOptimizationResult contains the results of optimizing the MOS mapping.
type MappingOptimizationResult struct {
	ParamsBefore []float64
	MSEBefore    float64
	ParamsAfter  []float64
	MSEAfter     float64
}

// OptimizeMOSMapping optimizes the MOS mapping parameters.
func (r ReferenceBundles) OptimizeMapping() (*MappingOptimizationResult, error) {
	startParams := goohrli.DefaultParameters(aio.DefaultSampleRate)
	errors := []error{}
	p := optimize.Problem{
		Func: func(x []float64) float64 {
			params := startParams
			for index := range params.MOSMapperParams {
				params.MOSMapperParams[index] = math.Abs(x[index])
			}
			z := goohrli.New(params)
			result, err := r.ZimtohrliMSE(z, false)
			if err != nil {
				errors = append(errors, err)
			}
			return result
		},
		Status: func() (optimize.Status, error) {
			if len(errors) > 0 {
				return optimize.Failure, fmt.Errorf("%+v", errors)
			}
			return optimize.NotTerminated, nil
		},
	}
	result := &MappingOptimizationResult{
		ParamsBefore: startParams.MOSMapperParams[:],
		MSEBefore:    p.Func(startParams.MOSMapperParams[:]),
	}
	optResult, err := optimize.Minimize(p, startParams.MOSMapperParams[:], &optimize.Settings{Concurrent: runtime.NumCPU()}, nil)
	if err != nil {
		return nil, err
	}
	if err := optResult.Status.Err(); err != nil {
		return nil, err
	}
	result.ParamsAfter = optResult.X
	result.MSEAfter = optResult.F
	return result, nil
}

// OptimizeEvent is a step in the optimization process.
type OptimizeEvent struct {
	Parameters goohrli.Parameters
	Step       int
	MSE        float64
}

// Recorder logs optimization progress.
type Recorder struct {
	Output *os.File

	startParameters goohrli.Parameters
}

func (r *Recorder) Init() error {
	return nil
}

func (r *Recorder) paramsToX(p goohrli.Parameters) []float64 {
	return []float64{p.FrequencyResolution / r.startParameters.FrequencyResolution}
}

func (r *Recorder) xToParams(x []float64) goohrli.Parameters {
	cpy := r.startParameters
	cpy.FrequencyResolution *= x[0]
	return cpy
}

func (r *Recorder) Record(loc *optimize.Location, op optimize.Operation, stats *optimize.Stats) error {
	params := r.xToParams(loc.X)
	switch op {
	case optimize.InitIteration:
		log.Printf("Initialized solution %+v with MSE %v", params, loc.F)
	case optimize.MajorIteration:
		log.Printf("%v iterations, candidate solution %+v with MSE %v", stats.MajorIterations, params, loc.F)
	case optimize.FuncEvaluation:
		log.Printf("%v iterations, evaluated at %+v with MSE %v", stats.MajorIterations, params, loc.F)
	case optimize.MethodDone:
		log.Printf("Solution %+v found with MSE %v", loc.F, loc.X)
	}
	if r.Output == nil {
		return nil
	}
	ev := OptimizeEvent{
		Parameters: params,
		Step:       stats.MajorIterations,
		MSE:        loc.F,
	}
	b, err := json.Marshal(ev)
	if err != nil {
		return err
	}
	if _, err := r.Output.WriteString(string(b) + "\n"); err != nil {
		return err
	}
	return r.Output.Sync()
}

// Optimize will use optimize a Zimtohrli metric for predicting these bundles.
func (r ReferenceBundles) Optimize(recorder *Recorder) error {
	recorder.startParameters = goohrli.DefaultParameters(aio.DefaultSampleRate)
	errors := []error{}
	p := optimize.Problem{
		Func: func(x []float64) float64 {
			z := goohrli.New(recorder.xToParams(x))
			mse, err := r.CalculateZimtohrliMSE(z)
			if err != nil {
				errors = append(errors, err)
			}
			return mse
		},
		Status: func() (optimize.Status, error) {
			if len(errors) > 0 {
				return optimize.Failure, fmt.Errorf("%+v", errors)
			}
			return optimize.NotTerminated, nil
		},
	}
	optResult, err := optimize.Minimize(p, recorder.paramsToX(recorder.startParameters), &optimize.Settings{Recorder: recorder}, nil)
	if err != nil {
		return err
	}
	if err := optResult.Status.Err(); err != nil {
		return err
	}
	return nil
}

func gitIdentity() (*string, error) {
	if _, err := exec.Command("git", "rev-parse").CombinedOutput(); err != nil {
		return nil, nil
	}
	repo, err := exec.Command("git", "config", "--get", "remote.origin.url").CombinedOutput()
	if err != nil {
		return nil, err
	}
	desc, err := exec.Command("git", "describe", "--tags").CombinedOutput()
	if err != nil {
		return nil, err
	}
	branch, err := exec.Command("git", "branch", "--show-current").CombinedOutput()
	if err != nil {
		return nil, err
	}
	result := fmt.Sprintf("Revision %s, branch %s, origin %s", strings.TrimSpace(string(desc)), strings.TrimSpace(string(branch)), strings.TrimSpace(string(repo)))
	return &result, nil
}

// Report returns a Markdown report based on the bundles.
func (r ReferenceBundles) Report() (string, error) {
	res := &bytes.Buffer{}
	fmt.Fprintf(res, `# Zimtohrli correlation report

Created at %s
	
`, time.Now().Format(time.DateOnly))
	id, err := gitIdentity()
	if err != nil {
		log.Fatal(err)
	}
	if id != nil {
		fmt.Fprintf(res, "%s\n\n", *id)
	}
	for _, bundle := range r {
		fmt.Fprintf(res, "## %s\n\n", filepath.Base(bundle.Dir))
		if bundle.IsJND() {
			accuracy, err := bundle.JNDAccuracy()
			if err != nil {
				return "", err
			}
			fmt.Fprintln(res, accuracy)
		} else {
			corrTable, err := bundle.Correlate()
			if err != nil {
				return "", err
			}
			fmt.Fprintln(res, corrTable)
		}
	}

	fmt.Fprintf(res, "## Global leaderboard across all studies\n\n")

	board, err := r.Leaderboard(2)
	if err != nil {
		return "", err
	}
	fmt.Fprint(res, board)
	return res.String(), nil
}

// MSEScore is MSE for a score type across a set of studies.
type MSEScore struct {
	Decimals  int
	ScoreType ScoreType
	MSE       float64
	MinScore  float64
	MaxScore  float64
	MeanScore float64
}

// MSEScores contains the MSE for multiple score types.
type MSEScores []MSEScore

func (m MSEScores) String() string {
	table := Table{Row{"Score type", "MSE", "Min score", "Max score", "Mean score"}, nil}
	for _, score := range m {
		precisionString := fmt.Sprintf("%%.%df", score.Decimals)
		table = append(table, Row{string(score.ScoreType), fmt.Sprintf(precisionString, score.MSE), fmt.Sprintf(precisionString, score.MinScore), fmt.Sprintf(precisionString, score.MaxScore), fmt.Sprintf(precisionString, score.MeanScore)})
	}
	return fmt.Sprintf("### Mean square error (1 - Spearman correlation, or 1 - accuracy) per score type\n\n%s", table.String())
}

func (m MSEScores) Len() int {
	return len(m)
}

func (m MSEScores) Less(i, j int) bool {
	return m[i].MSE < m[j].MSE
}

func (m MSEScores) Swap(i, j int) {
	m[i], m[j] = m[j], m[i]
}

// Leaderboard returns the sorted mean squared errors for each score type that is represented in all bundles.
func (r ReferenceBundles) Leaderboard(decimals int) (MSEScores, error) {
	representedScoreTypes := map[ScoreType]int{}
	for index, bundle := range r {
		if index == 0 {
			for scoreType, count := range bundle.ScoreTypes {
				if scoreType != MOS && scoreType != JND {
					representedScoreTypes[scoreType] = count
				}
			}
		} else {
			for previouslyFoundScoreType := range representedScoreTypes {
				if count, found := bundle.ScoreTypes[previouslyFoundScoreType]; !found {
					delete(representedScoreTypes, previouslyFoundScoreType)
				} else if previouslyFoundScoreType != MOS && previouslyFoundScoreType != JND {
					representedScoreTypes[previouslyFoundScoreType] += count
				}
			}
		}
	}

	sumOfSquares := map[ScoreType]float64{}
	sums := map[ScoreType]float64{}
	mins := map[ScoreType]float64{}
	maxs := map[ScoreType]float64{}

	addScore := func(scoreType ScoreType, score float64) {
		sums[scoreType] += score
		if currentMin, found := mins[scoreType]; !found || (found && score < currentMin) {
			mins[scoreType] = score
		}
		if currentMax, found := maxs[scoreType]; !found || (found && score > currentMax) {
			maxs[scoreType] = score
		}
		loss := 1.0 - score
		sumOfSquares[scoreType] += loss * loss
	}
	for _, bundle := range r {
		if bundle.IsJND() {
			accuracies, err := bundle.JNDAccuracy()
			if err != nil {
				return nil, err
			}
			for _, accuracy := range accuracies {
				if _, found := representedScoreTypes[accuracy.ScoreType]; found {
					addScore(accuracy.ScoreType, accuracy.Accuracy)
				}
			}
		} else {
			correlations, err := bundle.Correlate()
			if err != nil {
				return nil, err
			}
			for _, row := range correlations {
				if row[0].ScoreTypeA == MOS {
					for _, correlation := range row {
						if _, found := representedScoreTypes[correlation.ScoreTypeB]; found {
							addScore(correlation.ScoreTypeB, correlation.Score)
						}
					}
				}
			}
		}
	}
	result := MSEScores{}
	numStudiesRecpripcal := 1.0 / float64(len(r))
	for scoreType, squareSum := range sumOfSquares {
		result = append(result, MSEScore{
			Decimals:  decimals,
			ScoreType: scoreType,
			MSE:       squareSum * numStudiesRecpripcal,
			MeanScore: sums[scoreType] * numStudiesRecpripcal,
			MinScore:  mins[scoreType],
			MaxScore:  maxs[scoreType],
		})
	}
	sort.Sort(result)
	return result, nil
}

// OpenBundles is a shortcut to opening multiple bundles from a glob.
func OpenBundles(glob string) (ReferenceBundles, error) {
	studies, err := OpenStudies(glob)
	if err != nil {
		return nil, err
	}
	defer studies.Close()
	if len(studies) == 0 {
		return nil, fmt.Errorf("no studies found in %v", glob)
	}
	return studies.ToBundles()
}

// OpenStudies returns the studies contained in the directories defined by the glob.
func OpenStudies(glob string) (Studies, error) {
	directories, err := filepath.Glob(glob)
	if err != nil {
		return nil, err
	}
	result := make(Studies, len(directories))
	for index, dir := range directories {
		if result[index], err = OpenStudy(dir); err != nil {
			return nil, err
		}
	}
	if len(result) == 0 {
		return nil, fmt.Errorf("no studies found in %v", glob)
	}
	return result, nil
}

// Close closes the studies.
func (s Studies) Close() error {
	for _, study := range s {
		if err := study.Close(); err != nil {
			return err
		}
	}
	return nil
}

// OpenStudy opens a study from a database directory.
// If the study doesn't exist, it will be created.
func OpenStudy(dir string) (*Study, error) {
	err := os.MkdirAll(dir, 0755)
	if err != nil && !os.IsExist(err) {
		return nil, fmt.Errorf("trying to create %q: %v", dir, err)
	}
	dbPath := filepath.Join(dir, "db.sqlite3")
	_, err = os.Stat(dbPath)
	if os.IsNotExist(err) {
		if _, err = os.Create(dbPath); err != nil {
			return nil, err
		}
	} else if err != nil {
		return nil, err
	}
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("trying to open %q: %v", dbPath, err)
	}
	if _, err := db.Exec("CREATE TABLE IF NOT EXISTS OBJ (ID BLOB PRIMARY KEY, DATA BLOB)"); err != nil {
		return nil, fmt.Errorf("trying to ensure object table: %v", err)
	}
	return &Study{
		dir: dir,
		db:  db,
	}, nil
}

// Close closes the study.
func (s *Study) Close() error {
	return s.db.Close()
}

// Measurement returns distance between sounds.
type Measurement func(reference, distortion *audio.Audio) (float64, error)

// Calculate computes measurements and populates the scores of the distortions.
func (r *ReferenceBundle) Calculate(measurements map[ScoreType]Measurement, pool *worker.Pool[any], force bool) error {
	for _, loopRef := range r.References {
		refNeededMeasurements := map[ScoreType]Measurement{}
		for _, dist := range loopRef.Distortions {
			for scoreType, measurement := range measurements {
				if _, found := dist.Scores[scoreType]; force || !found {
					refNeededMeasurements[scoreType] = measurement
				}
			}
		}
		if len(refNeededMeasurements) == 0 {
			continue
		}
		ref := loopRef
		pool.Submit(func(func(any)) error {
			refAudio, err := ref.Load(r.Dir)
			if err != nil {
				return err
			}
			for _, loopDist := range ref.Distortions {
				distNeededMeasurements := map[ScoreType]Measurement{}
				for scoreType, measurement := range refNeededMeasurements {
					if _, found := loopDist.Scores[scoreType]; force || !found {
						distNeededMeasurements[scoreType] = measurement
					}
				}
				if len(distNeededMeasurements) == 0 {
					continue
				}
				dist := loopDist
				pool.Submit(func(func(any)) error {
					distAudio, err := dist.Load(r.Dir)
					if err != nil {
						return err
					}
					for loopScoreType := range distNeededMeasurements {
						scoreType := loopScoreType
						pool.Submit(func(func(any)) error {
							score, err := distNeededMeasurements[scoreType](refAudio, distAudio)
							if err != nil {
								return err
							}
							if math.IsNaN(score) {
								return fmt.Errorf("NaN scores not allowed")
							}
							dist.Scores[scoreType] = score
							return nil
						})
					}
					return nil
				})
			}
			return nil
		})
	}
	return pool.Error()
}

// ViewEachReference returns each reference in the study.
func (s *Study) ViewEachReference(f func(*Reference) error) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	rows, err := tx.Query("SELECT DATA FROM OBJ")
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var value []byte
		if err := rows.Scan(&value); err != nil {
			return err
		}
		ref := &Reference{}
		if err := json.Unmarshal(value, ref); err != nil {
			return err
		}
		if err := f(ref); err == io.EOF {
			break
		} else if err != nil {
			return err
		}
	}
	return nil
}

// Put inserts some references into a study.
func (s *Study) Put(refs []*Reference) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	if err := func() error {
		for _, ref := range refs {
			b, err := json.Marshal(ref)
			if err != nil {
				return err
			}
			if _, err = tx.Exec("INSERT INTO OBJ (ID, DATA) VALUES (?, ?) ON CONFLICT (ID) DO UPDATE SET DATA = ?", []byte(ref.Name), b, b); err != nil {
				return err
			}
		}
		return nil
	}(); err != nil {
		if rerr := tx.Rollback(); rerr != nil {
			return rerr
		}
		return err
	}
	return tx.Commit()
}

// Distortion contains data for a distortion of a reference.
type Distortion struct {
	Name   string
	Path   string
	Scores map[ScoreType]float64
}

// Load returns the audio for this distortion.
func (d *Distortion) Load(dir string) (*audio.Audio, error) {
	return aio.Load(filepath.Join(dir, d.Path))
}

// Reference contains data for a reference.
type Reference struct {
	Name        string
	Path        string
	Distortions []*Distortion
}

// Load returns the audio for this reference.
func (r *Reference) Load(dir string) (*audio.Audio, error) {
	return aio.Load(filepath.Join(dir, r.Path))
}
