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

// Package dataset contains structs and methods common for listening test datasets.
package dataset

import (
	"crypto/sha512"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"sync/atomic"

	"github.com/dgryski/go-onlinestats"
	"github.com/google/zimtohrli/go/goohrli"
	"github.com/google/zimtohrli/go/worker"
	"github.com/youpy/go-wav"
)

// LoadLocal reads a local dataset from a file and returns it.
func LoadLocal(path string) (*Dataset, error) {
	b, err := os.ReadFile(filepath.Join(path, "dataset.json"))
	if err != nil {
		return nil, err
	}
	data := &Dataset{}
	if err := json.Unmarshal(b, data); err != nil {
		return nil, err
	}
	for _, ref := range data.References {
		ref.Provider = FileCopyProvider(filepath.Join(path, ref.Name))
		for _, dist := range ref.Distortions {
			dist.Provider = FileCopyProvider(filepath.Join(path, dist.Name))
		}
	}
	return data, nil
}

// Format defines an audio file format.
// Typically this isn't important, since ffmpeg will convert almost anything to WAV correctly.
type Format string

const (
	// Wav format
	Wav Format = "wav"
	// Ogg format
	Ogg Format = "ogg"
	// Opus format
	Opus Format = "opus"
	// Aac format
	Aac Format = "aac"
	// Mp3 format
	Mp3 Format = "mp3"
	// Faac format
	Faac Format = "faac"
)

// PathProvider is a function that takes a wanted mean volume level in dB FS and returns a path
// to a WAV file at 48kHz sample rate, and the resulting mean volume level in dB FS.
// The file is always temporary and should be removed once not needed any more.
type PathProvider func(wantedMeanVolume *float64) (path string, outMeanVolume float64, err error)

var (
	volumedetectRegexp = regexp.MustCompile("\\smean_volume: (\\S+) dB")
)

func getAmpFromFFMPEGOutput(output []byte) (float64, error) {
	match := volumedetectRegexp.FindStringSubmatch(string(output))
	if match == nil {
		return 0, fmt.Errorf("ffmpeg output didn't print max volume:\n%s", string(output))
	}
	dBFS, err := strconv.ParseFloat(match[1], 64)
	if err != nil {
		return 0, err
	}
	return dBFS, nil
}

// FfmpegResample resamples a source file to 48kHz sample rate, possibly changes its mean volume, and returns a path to a new file containing the result along with the mean volume of the output.
func FfmpegResample(sourcePath string, wantedMeanVolume *float64) (destPath string, outMeanVolume float64, err error) {
	neededAmplification := 0.0
	if wantedMeanVolume != nil {
		ffmpegOutput, err := execute("ffmpeg", "-i", sourcePath, "-af", "volumedetect", "-vn", "-sn", "-dn", "-f", "null", "/dev/null")
		if err != nil {
			return "", 0, err
		}
		inMaxAmplitude, err := getAmpFromFFMPEGOutput(ffmpegOutput)
		if err != nil {
			return "", 0, err
		}
		neededAmplification = *wantedMeanVolume - inMaxAmplitude
	}
	out, err := os.CreateTemp(os.TempDir(), fmt.Sprintf("zimtohrli_listening_test.*.wav"))
	if err != nil {
		return "", 0, err
	}
	out.Close()
	ffmpegOutput, err := execute("ffmpeg", "-i", sourcePath, "-y", "-af", fmt.Sprintf("volume=%fdB,volumedetect", neededAmplification), "-vn", "-ar", "48000", out.Name())
	if err != nil {
		return "", 0, err
	}
	dBFS, err := getAmpFromFFMPEGOutput(ffmpegOutput)
	if err != nil {
		return "", 0, err
	}
	return out.Name(), dBFS, nil
}

// Distortion contains metadata for a distortion.
type Distortion struct {
	Name     string
	Provider PathProvider `json:"-"`
	Format   Format
	Score    float64
}

func execute(binary string, args ...string) ([]byte, error) {
	cmd := exec.Command(binary, args...)
	b, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("%v: %v\n%s", cmd, err, string(b))
	}
	return b, nil
}

// FileCopyProvider returns a PathProvider that copies a file from a path.
func FileCopyProvider(path string) PathProvider {
	return func(wantedMaxAmplitude *float64) (string, float64, error) {
		return FfmpegResample(path, wantedMaxAmplitude)
	}
}

// URLPathProvider returns a PathProvider that downloads a file from a URL.
func URLPathProvider(u *url.URL) PathProvider {
	return func(wantedMaxAmplitude *float64) (string, float64, error) {
		temporaryOut, err := os.CreateTemp(os.TempDir(), fmt.Sprintf("zimtohrli_listening_test.*.%s", filepath.Base(u.Path)))
		if err != nil {
			return "", 0, err
		}
		defer os.RemoveAll(temporaryOut.Name())
		if err := func() error {
			defer temporaryOut.Close()
			resp, err := http.Get(u.String())
			if err != nil {
				return err
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				return fmt.Errorf("failed to download %q: %v", u.String(), resp.StatusCode)
			}
			if _, err := io.Copy(temporaryOut, resp.Body); err != nil {
				return err
			}
			return nil
		}(); err != nil {
			return "", 0, err
		}
		return FfmpegResample(temporaryOut.Name(), wantedMaxAmplitude)
	}
}

// Reference contains metadata for a reference.
type Reference struct {
	Name        string
	Provider    PathProvider `json:"-"`
	Format      Format
	Distortions []*Distortion
}

type metricRunnerReferenceKey struct {
	ScoreTypes []ScoreType
	Input      *Reference
}

type metricRunnerReferenceResult struct {
	metricRunnerReferenceKey
	Output []*Reference
}

func (r *metricRunnerReferenceResult) score(m MetricRunner, pool *worker.Pool[any], progressDirectory string) error {
	if len(m.ScoreTypes()) != len(r.Output) {
		return fmt.Errorf("runner produces scores for %+v, but got %v references to populate", len(m.ScoreTypes()), len(r.Output))
	}

	resultPath := ""
	if progressDirectory != "" {
		if err := os.MkdirAll(progressDirectory, 0755); err != nil && !os.IsExist(err) {
			return err
		}
		jsonBytes, err := json.Marshal(r.metricRunnerReferenceKey)
		if err != nil {
			return err
		}
		hash := sha512.Sum512(jsonBytes)
		resultPath = filepath.Join(progressDirectory, fmt.Sprintf("result-%s.json", hex.EncodeToString(hash[:])))
		resultFile, err := os.Open(resultPath)
		if err == nil {
			defer resultFile.Close()
			if err := json.NewDecoder(resultFile).Decode(r); err != nil {
				return err
			}
			return nil
		}
		if !os.IsNotExist(err) {
			return err
		}
	}

	for _, out := range r.Output {
		out.Name, out.Format = r.Input.Name, r.Input.Format
		for _, distortion := range r.Input.Distortions {
			cpy := *distortion
			out.Distortions = append(out.Distortions, &cpy)
		}
	}

	referencePath, referenceMeanVolume, err := r.Input.Provider(nil)
	referenceMeanVolumePtr := &referenceMeanVolume
	if err != nil {
		return err
	}
	if !m.NeedsVolumeNormalization() {
		referenceMeanVolumePtr = nil
	}

	var calculations int32
	for distortionIndex, distortion := range r.Input.Distortions {
		distortionPath, distortionMeanVolume, err := distortion.Provider(referenceMeanVolumePtr)
		if err != nil {
			return err
		}
		if m.NeedsVolumeNormalization() && math.Abs(referenceMeanVolume-distortionMeanVolume) > 1 {
			return fmt.Errorf("reference %q has max volume %v, but distortion %q has max volume %v", r.Input.Name, referenceMeanVolume, distortion.Name, distortionMeanVolume)
		}
		calculations++
		pool.Submit(func(func(any)) error {
			scores, err := m.Distances(referencePath, distortionPath)
			if err != nil {
				return err
			}
			for outIndex, out := range r.Output {
				out.Distortions[distortionIndex].Score = scores[outIndex]
			}
			if atomic.AddInt32(&calculations, -1) == 0 && os.Getenv("KEEP_TEMP_FILES") != "true" {
				os.RemoveAll(referencePath)
				os.RemoveAll(distortionPath)
			}
			return nil
		})
	}

	if progressDirectory != "" {
		resultFile, err := os.Create(resultPath)
		if err != nil {
			return err
		}
		defer resultFile.Close()
		if err := json.NewEncoder(resultFile).Encode(r); err != nil {
			return err
		}
	}
	return nil
}

// Score downloads the reference and distortions and populates the output references with metadata and scores computed using the pool.
// If progressDirectory != "", it will used as result cache.
func (r *Reference) Score(m MetricRunner, output []*Reference, pool *worker.Pool[any], progressDirectory string) error {
	mrs := &metricRunnerReferenceResult{
		metricRunnerReferenceKey: metricRunnerReferenceKey{
			ScoreTypes: m.ScoreTypes(),
			Input:      r,
		},
		Output: output,
	}
	return mrs.score(m, pool, progressDirectory)
}

// ScoreType is the type of score for a dataset.
type ScoreType string

const (
	// Mos is the score type for datasets scored using mean opinion score.
	Mos ScoreType = "MOS"
	// ZimtohrliResult is the score type for datasets scored using zimtohrli.
	ZimtohrliResult ScoreType = "Zimtohrli"
	// VisqolResult is the score type for datasets scored using visqol.
	VisqolResult ScoreType = "visqol"
	// ParlaqResult is the score type for datasets scored using parlaq.
	ParlaqResult ScoreType = "parlaq"
	// WarpQResult is the score type for datasets scored using warp-q.
	WarpQResult ScoreType = "warp-q"
	// GvpmosResult is the score type for datasets scored using gvpmos.
	GvpmosResult ScoreType = "gvpmos"
	// DnsmosSignalResult is the score type for datasets scored using dnsmos signal quality.
	DnsmosSignalResult ScoreType = "dnsmos_sig"
	// DnsmosBackgroundResult is the score type for datasets scored using dnsmos background noise quality.
	DnsmosBackgroundResult ScoreType = "dnsmos_bak"
	// DnsmosOverallResult is the score type for datasets scored using dnsmos overall quality.
	DnsmosOverallResult ScoreType = "dnsmos_ovr"
)

// Dataset contains metadata for a dataset.
type Dataset struct {
	// References contains reference files for the dataset.
	References []*Reference
	// ScoreType is the type of score for the dataset.
	ScoreType ScoreType
}

// MetricRunner is able to run a set of metrics to compare references and distortions.
type MetricRunner interface {
	// ScoreTypes returns the types of scores for a metric.
	ScoreTypes() []ScoreType
	// Distances returns the distances between a reference and a distortion, according to the score
	// types of this runner.
	Distances(reference, distortion string) ([]float64, error)
	// NeedsVolumeNormalization returns true if this metric requires volumes to be normalized before
	// Distances is called.
	NeedsVolumeNormalization() bool
}

// RingliPythonMetrics is a wrapper around a ringli python metrics binary.
type RingliPythonMetrics struct {
	AnalyzeBinaryPath string
}

// NeedsVolumeNormalization implements Metric.NeedsVolumeNormalization.
func (r RingliPythonMetrics) NeedsVolumeNormalization() bool {
	return true
}

// ScoreTypes implements Metric.ScoreType.
func (r RingliPythonMetrics) ScoreTypes() []ScoreType {
	return []ScoreType{VisqolResult, ParlaqResult, WarpQResult, GvpmosResult, DnsmosSignalResult, DnsmosBackgroundResult, DnsmosOverallResult}
}

type ringliPythonMetricOutput struct {
	DnsmosBak float64 `json:"dnsmos_bak"`
	DnsmosOvr float64 `json:"dnsmos_ovr"`
	DnsmosSig float64 `json:"dnsmos_sig"`
	Gvpmos    float64 `json:"gvpmos"`
	Parlaq    float64 `json:"parlaq"`
	Visqol    float64 `json:"visqol"`
	WarpQ     float64 `json:"warp-q"`
}

// Distances implements Metric.Distance.
func (r RingliPythonMetrics) Distances(reference, distortion string) ([]float64, error) {
	outputJSONFile, err := os.CreateTemp(os.TempDir(), "zimtohrli_listening_test.*.json")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(outputJSONFile.Name())
	outputJSONFile.Close()
	_, err = execute(r.AnalyzeBinaryPath, reference, distortion, outputJSONFile.Name(), "30")
	if err != nil {
		return nil, err
	}
	inputJSONFile, err := os.Open(outputJSONFile.Name())
	if err != nil {
		return nil, err
	}
	defer inputJSONFile.Close()
	output := &ringliPythonMetricOutput{}
	if err := json.NewDecoder(inputJSONFile).Decode(output); err != nil {
		return nil, err
	}
	return []float64{output.Visqol, output.Parlaq, output.WarpQ, output.Gvpmos, output.DnsmosSig, output.DnsmosBak, output.DnsmosOvr}, nil
}

// Zimtohrli is a wrapper around a zimtohrli compare binary.
type Zimtohrli struct {
	Goohrli *goohrli.Goohrli
}

// ReadWAV reads a WAV file and returns the samples and sample rate.
func ReadWAV(path string) ([][]float32, float64, error) {
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

// MustCopy copies a file to a temporary directory and returns the path to the copy.
func MustCopy(path string) string {
	newPath, err := os.CreateTemp(os.TempDir(), fmt.Sprintf("zimtohrli_listening_test.*%s", filepath.Ext(path)))
	if err != nil {
		log.Panic(err)
	}
	newPath.Close()
	if err := exec.Command("cp", path, newPath.Name()).Run(); err != nil {
		log.Panic(err)
	}
	return newPath.Name()
}

// NeedsVolumeNormalization implements Metric.NeedsVolumeNormalization.
func (z *Zimtohrli) NeedsVolumeNormalization() bool {
	return false
}

func max(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

// Distances implements Metric.Distance.
func (z *Zimtohrli) Distances(reference, distortion string) ([]float64, error) {
	ref, refSampleRate, err := ReadWAV(reference)
	if err != nil {
		return nil, err
	}
	refMaxAmplitude := float32(0.0)
	for channelIndex := 0; channelIndex < len(ref); channelIndex++ {
		measurement := goohrli.Measure(ref[channelIndex])
		refMaxAmplitude = max(refMaxAmplitude, measurement.MaxAbsAmplitude)
	}
	dist, distSampleRate, err := ReadWAV(distortion)
	if err != nil {
		return nil, err
	}
	if len(ref) != len(dist) {
		return nil, fmt.Errorf("reference %q has %v channels, while distortion %q has %v channels", reference, len(ref), distortion, len(dist))
	}
	if refSampleRate != distSampleRate || refSampleRate != z.Goohrli.SampleRate() {
		return nil, fmt.Errorf("reference %q has sample rate %v, distortion %q has sample rate %v, and Zimtohrli instance expects sample rate %v", MustCopy(reference), refSampleRate, MustCopy(distortion), distSampleRate, z.Goohrli.SampleRate())
	}
	sumOfSquares := 0.0
	for channelIndex := 0; channelIndex < len(ref); channelIndex++ {
		goohrli.NormalizeAmplitude(math.MaxFloat32, dist[channelIndex])
		channelScore := float64(z.Goohrli.Distance(ref[channelIndex], dist[channelIndex]))
		if math.IsNaN(channelScore) || math.IsInf(channelScore, 0) {
			return nil, fmt.Errorf("zimtohrli failed to compute reasonable distance between the %v samples of %q and %v samples of %q for channel %v: %v", len(ref[channelIndex]), MustCopy(reference), len(dist[channelIndex]), MustCopy(distortion), channelIndex, channelScore)
		}
		sumOfSquares += channelScore * channelScore
	}
	res := math.Sqrt(sumOfSquares)
	if math.IsNaN(res) || math.IsInf(res, 0) {
		return nil, fmt.Errorf("zimtohrli failed to compute reasonable distances between %q and %q: %v", MustCopy(reference), MustCopy(distortion), res)
	}
	return []float64{res}, nil
}

// ScoreTypes implements Metric.ScoreType.
func (z Zimtohrli) ScoreTypes() []ScoreType {
	return []ScoreType{ZimtohrliResult}
}

// PerReferenceCorrelation contains per reference correlation data.
type PerReferenceCorrelation struct {
	SpearmanMedian float64
	SpearmanMean   float64
	SpearmanStdDev float64
}

// Correlation contains correlation data for a metric.
type Correlation struct {
	ScoreType    ScoreType
	Spearman     float64
	PerReference *PerReferenceCorrelation `json:",omitempty"`
}

type metricAndOutput struct {
	metric MetricRunner
	output []*Dataset
}

type scoreJob struct {
	metric MetricRunner
	input  *Reference
	output []*Reference
}

// Calculate returns the datasets resulting from computing the provided metrics on this dataset.
// Unless nil, updateProgress will be called each time a unit of work is done.
func (d *Dataset) Calculate(metrics []MetricRunner, updateProgress func(submitted, completed int), maxWorkers int, progressDirectory string) (map[ScoreType]*Dataset, error) {
	metricsAndOutputs := []*metricAndOutput{}
	for _, metric := range metrics {
		outputs := []*Dataset{}
		for _, scoreType := range metric.ScoreTypes() {
			outputs = append(outputs, &Dataset{ScoreType: scoreType})
		}
		metricsAndOutputs = append(metricsAndOutputs, &metricAndOutput{
			metric: metric,
			output: outputs,
		})
	}

	pool := &worker.Pool[any]{
		Workers:  maxWorkers,
		OnChange: updateProgress,
	}

	for _, loopRef := range d.References {
		ref := loopRef
		for _, loopMetricAndOutput := range metricsAndOutputs {
			metricAndOutput := loopMetricAndOutput
			metricRefs := []*Reference{}
			for _, output := range metricAndOutput.output {
				metricRef := &Reference{}
				output.References = append(output.References, metricRef)
				metricRefs = append(metricRefs, metricRef)
			}
			pool.Submit(func(func(any)) error {
				return ref.Score(metricAndOutput.metric, metricRefs, pool, progressDirectory)
			})
		}
	}
	if err := pool.Error(); err != nil {
		return nil, err
	}

	res := map[ScoreType]*Dataset{}
	for _, metricAndOutput := range metricsAndOutputs {
		for _, metricDataset := range metricAndOutput.output {
			res[metricDataset.ScoreType] = metricDataset
		}
	}
	return res, nil
}

func median(floats []float64) float64 {
	sort.Sort(sort.Float64Slice(floats))
	return floats[len(floats)/2]
}

// Correlate returns the spearman correlation for the provided datasets.
func (d *Dataset) Correlate(metricDatasets map[ScoreType]*Dataset) ([]Correlation, error) {
	res := []Correlation{}
	for _, metricDataset := range metricDatasets {
		referenceSpearmans := []float64{}
		metricScores := []float64{}
		correlateScores := []float64{}
		for referenceIndex := range d.References {
			metricReference := metricDataset.References[referenceIndex]
			correlateReference := d.References[referenceIndex]
			metricReferenceScores := []float64{}
			correlateReferenceScores := []float64{}
			for distortionIndex := range metricReference.Distortions {
				metricDistortion := metricReference.Distortions[distortionIndex]
				correlateDistortion := correlateReference.Distortions[distortionIndex]
				metricReferenceScores = append(metricReferenceScores, metricDistortion.Score)
				correlateReferenceScores = append(correlateReferenceScores, correlateDistortion.Score)
				metricScores = append(metricScores, metricDistortion.Score)
				correlateScores = append(correlateScores, correlateDistortion.Score)
			}
			if len(metricReference.Distortions) > 1 {
				spearman, _ := onlinestats.Spearman(correlateReferenceScores, metricReferenceScores)
				referenceSpearmans = append(referenceSpearmans, spearman)
			}
		}
		spearman, _ := onlinestats.Spearman(correlateScores, metricScores)
		corr := Correlation{
			ScoreType: metricDataset.ScoreType,
			Spearman:  spearman,
		}
		if len(referenceSpearmans) > 0 {
			perReference := &PerReferenceCorrelation{}
			perReference.SpearmanMedian = median(referenceSpearmans)
			perReference.SpearmanMean = onlinestats.Mean(referenceSpearmans)
			perReference.SpearmanStdDev = onlinestats.SampleStddev(referenceSpearmans)
			corr.PerReference = perReference
		}
		res = append(res, corr)
	}
	return res, nil
}
