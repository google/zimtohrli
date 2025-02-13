// Copyright 2025 The Zimtohrli Authors. All Rights Reserved.
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

// odaq downloads the listening test at https://zenodo.org/records/13377284.
package main

import (
	"archive/zip"
	"encoding/xml"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/google/zimtohrli/go/aio"
	"github.com/google/zimtohrli/go/data"
	"github.com/google/zimtohrli/go/progress"
	"github.com/google/zimtohrli/go/worker"
)

func withZipFiles(u string, f func(count int, each func(yield func(root string, path string) error) error) error) error {
	res, err := http.Get(u)
	if err != nil {
		return fmt.Errorf("GETing %q: %v", u, err)
	}
	if res.StatusCode != 200 {
		return fmt.Errorf("GETing %q: %v", u, res.Status)
	}

	tmpFile, err := os.CreateTemp("", "odaq.*.zip")
	if err != nil {
		return fmt.Errorf("creating temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	if err := func() error {
		defer tmpFile.Close()
		bar := progress.New(u)
		bar.Update(int(res.ContentLength), 0, 0)
		defer bar.Finish()
		buf := make([]byte, 1024*1024)
		read := 0
		sum := 0
		for read, err = res.Body.Read(buf); err == nil || err == io.EOF; read, err = res.Body.Read(buf) {
			sum += read
			bar.Update(int(res.ContentLength), sum, 0)
			if _, err := tmpFile.Write(buf[:read]); err != nil {
				return fmt.Errorf("writing to %q: %v", tmpFile.Name(), err)
			}
			if err == io.EOF {
				break
			}
		}
		if err != io.EOF {
			return fmt.Errorf("reading %q: %v", u, err)
		}
		return nil
	}(); err != nil {
		return err
	}
	zipReader, err := zip.OpenReader(tmpFile.Name())
	if err != nil {
		return fmt.Errorf("reading %q: %v", tmpFile.Name(), err)
	}
	defer zipReader.Close()

	tmpDir, err := os.MkdirTemp("", "odaq.*")
	if err != nil {
		return fmt.Errorf("creating temp directory: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	return f(len(zipReader.File), func(yield func(root string, path string) error) error {
		for _, file := range zipReader.File {
			if file.FileInfo().IsDir() {
				continue
			}
			destPath := filepath.Join(tmpDir, file.Name)
			if err := os.MkdirAll(filepath.Dir(destPath), 0700); err != nil {
				return fmt.Errorf("creating directory %q: %v", filepath.Dir(destPath), err)
			}
			dest, err := os.Create(destPath)
			if err != nil {
				return fmt.Errorf("creating %q: %v", destPath, err)
			}
			if err := func() error {
				defer dest.Close()
				reader, err := file.Open()
				if err != nil {
					return fmt.Errorf("opening zip reader for %q: %v", file.Name, err)
				}
				defer reader.Close()
				if _, err := io.Copy(dest, reader); err != nil {
					return fmt.Errorf("copying zip reader for %q to %q: %v", file.Name, destPath, err)
				}
				return nil
			}(); err != nil {
				return fmt.Errorf("copying zip %q to %q: %v", file.Name, destPath, err)
			}
			if err := func() error {
				return yield(tmpDir, destPath)
			}(); err != nil {
				return err
			}
		}
		return nil
	})
}

type result struct {
	FileName string  `xml:"fileName,attr"`
	Score    float64 `xml:"score,attr"`
}

type trial struct {
	Name    string   `xml:"trialName,attr"`
	Results []result `xml:"testFile"`
}

type mushra struct {
	Trials []trial `xml:"trials>trial"`
}

func readMushraXML(path string) (*mushra, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	result := &mushra{}
	if err := xml.NewDecoder(f).Decode(result); err != nil {
		return nil, err
	}
	return result, nil
}

func populate(dest string, workers int) error {
	study, err := data.OpenStudy(dest)
	if err != nil {
		return err
	}
	defer study.Close()

	refToDistToScores := map[string]map[string][]float64{}

	appendTrials := func(xmlPath string) error {
		m, err := readMushraXML(xmlPath)
		if err != nil {
			return fmt.Errorf("reading MUSHRA XML from %q: %v", xmlPath, err)
		}
		for _, trial := range m.Trials {
			refPath := filepath.Join("ODAQ", "ODAQ_listening_test", trial.Name, "reference.wav")
			distToScores, found := refToDistToScores[refPath]
			if !found {
				distToScores = map[string][]float64{}
				refToDistToScores[refPath] = distToScores
			}
			for _, result := range trial.Results {
				if result.FileName != "reference.wav" {
					distPath := filepath.Join("ODAQ", "ODAQ_listening_test", trial.Name, result.FileName)
					distToScores[distPath] = append(distToScores[distPath], result.Score)
				}
			}
		}
		return nil
	}

	if err := withZipFiles("https://zenodo.org/records/13377284/files/ODAQ_v1_BSU.zip?download=1", func(count int, each func(yield func(root string, path string) error) error) error {
		return each(func(root string, zipPath string) error {
			if filepath.Ext(zipPath) == ".xml" {
				if err := appendTrials(zipPath); err != nil {
					return fmt.Errorf("trying to append trials from %q: %v", zipPath, err)
				}
			}
			return nil
		})
	}); err != nil {
		return err
	}

	recodedPaths := map[string]string{}
	recodedPathLock := sync.Mutex{}
	if err := withZipFiles("https://zenodo.org/records/10405774/files/ODAQ.zip?download=1", func(count int, each func(yield func(root string, path string) error) error) error {
		bar := progress.New("Converting")
		pool := worker.Pool[any]{
			Workers: workers,
			OnChange: func(submitted, completed, errors int) {
				bar.Update(count, completed, errors)
			},
		}
		bar.Update(count, 0, 0)
		if err := each(func(root string, zipPath string) error {
			if ext := filepath.Ext(zipPath); ext == ".xml" {
				if err := appendTrials(zipPath); err != nil {
					return fmt.Errorf("trying to append trials from %q: %v", zipPath, err)
				}
			} else if ext == ".wav" {
				pool.Submit(func(func(any)) error {
					recodedPath, err := aio.Recode(zipPath, dest)
					if err != nil {
						return err
					}
					rel, err := filepath.Rel(root, zipPath)
					if err != nil {
						return err
					}
					recodedPathLock.Lock()
					defer recodedPathLock.Unlock()
					recodedPaths[rel] = recodedPath
					return nil
				})
			}
			return nil
		}); err != nil {
			return err
		}
		if err := pool.Error(); err != nil {
			return err
		}
		bar.Finish()
		return nil
	}); err != nil {
		return err
	}

	references := []*data.Reference{}
	for refPath, distToScores := range refToDistToScores {
		recodePath, found := recodedPaths[refPath]
		if !found {
			return fmt.Errorf("recoded path for %q not found", refPath)
		}
		ref := &data.Reference{
			Name: refPath,
			Path: recodePath,
		}
		for distPath, scores := range distToScores {
			recodePath, found = recodedPaths[distPath]
			if !found {
				return fmt.Errorf("recoded path for %q not found", refPath)
			}
			sum := 0.0
			for _, score := range scores {
				sum += score
			}
			mean := sum / float64(len(scores))
			ref.Distortions = append(ref.Distortions, &data.Distortion{
				Name: distPath,
				Scores: map[data.ScoreType]float64{
					data.MOS: mean,
				},
				Path: recodePath,
			})
		}
		references = append(references, ref)
	}

	study.Put(references)

	return nil
}

func main() {
	destination := flag.String("dest", "", "Destination directory.")
	workers := flag.Int("workers", runtime.NumCPU(), "Number of sounds converted in parallel.")
	flag.Parse()
	if *destination == "" {
		flag.Usage()
		os.Exit(1)
	}

	if err := populate(*destination, *workers); err != nil {
		log.Fatal(err)
	}
}
