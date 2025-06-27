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

// perceptual_audio creates a study from https://github.com/pranaymanocha/PerceptualAudio/blob/master/dataset/README.md.
//
// Download and unpack the dataset ZIP, and provide the unpacked directory
// as -source when running this binary.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/google/zimtohrli/go/aio"
	"github.com/google/zimtohrli/go/data"
	"github.com/google/zimtohrli/go/progress"
	"github.com/google/zimtohrli/go/worker"
)

func populate(source string, dest string, workers int) error {
	study, err := data.OpenStudy(dest)
	if err != nil {
		return err
	}
	defer study.Close()

	csvURL, err := url.Parse("https://raw.githubusercontent.com/pranaymanocha/PerceptualAudio/master/dataset/dataset_combined.txt")
	if err != nil {
		return err
	}
	res, err := http.Get(csvURL.String())
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		return fmt.Errorf("status code error: %d %s", res.StatusCode, res.Status)
	}

	lineReader := bufio.NewReader(res.Body)
	err = nil
	bar := progress.New("Transcoding")
	pool := worker.Pool[*data.Reference]{
		Workers:  workers,
		OnChange: bar.Update,
	}
	line := ""
	lineIndex := 0
	for line, err = lineReader.ReadString('\n'); err == nil; line, err = lineReader.ReadString('\n') {
		fields := strings.Split(strings.TrimSpace(line), "\t")
		jnd, err := strconv.ParseFloat(fields[2], 64)
		if err != nil {
			return err
		}
		refIndex := lineIndex
		pool.Submit(func(f func(*data.Reference)) error {
			ref := &data.Reference{
				Name: fmt.Sprintf("ref-%v", refIndex),
			}
			var err error
			ref.Path, err = aio.Fetch(filepath.Join(source, fields[0]), dest)
			if err != nil {
				return fmt.Errorf("unable to fetch %q", fields[0])
			}
			dist := &data.Distortion{
				Name: fmt.Sprintf("dist-%v", refIndex),
				Scores: map[data.ScoreType]float64{
					data.JND: jnd,
				},
			}
			dist.Path, err = aio.Fetch(filepath.Join(source, fields[1]), dest)
			if err != nil {
				return fmt.Errorf("unable to fetch %q", fields[1])
			}
			ref.Distortions = append(ref.Distortions, dist)
			f(ref)
			return nil
		})
		lineIndex++
	}
	if err := pool.Error(); err != nil {
		log.Println(err.Error())
	}
	bar.Finish()
	refs := []*data.Reference{}
	for ref := range pool.Results() {
		refs = append(refs, ref)
	}
	if err := study.Put(refs); err != nil {
		return err
	}
	return nil
}

func main() {
	source := flag.String("source", "", "Directory containing the unpacked http://percepaudio.cs.princeton.edu/icassp2020_perceptual/audio_perception.zip.")
	destination := flag.String("dest", "", "Destination directory.")
	workers := flag.Int("workers", runtime.NumCPU(), "Number of workers transcoding sounds.")
	flag.Parse()
	if *source == "" || *destination == "" {
		flag.Usage()
		os.Exit(1)
	}

	if err := populate(*source, *destination, *workers); err != nil {
		log.Fatal(err)
	}
}
