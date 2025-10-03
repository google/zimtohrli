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

// csv crates a Zimtohrli dataset based on a CSV file.
package main

import (
	"encoding/csv"
	"flag"
	"log"
	"os"
	"strconv"

	"github.com/google/zimtohrli/go/aio"
	"github.com/google/zimtohrli/go/data"
	"github.com/google/zimtohrli/go/progress"
	"github.com/google/zimtohrli/go/worker"
)

func populate(dst string, src string, refHeader string, distHeader string, mosHeader string, workers int) error {
	study, err := data.OpenStudy(dst)
	if err != nil {
		return err
	}
	defer study.Close()

	srcFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer srcFile.Close()

	csvReader := csv.NewReader(srcFile)
	headers, err := csvReader.Read()
	if err != nil {
		return err
	}
	indices := map[string]int{}
	for index, header := range headers {
		indices[header] = index
	}

	references := []*data.Reference{}
	err = nil
	bar := progress.New("Downloading")
	pool := worker.Pool[any]{
		Workers:  workers,
		OnChange: bar.Update,
	}
	defer bar.Finish()
	for records, err := csvReader.Read(); err == nil; records, err = csvReader.Read() {
		refURL := records[indices[refHeader]]
		distURL := records[indices[distHeader]]
		mos, err := strconv.ParseFloat(records[indices[mosHeader]], 64)
		if err != nil {
			return err
		}
		ref := &data.Reference{
			Name: refURL,
		}
		pool.Submit(func(func(any)) error {
			var err error
			ref.Path, err = aio.Fetch(ref.Name, dst)
			return err
		})
		pool.Submit(func(func(any)) error {
			dist := &data.Distortion{
				Name:   distURL,
				Scores: map[data.ScoreType]float64{},
			}
			pool.Submit(func(func(any)) error {
				var err error
				dist.Path, err = aio.Fetch(dist.Name, dst)
				return err
			})
			dist.Scores[data.MOS] = mos
			ref.Distortions = append(ref.Distortions, dist)
			return nil
		})
		references = append(references, ref)
	}
	if err := pool.Error(); err != nil {
		return err
	}
	if err := study.Put(references); err != nil {
		return err
	}
	return nil
}

func main() {
	destination := flag.String("dst", "", "Destination directory.")
	workers := flag.Int("workers", 1, "Number of workers downloading sounds.")
	source := flag.String("src", "", "Source CSV.")
	refHeader := flag.String("ref_header", "", "Header in the CSV with the URL to the reference file.")
	distHeader := flag.String("dist_header", "", "Header in the CSV with the URL to the distortion file.")
	mosHeader := flag.String("mos_header", "", "Header in the CSV with the MOS score.")
	flag.Parse()

	if *destination == "" || *source == "" || *refHeader == "" || *distHeader == "" || *mosHeader == "" {
		flag.Usage()
		os.Exit(1)
	}

	if err := populate(*destination, *source, *refHeader, *distHeader, *mosHeader, *workers); err != nil {
		log.Fatal(err)
	}
}
