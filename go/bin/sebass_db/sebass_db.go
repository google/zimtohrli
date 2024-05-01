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

// sebass_db creates a study from https://www.audiolabs-erlangen.de/resources/2019-WASPAA-SEBASS/.
//
// It currently supports SASSEC, SiSEC08, SAOC, and PEASS-DB. SiSEC18 at
// that web site doesn't contain all audio, and is not currently supported.
//
// Download and unpac one of the supported ZIP files, and use the directory
// it unpacked as -source when running this binary.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"

	"github.com/google/zimtohrli/go/aio"
	"github.com/google/zimtohrli/go/data"
	"github.com/google/zimtohrli/go/progress"
	"github.com/google/zimtohrli/go/worker"
)

func populate(source string, dest string, workers int, failFast bool) error {
	study, err := data.OpenStudy(dest)
	if err != nil {
		return err
	}
	defer study.Close()

	csvFiles, err := filepath.Glob(filepath.Join(source, "*.csv"))
	if err != nil {
		return err
	}
	for _, csvFile := range csvFiles {
		signals := "Signals"
		switch filepath.Base(csvFile) {
		case "SAOC_1_anonymized.csv":
			signals = "Signals_1"
		case "SAOC_2_anonymized.csv":
			signals = "Signals_2"
		case "SAOC_3_anonymized.csv":
			signals = "Signals_3"
		}
		fileReader, err := os.Open(csvFile)
		if err != nil {
			return err
		}
		defer fileReader.Close()
		csvReader := csv.NewReader(fileReader)
		header, err := csvReader.Read()
		if err != nil {
			return err
		}
		if !reflect.DeepEqual(header, []string{"Testname", "Listener", "Trial", "Condition", "Ratingscore"}) {
			return fmt.Errorf("header %+v doesn't match expected SEBASS-DB header", header)
		}
		err = nil
		bar := progress.New(fmt.Sprintf("Transcoding from %q", csvFile))
		pool := worker.Pool[*data.Reference]{
			Workers:  workers,
			OnChange: bar.Update,
			FailFast: failFast,
		}
		var loopLine []string
		lineIndex := 0
		for loopLine, err = csvReader.Read(); err == nil; loopLine, err = csvReader.Read() {
			line := loopLine
			if len(line) == 0 {
				continue
			}
			if line[3] == "anchor" {
				line[3] = "anker_mix"
			}
			if line[3] == "hidden_ref" {
				line[3] = "orig"
			}
			if line[3] == "SAOC" {
				continue
			}
			mos, err := strconv.ParseFloat(line[4], 64)
			if err != nil {
				return err
			}
			if math.IsNaN(mos) {
				continue
			}
			refIndex := lineIndex
			pool.Submit(func(f func(*data.Reference)) error {
				ref := &data.Reference{
					Name: fmt.Sprintf("ref-%v", refIndex),
				}
				var err error
				path := filepath.Join(source, signals, "orig", fmt.Sprintf("%s.wav", line[2]))
				ref.Path, err = aio.Recode(path, dest)
				if err != nil {
					return fmt.Errorf("unable to fetch %q", path)
				}
				dist := &data.Distortion{
					Name: fmt.Sprintf("dist-%v", refIndex),
					Scores: map[data.ScoreType]float64{
						data.MOS: mos,
					},
				}
				path = filepath.Join(source, signals, line[3], fmt.Sprintf("%s.wav", line[2]))
				dist.Path, err = aio.Recode(path, dest)
				if err != nil {
					return fmt.Errorf("unable to fetch %q", path)
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
	}
	return nil
}

func main() {
	source := flag.String("source", "", "Directory containing one of the unpacked datasets from https://www.audiolabs-erlangen.de/resources/2019-WASPAA-SEBASS/.")
	destination := flag.String("dest", "", "Destination directory.")
	workers := flag.Int("workers", runtime.NumCPU(), "Number of workers transcoding sounds.")
	failFast := flag.Bool("fail_fast", false, "Whether to exit immediately at the first error.")
	flag.Parse()
	if *source == "" || *destination == "" {
		flag.Usage()
		os.Exit(1)
	}

	if err := populate(*source, *destination, *workers, *failFast); err != nil {
		log.Fatal(err)
	}
}
