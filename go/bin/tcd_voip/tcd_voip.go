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

// tcd_voip creates a study from https://qxlab.ucd.ie/index.php/tcd-voip-dataset/.
//
// It requires the user to open "TCD VOIP - Test Set Conditions and MOS Results.xlsx"
// in a compatible spreadsheet application and export the "Subjective Test Scores"
// tab to a CSV file.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"

	"github.com/google/zimtohrli/go/aio"
	"github.com/google/zimtohrli/go/data"
	"github.com/google/zimtohrli/go/progress"
	"github.com/google/zimtohrli/go/worker"
)

var fileReg = regexp.MustCompile("[^_]+(_[^_]+_([^_]+)_[^_]+.wav)")

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
	if len(csvFiles) != 1 {
		return fmt.Errorf("not exactly one .csv file in %q", source)
	}
	csvFile, err := os.Open(csvFiles[0])
	if err != nil {
		return err
	}
	defer csvFile.Close()
	csvReader := csv.NewReader(csvFile)
	header, err := csvReader.Read()
	if err != nil {
		return err
	}
	if strings.Join(header, ",") != "Filename,ConditionID,sample MOS,listener  # ->,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24" {
		return fmt.Errorf("header %+v doesn't match expected TCD VOIP header", header)
	}
	err = nil
	bar := progress.New("Transcoding")
	pool := worker.Pool[*data.Reference]{
		Workers:  workers,
		OnChange: bar.Update,
		FailFast: failFast,
	}
	var loopLine []string
	lineIndex := 0
	for loopLine, err = csvReader.Read(); err == nil; loopLine, err = csvReader.Read() {
		line := loopLine
		match := fileReg.FindStringSubmatch(line[0])
		if match == nil {
			return fmt.Errorf("line %+v doesn't have a file matching %v", line, fileReg)
		}
		distPath := filepath.Join(source, "Test Set", strings.ToLower(match[2]), line[0])
		if _, err := os.Stat(distPath); err != nil {
			return err
		}
		refPath := filepath.Join(source, "Test Set", strings.ToLower(match[2]), "ref", fmt.Sprintf("R%s", match[1]))
		if _, err := os.Stat(refPath); err != nil {
			return err
		}
		mos, err := strconv.ParseFloat(line[2], 64)
		if err != nil {
			return err
		}
		refIndex := lineIndex
		pool.Submit(func(f func(*data.Reference)) error {
			ref := &data.Reference{
				Name: fmt.Sprintf("ref-%v", refIndex),
			}
			var err error
			ref.Path, err = aio.Recode(refPath, dest)
			if err != nil {
				return fmt.Errorf("unable to fetch %q", refPath)
			}
			dist := &data.Distortion{
				Name: fmt.Sprintf("dist-%v", refIndex),
				Scores: map[data.ScoreType]float64{
					data.MOS: mos,
				},
			}
			dist.Path, err = aio.Recode(distPath, dest)
			if err != nil {
				return fmt.Errorf("unable to fetch %q", distPath)
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
	source := flag.String("source", "", "Directory containing the unpacked Dataset zip from https://qxlab.ucd.ie/index.php/tcd-voip-dataset/ along with a CSV export of the 'Subjective Test Scores' tab of 'TCD VOIP - Test Set Conditions and MOS Results.xlsx'.")
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
