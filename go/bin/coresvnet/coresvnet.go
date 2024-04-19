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

// coresvnet downloads the listening test at https://listening-test.coresv.net/results.htm.
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"strconv"

	"github.com/PuerkitoBio/goquery"
	"github.com/google/zimtohrli/go/aio"
	"github.com/google/zimtohrli/go/data"
	"github.com/google/zimtohrli/go/progress"
	"github.com/google/zimtohrli/go/worker"
)

func populate(dest string, workers int) error {
	study, err := data.OpenStudy(dest)
	if err != nil {
		return err
	}

	rootURL, err := url.Parse("https://listening-test.coresv.net/results.htm")
	if err != nil {
		return err
	}
	res, err := http.Get(rootURL.String())
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		return fmt.Errorf("status code error: %d %s", res.StatusCode, res.Status)
	}
	doc, err := goquery.NewDocumentFromReader(res.Body)
	if err != nil {
		return err
	}
	resultTable := doc.Find("h2#list3:contains(\"All sets of tracks (5 sets, each 8 tracks)\")").Next().Find("table.table")

	references := []*data.Reference{}
	err = nil
	bar := progress.New("Downloading")
	pool := worker.Pool[any]{
		Workers:  workers,
		OnChange: bar.Update,
	}
	resultTable.Find("tbody > tr").Each(func(index int, sel *goquery.Selection) {
		columns := sel.Find("td")
		if columns.Length() != 8 {
			return
		}
		u, parseErr := rootURL.Parse(columns.Eq(0).Find("a").AttrOr("href", ""))
		if parseErr != nil {
			err = parseErr
			return
		}
		ref := &data.Reference{
			Name: u.String(),
		}
		pool.Submit(func(func(any)) error {
			var err error
			ref.Path, err = aio.Fetch(ref.Name, dest)
			return err
		})
		for columnIndex := 2; columnIndex < columns.Length(); columnIndex++ {
			u, parseErr := rootURL.Parse(columns.Eq(columnIndex).Find("a").AttrOr("href", ""))
			if parseErr != nil {
				err = parseErr
				return
			}
			dist := &data.Distortion{
				Name:   u.String(),
				Scores: map[data.ScoreType]float64{},
			}
			pool.Submit(func(func(any)) error {
				var err error
				dist.Path, err = aio.Fetch(dist.Name, dest)
				return err
			})
			score, parseErr := strconv.ParseFloat(columns.Eq(columnIndex).Text(), 64)
			if parseErr != nil {
				err = parseErr
				return
			}
			dist.Scores[data.MOS] = score
			ref.Distortions = append(ref.Distortions, dist)
		}
		references = append(references, ref)
	})
	if err := pool.Error(); err != nil {
		return err
	}
	for _, ref := range references {
		if err := study.Put(ref); err != nil {
			return err
		}
	}
	fmt.Println()
	return nil
}

func main() {
	destination := flag.String("dest", "", "Destination directory.")
	workers := flag.Int("workers", 1, "Number of workers downloading sounds.")
	flag.Parse()
	if *destination == "" {
		flag.Usage()
		os.Exit(1)
	}

	if err := populate(*destination, *workers); err != nil {
		log.Fatal(err)
	}
}
