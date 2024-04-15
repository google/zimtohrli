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

// Package coresvnet provides functions to fetch the listening test at https://listening-test.coresv.net/results.htm.
package coresvnet

import (
	"fmt"
	"net/http"
	"net/url"
	"strconv"

	"github.com/PuerkitoBio/goquery"
	"github.com/google/zimtohrli/go/dataset"
)

// Fetch returns the metadata for the listening test at https://listening-test.coresv.net/results.htm.
func Fetch() (*dataset.Dataset, error) {
	data := &dataset.Dataset{
		ScoreType: dataset.Mos,
	}

	rootURL, err := url.Parse("https://listening-test.coresv.net/results.htm")
	if err != nil {
		return nil, err
	}
	res, err := http.Get(rootURL.String())
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		return nil, fmt.Errorf("status code error: %d %s", res.StatusCode, res.Status)
	}
	doc, err := goquery.NewDocumentFromReader(res.Body)
	if err != nil {
		return nil, err
	}
	resultTable := doc.Find("h2#list3:contains(\"All sets of tracks (5 sets, each 8 tracks)\")").Next().Find("table.table")

	formats := []dataset.Format{"", "", dataset.Opus, dataset.Aac, dataset.Ogg, dataset.Mp3, dataset.Faac, dataset.Faac}

	err = nil
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
		ref := &dataset.Reference{
			Name:     u.String(),
			Provider: dataset.URLPathProvider(u),
			Format:   dataset.Wav,
		}
		for columnIndex := 2; columnIndex < columns.Length(); columnIndex++ {
			u, parseErr := rootURL.Parse(columns.Eq(columnIndex).Find("a").AttrOr("href", ""))
			if parseErr != nil {
				err = parseErr
				return
			}
			dist := &dataset.Distortion{
				Name:     u.String(),
				Provider: dataset.URLPathProvider(u),
				Format:   formats[columnIndex],
			}
			score, parseErr := strconv.ParseFloat(columns.Eq(columnIndex).Text(), 64)
			if parseErr != nil {
				err = parseErr
				return
			}
			dist.Score = score
			ref.Distortions = append(ref.Distortions, dist)
		}
		data.References = append(data.References, ref)
	})
	return data, err
}
