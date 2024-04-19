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
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"path/filepath"
	"sort"

	"github.com/dgryski/go-onlinestats"
	"github.com/google/zimtohrli/go/aio"
	"github.com/google/zimtohrli/go/audio"
	"github.com/google/zimtohrli/go/worker"

	badger "github.com/dgraph-io/badger/v4"
)

const (
	MOS       ScoreType = "MOS"
	Zimtohrli ScoreType = "Zimtohrli"
)

// ScoreType represents a type of score, such as MOS or Zimtohrli.
type ScoreType string

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
	db  *badger.DB
}

// OpenStudy opens a study from a database directory.
// If the study doesn't exist, it will be created.
func OpenStudy(dir string) (*Study, error) {
	db, err := badger.Open(badger.DefaultOptions(filepath.Join(dir, "index.badgerdb")))
	if err != nil {
		return nil, err
	}
	return &Study{
		dir: dir,
		db:  db,
	}, nil
}

type CorrelationScore struct {
	ScoreTypeA ScoreType
	ScoreTypeB ScoreType
	Score      float64
}

type CorrelationTable [][]CorrelationScore

func (c CorrelationTable) String() string {
	result := Table{}
	header := Row{""}
	for _, score := range c[0] {
		header = append(header, string(score.ScoreTypeB))
	}
	result = append(result, header)
	for _, scores := range c {
		row := Row{string(scores[0].ScoreTypeA)}
		for _, score := range scores {
			row = append(row, fmt.Sprintf("%.2f", score.Score))
		}
		result = append(result, row)
	}
	return result.String(2)
}

// Correlate returns a table of all scores in the study Spearman correlated to each other.
func (s *Study) Correlate() (CorrelationTable, error) {
	scores := map[ScoreType][]float64{}
	if err := s.ViewEachReference(func(ref *Reference) error {
		for _, dist := range ref.Distortions {
			for scoreType, score := range dist.Scores {
				scores[scoreType] = append(scores[scoreType], score)
			}
		}
		return nil
	}); err != nil {
		return nil, err
	}
	sortedScoreTypes := ScoreTypes{}
	for scoreType := range scores {
		sortedScoreTypes = append(sortedScoreTypes, scoreType)
	}
	sort.Sort(sortedScoreTypes)
	result := CorrelationTable{}
	for _, scoreTypeA := range sortedScoreTypes {
		row := []CorrelationScore{}
		for _, scoreTypeB := range sortedScoreTypes {
			spearman, _ := onlinestats.Spearman(scores[scoreTypeA], scores[scoreTypeB])
			row = append(row, CorrelationScore{
				ScoreTypeA: scoreTypeA,
				ScoreTypeB: scoreTypeB,
				Score:      math.Abs(spearman),
			})
		}
		result = append(result, row)
	}
	return result, nil
}

// Measurement returns distance between sounds.
type Measurement func(reference, distortion *audio.Audio) (float64, error)

// Calculate computes measurements and populates the scores of the distortions.
func (s *Study) Calculate(measurements map[ScoreType]Measurement, pool *worker.Pool[any]) error {
	refs := []*Reference{}
	if err := s.ViewEachReference(func(ref *Reference) error {
		refs = append(refs, ref)
		return nil
	}); err != nil {
		return err
	}
	for _, loopRef := range refs {
		ref := loopRef
		pool.Submit(func(func(any)) error {
			refAudio, err := ref.Load(s.dir)
			if err != nil {
				log.Fatal(err)
			}
			for _, loopDist := range ref.Distortions {
				dist := loopDist
				pool.Submit(func(func(any)) error {
					distAudio, err := dist.Load(s.dir)
					if err != nil {
						return err
					}
					for loopScoreType := range measurements {
						scoreType := loopScoreType
						pool.Submit(func(func(any)) error {
							score, err := measurements[scoreType](refAudio, distAudio)
							if err != nil {
								return err
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
	if err := pool.Error(); err != nil {
		return err
	}
	for _, ref := range refs {
		if err := s.Put(ref); err != nil {
			return err
		}
	}
	return nil
}

// ViewEachReference returns each reference in the study.
func (s *Study) ViewEachReference(f func(*Reference) error) error {
	return s.db.View(func(txn *badger.Txn) error {
		iter := txn.NewIterator(badger.DefaultIteratorOptions)
		defer iter.Close()
		for iter.Rewind(); iter.Valid(); iter.Next() {
			if err := iter.Item().Value(func(value []byte) error {
				ref := &Reference{}
				if err := json.Unmarshal(value, ref); err != nil {
					return err
				}
				return f(ref)
			}); err == io.EOF {
				break
			} else if err != nil {
				return err
			}
		}
		return nil
	})
}

// UpdateEachReference returns each reference in the study and saves it after it's handled.
func (s *Study) UpdateEachReference(f func(*Reference) error) error {
	return s.db.Update(func(txn *badger.Txn) error {
		iter := txn.NewIterator(badger.DefaultIteratorOptions)
		defer iter.Close()
		for iter.Rewind(); iter.Valid(); iter.Next() {
			if err := iter.Item().Value(func(value []byte) error {
				ref := &Reference{}
				if err := json.Unmarshal(value, ref); err != nil {
					return err
				}
				if err := f(ref); err != nil {
					return err
				}
				b, err := json.Marshal(ref)
				if err != nil {
					return err
				}
				return txn.Set(iter.Item().Key(), b)
			}); err == io.EOF {
				break
			} else if err != nil {
				return err
			}
		}
		return nil
	})
}

// Put inserts a reference into a study.
func (s *Study) Put(ref *Reference) error {
	b, err := json.Marshal(ref)
	if err != nil {
		return err
	}
	if err := s.db.Update(func(txn *badger.Txn) error {
		return txn.Set([]byte(ref.Name), b)
	}); err != nil {
		return err
	}
	return nil
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
