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

// Package resource contains tools to handle resource pools.
package resource

import (
	"sync"

	"github.com/google/zimtohrli/go/worker"
)

// Closer can close.
type Closer interface {
	Close() error
}

// Pool produces, keeps tracks of, and releases resources.
type Pool[T Closer] struct {
	Create func() (T, error)

	lock      sync.Mutex
	sequence  int
	resources map[int]T
}

// Get returns a resource, which might create a new one.
func (p *Pool[T]) Get() (T, error) {
	p.lock.Lock()
	defer p.lock.Unlock()
	if p.resources == nil {
		p.resources = map[int]T{}
	}
	for id, res := range p.resources {
		delete(p.resources, id)
		return res, nil
	}
	return p.Create()
}

// Return returns a resource to the pool.
func (p *Pool[T]) Return(t T) {
	p.lock.Lock()
	defer p.lock.Unlock()
	if p.resources == nil {
		p.resources = map[int]T{}
	}
	p.sequence++
	p.resources[p.sequence] = t
}

// Close releases all the resources of the pool.
func (p *Pool[T]) Close() error {
	p.lock.Lock()
	defer p.lock.Unlock()
	errs := worker.Errors{}
	for _, res := range p.resources {
		if err := res.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return errs
	}
	return nil
}
