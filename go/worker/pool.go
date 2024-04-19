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

// Package worker contains functionality to parallelize tasks with a pool of workers.
package worker

import (
	"fmt"
	"sync"
	"sync/atomic"
)

// Pool is a pool of workers.
type Pool[T any] struct {
	Workers  int
	OnChange func(submitted, completed, errors int)

	startOnce sync.Once

	jobs             chan func(func(T)) error
	jobsWaitGroup    sync.WaitGroup
	results          chan T
	resultsWaitGroup sync.WaitGroup
	errors           chan error
	errorsWaitGroup  sync.WaitGroup

	submittedJobs uint32
	completedJobs uint32
	errorJobs     uint32
}

func (p *Pool[T]) init() {
	p.startOnce.Do(func() {
		p.jobs = make(chan func(func(T)) error)
		p.results = make(chan T)
		p.errors = make(chan error)
		for i := 0; i < p.Workers; i++ {
			go func() {
				for job := range p.jobs {
					if err := job(func(t T) {
						p.resultsWaitGroup.Add(1)
						go func() {
							p.results <- t
							p.resultsWaitGroup.Done()
						}()
					}); err != nil {
						p.errorsWaitGroup.Add(1)
						go func() {
							p.errors <- err
							p.errorsWaitGroup.Done()
						}()
						atomic.AddUint32(&p.errorJobs, 1)
						p.change()
					}
					p.jobsWaitGroup.Done()
					atomic.AddUint32(&p.completedJobs, 1)
					p.change()
				}
			}()
		}
	})
}

func (p *Pool[T]) change() {
	if p.OnChange != nil {
		p.OnChange(int(atomic.LoadUint32(&p.submittedJobs)), int(atomic.LoadUint32(&p.completedJobs)), int(atomic.LoadUint32(&p.errorJobs)))
	}
}

// Submit submits a job to the pool.
func (p *Pool[T]) Submit(job func(func(T)) error) error {
	p.init()

	p.jobsWaitGroup.Add(1)
	atomic.AddUint32(&p.submittedJobs, 1)
	p.change()

	go func() {
		p.jobs <- job
	}()
	return nil
}

// Error waits for all submitted jobs to finish, closes the submission channel, and returns whether
// any of the jobs produced an error.
//
// Must be called after all jobs are added.
func (p *Pool[T]) Error() error {
	p.init()

	p.jobsWaitGroup.Wait()
	close(p.jobs)
	go func() {
		p.errorsWaitGroup.Wait()
		close(p.errors)
	}()
	result := []error{}
	for err := range p.errors {
		result = append(result, err)
	}
	if len(result) > 0 {
		return fmt.Errorf("%+v", result)
	}
	return nil
}

// Results returns all results produced. The result channel will close once all results are processed.
//
// Must be called if any jobs might have produced results.
//
// Error() must be called before Results().
func (p *Pool[T]) Results() <-chan T {
	go func() {
		p.resultsWaitGroup.Wait()
		close(p.results)
	}()
	return p.results
}
