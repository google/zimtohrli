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

// Package pipe manages services communicating via pipes.
package pipe

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"

	"github.com/google/zimtohrli/go/aio"
	"github.com/google/zimtohrli/go/audio"
	"github.com/google/zimtohrli/go/data"
	"github.com/google/zimtohrli/go/resource"
)

// MeterPool contains a resource pool of pipe-communicating processes.
type MeterPool struct {
	*resource.Pool[*Metric]

	ScoreType data.ScoreType
}

// NewMeterPool returns a new pool of pipe-communicating processes.
func NewMeterPool(path string) (*MeterPool, error) {
	result := &MeterPool{
		Pool: &resource.Pool[*Metric]{
			Create: func() (*Metric, error) { return StartMetric(path) },
		},
	}
	metric, err := result.Get()
	if err != nil {
		return nil, err
	}
	defer result.Pool.Return(metric)
	result.ScoreType, err = metric.ScoreType()
	return result, err
}

// Close closes all the processes in the pool.
func (m *MeterPool) Close() error {
	return m.Pool.Close()
}

// Measure returns the distance between ref and dist using a metric in the pool, and then returns it to the pool.
func (m *MeterPool) Measure(ref, dist *audio.Audio) (float64, error) {
	metric, err := m.Pool.Get()
	if err != nil {
		return 0, err
	}
	result, err := metric.Measure(ref, dist)
	if err != nil {
		return 0, err
	}
	m.Pool.Return(metric)
	return result, nil
}

// Metric wraps a pipe-communicating process.
type Metric struct {
	scoreType data.ScoreType
	stdin     io.WriteCloser
	stdout    *bufio.Reader
	stderr    *bytes.Buffer
	nextLine  string
}

// StartMetric starts a new pipe-communicating process.
func StartMetric(path string) (*Metric, error) {
	cmd := exec.Command(path)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("creating stdin pipe for %v: %v", cmd, err)
	}
	stderr := &bytes.Buffer{}
	cmd.Stderr = stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("creating stdout pipe for %v: %v", cmd, err)
	}
	m := &Metric{
		stdin:  stdin,
		stderr: stderr,
		stdout: bufio.NewReader(stdout),
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("running %v: %v\n%s", cmd, err, stderr)
	}
	return m, nil
}

func (m *Metric) awaitReady() error {
	if m.scoreType != "" {
		return nil
	}
	var err error
	for ; err == nil && !strings.HasPrefix(m.nextLine, "READY:"); m.nextLine, err = m.stdout.ReadString('\n') {
	}
	if err != nil {
		return fmt.Errorf("waiting for READY: %v\n%s", err, m.stderr)
	}
	scoreType, found := strings.CutPrefix(strings.TrimSpace(m.nextLine), "READY:")
	if !found {
		return fmt.Errorf("%q doesn't have the prefix 'READY:'", m.nextLine)
	}
	m.scoreType = data.ScoreType(scoreType)
	return nil
}

// ScoreType waits for the process to emit it's score type and returns it.
func (m *Metric) ScoreType() (data.ScoreType, error) {
	if err := m.awaitReady(); err != nil {
		return "", err
	}
	return m.scoreType, nil
}

func (m *Metric) await(msg string) error {
	var err error
	for ; err == nil && strings.TrimSpace(m.nextLine) != msg; m.nextLine, err = m.stdout.ReadString('\n') {
	}
	if err != nil {
		return fmt.Errorf("waiting for %q: %v\n%s", msg, err, m.stderr)
	}
	return nil
}

// Measure waits until the process has emitted it's score type (which signals that it's ready) and returns the score for the provided ref and dist.
func (m *Metric) Measure(ref, dist *audio.Audio) (float64, error) {
	if err := m.awaitReady(); err != nil {
		return 0, err
	}
	refPath, err := aio.DumpWAV(ref)
	if err != nil {
		return 0, fmt.Errorf("dumping referenc audio: %v", err)
	}
	defer os.RemoveAll(refPath)
	distPath, err := aio.DumpWAV(dist)
	if err != nil {
		return 0, fmt.Errorf("dumping distortion audio: %v", err)
	}
	defer os.RemoveAll(distPath)
	if err := m.await("REF"); err != nil {
		return 0, err
	}
	if _, err := fmt.Fprintln(m.stdin, refPath); err != nil {
		return 0, fmt.Errorf("printing ref path: %v\n%s", err, m.stderr)
	}
	if err := m.await("DIST"); err != nil {
		return 0, err
	}
	if _, err := fmt.Fprintln(m.stdin, distPath); err != nil {
		return 0, fmt.Errorf("printing dist path: %v\n%s", err, m.stderr)
	}
	for ; err == nil && !strings.HasPrefix(m.nextLine, "SCORE="); m.nextLine, err = m.stdout.ReadString('\n') {
	}
	if err != nil {
		return 0, fmt.Errorf("waiting for SCORE=: %v\n%s", err, m.stderr)
	}
	scoreString, found := strings.CutPrefix(strings.TrimSpace(m.nextLine), "SCORE=")
	if !found {
		return 0, fmt.Errorf("%q doesn't have the prefix SCORE=", m.nextLine)
	}
	return strconv.ParseFloat(scoreString, 64)
}

// Close closes the process by closing it's stdin.
func (m *Metric) Close() error {
	return m.stdin.Close()
}
