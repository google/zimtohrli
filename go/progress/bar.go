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

// Package progress paints a very simple progress bar on the screen
package progress

import (
	"bytes"
	"fmt"
	"log"
	"math"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

type winsize struct {
	Row    uint16
	Col    uint16
	Xpixel uint16
	Ypixel uint16
}

func getTerminalWidth() (int, error) {
	ws := &winsize{}
	retCode, _, errno := syscall.Syscall(syscall.SYS_IOCTL,
		uintptr(syscall.Stdin),
		uintptr(syscall.TIOCGWINSZ),
		uintptr(unsafe.Pointer(ws)))

	if int(retCode) == -1 {
		return 0, fmt.Errorf("Syscall returned %v", errno)
	}
	return int(ws.Col), nil
}

// New returns a new progress bar.
func New(name string) *Bar {
	now := time.Now()
	return &Bar{
		name:       name,
		created:    now,
		lastRender: now,
	}
}

// Bar contains state for a progress bar.
type Bar struct {
	name              string
	created           time.Time
	completed         int
	errors            int
	total             int
	emaCompletedSpeed float64
	emaFractionSpeed  float64
	lastRender        time.Time
	lock              sync.Mutex
}

// AddCompleted adds completed tasks to the bar and renders it.
func (b *Bar) AddCompleted(num int) {
	b.Update(b.total, b.completed+num, b.errors)
}

// Finish prints the final actual time of completion and a newline.
func (b *Bar) Finish() {
	prefix := fmt.Sprintf("%s, %d/%d/%d ", b.name, b.completed, b.errors, b.total)
	atc := time.Since(b.created)
	speed := float64(b.completed) / float64(atc)
	round := time.Minute
	if atc < time.Minute {
		round = time.Second
	}
	suffix := fmt.Sprintf(" %.2f/s ATC: %s", speed*float64(time.Second), atc.Round(round))

	fmt.Printf("\r%s%s%s\n", prefix, b.filler(prefix, suffix), suffix)
}

func (b *Bar) filler(prefix, suffix string) string {
	width, err := getTerminalWidth()
	if err != nil {
		log.Println(err)
		return ""
	}
	numFiller := width - len(prefix) - len(suffix)
	completedFiller := int(float64(numFiller) * float64(b.completed) / float64(b.total))
	errorFiller := int(float64(numFiller) * float64(b.errors) / float64(b.total))
	filler := &bytes.Buffer{}
	for i := 0; i < numFiller; i++ {
		if i < completedFiller {
			fmt.Fprintf(filler, "#")
		} else if i < completedFiller+errorFiller {
			fmt.Fprintf(filler, "☠")
		} else {
			fmt.Fprintf(filler, " ")
		}
	}
	return filler.String()
}

// Update update completed and total tasks to the bar and updates it.
func (b *Bar) Update(total, completed, errors int) {
	b.lock.Lock()
	defer b.lock.Unlock()

	prefix := fmt.Sprintf("%s, %d/%d/%d ", b.name, completed, errors, total)

	now := time.Now()
	fraction := float64(completed) / float64(total)
	if timeLived := now.Sub(b.created); timeLived < 10*time.Second {
		b.emaCompletedSpeed = float64(completed) / float64(timeLived)
		b.emaFractionSpeed = fraction / float64(timeLived)
	} else {
		timeUsed := now.Sub(b.lastRender)
		currentCompletedSpeed := float64(completed-b.completed) / float64(timeUsed)
		currentFractionSpeed := (fraction - (float64(b.completed) / float64(b.total))) / float64(timeUsed)
		minutesUsed := float64(timeUsed) / float64(time.Minute)
		smoothingM1 := math.Pow(0.5, minutesUsed)
		b.emaCompletedSpeed = (1-smoothingM1)*currentCompletedSpeed + smoothingM1*b.emaCompletedSpeed
		b.emaFractionSpeed = (1-smoothingM1)*currentFractionSpeed + smoothingM1*b.emaFractionSpeed
	}
	eta := time.Duration((1 - fraction) / b.emaFractionSpeed)
	round := time.Minute
	if eta < time.Minute {
		round = time.Second
	}
	suffix := fmt.Sprintf(" %.2f/s ETA: %s", b.emaCompletedSpeed*float64(time.Second), eta.Round(round))

	b.completed = completed
	b.errors = errors
	b.total = total
	b.lastRender = now

	fmt.Printf("\r%s%s%s", prefix, b.filler(prefix, suffix), suffix)
}
