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
	"math"
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
	return &Bar{
		name:       name,
		lastRender: time.Now(),
	}
}

// Bar contains state for a progress bar.
type Bar struct {
	name       string
	completed  int
	total      int
	emaSpeed   float64
	lastRender time.Time
}

// AddCompleted adds completed tasks to the bar and renders it.
func (b *Bar) AddCompleted(num int) {
	b.Update(b.completed+num, b.total)
}

// Update update completed and total tasks to the bar and updates it.
func (b *Bar) Update(completed, total int) {
	prefix := fmt.Sprintf("%s, %d/%d ", b.name, completed, total)

	now := time.Now()
	timeUsed := now.Sub(b.lastRender)
	currentSpeed := float64(completed-b.completed) / float64(timeUsed)
	minutesUsed := float64(timeUsed) / float64(time.Minute)
	smoothingM1 := math.Pow(0.5, minutesUsed*0.1)
	if smoothingM1 > 0.999 {
		smoothingM1 = 0.999
	}
	b.emaSpeed = (1-smoothingM1)*currentSpeed + smoothingM1*b.emaSpeed
	suffix := fmt.Sprintf(" %.2f/s ETA: %s", b.emaSpeed*float64(time.Second), time.Duration(float64(b.total-b.completed)/b.emaSpeed))

	b.completed = completed
	b.total = total
	b.lastRender = now

	width, err := getTerminalWidth()
	if err != nil {
		fmt.Printf("\r%s", err)
		return
	}
	numFiller := width - len(prefix) - len(suffix)
	doneFiller := int(float64(numFiller) * float64(b.completed) / float64(b.total))
	filler := &bytes.Buffer{}
	for i := 0; i < numFiller; i++ {
		if i < doneFiller {
			fmt.Fprintf(filler, "#")
		} else {
			fmt.Fprintf(filler, " ")
		}
	}
	fmt.Printf("\r%s%s%s", prefix, filler, suffix)
}
