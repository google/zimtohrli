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

// Package aio handles audio in/out.
package aio

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/google/zimtohrli/go/audio"
)

// Load loads audio from an ffmpeg-decodable file from a path (which may be a URL).
func Load(path string) (*audio.Audio, error) {
	cmd := exec.Command("ffmpeg", "-i", path, "-vn", "-c:a", "pcm_s16le", "-f", "wav", "-ar", "48000", "-")
	stdout, stderr := &bytes.Buffer{}, &bytes.Buffer{}
	cmd.Stdout, cmd.Stderr = stdout, stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("while executing %v: %v\n%v", cmd, err, stderr.String())
	}
	w, err := audio.ReadWAV(stdout)
	if err != nil {
		return nil, err
	}
	return w.Audio()
}

// Fetch calls Recode if path ends with .wav, otherwise Copy.
func Fetch(path string, dir string) (string, error) {
	if strings.ToLower(filepath.Ext(path)) == ".wav" {
		return Recode(path, dir)
	}
	return Copy(path, dir)
}

// Copy copies any file from a path (which may be a URL) and returns a path inside dir containing the file.
func Copy(path string, dir string) (string, error) {
	outFile, err := os.CreateTemp(dir, fmt.Sprintf("zimtohrli.go.io.Copy.*%s", filepath.Ext(path)))
	if err != nil {
		return "", err
	}
	outFile.Close()
	cmd := exec.Command("ffmpeg", "-y", "-i", path, "-vn", "-c:a", "copy", outFile.Name())
	ffmpegResult, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("trying to execute %v: %v\n%s", cmd, err, ffmpegResult)
	}
	return filepath.Rel(dir, outFile.Name())
}

// Recode copies an ffmpeg-decodable file from path (which may be a URL) and returns a path inside dir containing a FLAC encoded version of it.
func Recode(path string, dir string) (string, error) {
	flacFile, err := os.CreateTemp(dir, "zimtohrli.go.io.Recode.*.flac")
	if err != nil {
		return "", err
	}
	flacFile.Close()
	cmd := exec.Command("ffmpeg", "-y", "-i", path, "-vn", "-c:a", "flac", "-f", "flac", flacFile.Name())
	ffmpegResult, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("trying to execute %v: %v\n%s", cmd, err, ffmpegResult)
	}
	return filepath.Rel(dir, flacFile.Name())
}
