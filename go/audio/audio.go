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

// Package audio handles WAV data.
package audio

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"unsafe"
)

// FixString contains 4 bytes that can render to a String.
type FixString [4]byte

func (f FixString) String() string {
	return string(f[:])
}

// RIFFHeader contains a RIFF header.
type RIFFHeader struct {
	ChunkID   FixString
	ChunkSize int32
	Format    FixString
}

// ReadRIFFHeader reads a RIFF header.
func ReadRIFFHeader(r io.Reader) (*RIFFHeader, error) {
	result := &RIFFHeader{}
	if err := binary.Read(r, binary.LittleEndian, result); err != nil {
		return nil, err
	}
	return result, nil
}

// ChunkHeader contains a chunk header.
type ChunkHeader struct {
	SubChunkID   FixString
	SubChunkSize int32
}

// ReadChunkHeader reads a chunk header.
func ReadChunkHeader(r io.Reader) (*ChunkHeader, error) {
	result := &ChunkHeader{}
	if err := binary.Read(r, binary.LittleEndian, result); err != nil {
		return nil, err
	}
	return result, nil
}

// FormatChunk contains a format chunk.
type FormatChunk struct {
	AudioFormat   uint16
	NumChannels   int16
	SampleRate    int32
	ByteRate      int32
	BlockAlign    int16
	BitsPerSample int16
}

// ReadFormatChunk reads a format chunk.
func ReadFormatChunk(r io.Reader) (*FormatChunk, error) {
	result := &FormatChunk{}
	if err := binary.Read(r, binary.LittleEndian, result); err != nil {
		return nil, err
	}
	return result, nil
}

// WAV contains a WAV file.
type WAV struct {
	RIFFHeader  *RIFFHeader
	FormatChunk *FormatChunk
	Data        []byte
}

// Audio contains audio data.
type Audio struct {
	// Samples is a (num_channels, num_samples)-shaped array containing samples between -1 and 1.
	Samples [][]float32
	// Rate is the sample rate of the sound.
	Rate float64
	// MaxAbsAmplitude contains the max amplitude of the sound.
	MaxAbsAmplitude float32
}

// Amplify multiplies all samples in the audio with the amplification.
func (a *Audio) Amplify(amplification float32) {
	for _, channel := range a.Samples {
		for sampleIndex := range channel {
			channel[sampleIndex] *= amplification
		}
	}
	a.MaxAbsAmplitude *= float32(math.Abs(float64(amplification)))
}

// Audio returns the audio in a WAV file.
func (w *WAV) Audio() (*Audio, error) {
	result := &Audio{
		Samples: make([][]float32, w.FormatChunk.NumChannels),
		Rate:    float64(w.FormatChunk.SampleRate),
	}
	if w.FormatChunk.AudioFormat == 1 {
		pcmSamples := unsafe.Slice((*int16)(unsafe.Pointer(&w.Data[0])), len(w.Data)/2)
		numFrames := len(pcmSamples) / int(w.FormatChunk.NumChannels)
		for channelIndex := 0; channelIndex < int(w.FormatChunk.NumChannels); channelIndex++ {
			result.Samples[channelIndex] = make([]float32, numFrames)
		}
		scaleReciprocal := 1.0 / float32(int(1)<<(w.FormatChunk.BitsPerSample-1))
		for sampleIndex, sample := range pcmSamples {
			scaledSample := float32(sample) * scaleReciprocal
			absSample := scaledSample
			if absSample < 0 {
				absSample = -absSample
			}
			if absSample > result.MaxAbsAmplitude {
				result.MaxAbsAmplitude = absSample
			}
			result.Samples[sampleIndex%int(w.FormatChunk.NumChannels)][sampleIndex/int(w.FormatChunk.NumChannels)] = scaledSample
		}
	} else if w.FormatChunk.AudioFormat == 3 {
		return nil, fmt.Errorf("blah")
	} else {
		return nil, fmt.Errorf("not audio format 1 (PCM) or 3 (IEEE float)")
	}
	return result, nil
}

// ReadWAV reads a WAV file from a reader.
func ReadWAV(r io.Reader) (*WAV, error) {
	result := &WAV{}
	var err error
	if result.RIFFHeader, err = ReadRIFFHeader(r); err != nil {
		return nil, fmt.Errorf("while reading RIFF header: %v", err)
	}
	if result.RIFFHeader.ChunkID.String() != "RIFF" {
		return nil, fmt.Errorf("not RIFF")
	}
	if result.RIFFHeader.Format.String() != "WAVE" {
		return nil, fmt.Errorf("not WAVE")
	}
	for {
		chunkHeader, err := ReadChunkHeader(r)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("while reading chunk header: %v", err)
		}
		if chunkHeader.SubChunkID.String() == "fmt " {
			if result.FormatChunk, err = ReadFormatChunk(r); err != nil {
				return nil, fmt.Errorf("while reading format chunk: %v", err)
			}
			if result.FormatChunk.AudioFormat != 1 {
				return nil, fmt.Errorf("not audio format 1 (PCM): %x", result.FormatChunk.AudioFormat)
			}
			if result.FormatChunk.NumChannels != 1 && result.FormatChunk.NumChannels != 2 {
				return nil, fmt.Errorf("not 1 or 2 channels: %v", result.FormatChunk.NumChannels)
			}
			if result.FormatChunk.BitsPerSample != 16 {
				return nil, fmt.Errorf("not 16 bits: %v", result.FormatChunk.BitsPerSample)
			}
		} else if chunkHeader.SubChunkID.String() == "data" {
			if result.Data, err = io.ReadAll(r); err != nil {
				return nil, fmt.Errorf("while reading data chunk: %v", err)
			}
			break
		} else {
			buf := make([]byte, chunkHeader.SubChunkSize)
			if readBytes, err := r.Read(buf); readBytes != int(chunkHeader.SubChunkSize) || err != nil {
				return nil, fmt.Errorf("tried to read %v bytes of %q chunk: read %v bytes, got %v", chunkHeader.SubChunkSize, chunkHeader.SubChunkID.String(), readBytes, err)
			}
		}
	}
	if result.FormatChunk == nil {
		return nil, fmt.Errorf("no format chunk")
	}
	if result.Data == nil {
		return nil, fmt.Errorf("no data chunk")
	}
	return result, nil
}
