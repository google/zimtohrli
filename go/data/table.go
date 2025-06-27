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

package data

import (
	"bytes"
	"fmt"
)

// Row is a row of table data.
type Row []string

// Table is table structured data that can render in straight columns in a terminal.
type Table []Row

// String returns a string representation of the table with colSpacing blanks between columns.
func (t Table) String() string {
	maxCells := 0
	for _, row := range t {
		if len(row) > maxCells {
			maxCells = len(row)
		}
	}
	maxCellWidths := make([]int, maxCells)
	for _, row := range t {
		for cellIndex, cell := range row {
			if len(cell) > maxCellWidths[cellIndex] {
				maxCellWidths[cellIndex] = len(cell)
			}
		}
	}
	out := &bytes.Buffer{}
	for _, row := range t {
		fmt.Fprint(out, "|")
		for cellIndex, maxCellWidth := range maxCellWidths {
			if row == nil {
				for i := 0; i < maxCellWidth+1; i++ {
					fmt.Fprint(out, "-")
				}
			} else {
				fmt.Fprint(out, row[cellIndex])
				for i := len(row[cellIndex]); i < maxCellWidth+1; i++ {
					fmt.Fprint(out, " ")
				}
			}
			fmt.Fprint(out, "|")
		}
		fmt.Fprint(out, "\n")
	}
	return out.String()
}
