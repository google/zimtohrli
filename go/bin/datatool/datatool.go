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

// datatool allows a user to interact with the known datasets.
//
// Example executions:
//
// To download and show the coresv.net dataset:
// tool --dataset=coresv.net --show
//
// To download and calculate Zimtohrli and a bunch of other metrics on the coresv.net dataset:
// tool --dataset=coresv.net --zimtohrli_compare=third_party/zimtohrli/cpp/compare --ringli_analyze=util/compression/ringli/analysis/analyze_python_metrics --calculate ~/tmp/coresv.net.metrics.json
//
// To calculate the correlation scores of the downloaded dataset:
// tool --dataset=coresv.net --correlate ~/tmp/coresv.net.metrics.json
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"

	"github.com/google/zimtohrli/go/dataset/runner"
)

func main() {
	setup := runner.Setup{
		Show:                         flag.Bool("show", false, runner.SetupFlagString("Show")),
		Calculate:                    flag.String("calculate", "", runner.SetupFlagString("Calculate")),
		Correlate:                    flag.String("correlate", "", runner.SetupFlagString("Correlate")),
		CopyToDirectory:              flag.String("copy_to_directory", "", runner.SetupFlagString("CopyToDirectory")),
		OptimizeDatasets:             flag.String("optimize_datasets", "", runner.SetupFlagString("OptimizeDatasets")),
		OptimizeValidationFraction:   flag.Float64("optimize_validation_fraction", 0.1, runner.SetupFlagString("OptimizeValidationFraction")),
		OptimizeMinibatchFraction:    flag.Float64("optimize_minibatch_fraction", 0.1, runner.SetupFlagString("OptimizeMinibatchFraction")),
		ZimtohrliCompare:             flag.Bool("zimtohrli_compare", false, runner.SetupFlagString("ZimtohrliCompare")),
		ZimtohrliFrequencyResolution: flag.Float64("zimtohrli_frequency_resolution", 4.0, runner.SetupFlagString("ZimtohrliFrequencyResolution")),
		LocalDatasetPath:             flag.String("local_dataset_path", "", runner.SetupFlagString("LocalDatasetPath")),
		MaxWorkers:                   flag.Int("max_workers", runtime.NumCPU(), runner.SetupFlagString("MaxWorkers")),
		ProgressDirectory:            flag.String("progress_directory", "", runner.SetupFlagString("ProgressDirectory")),
	}
	setup.Dataset = flag.String("dataset", "", fmt.Sprintf(runner.SetupFlagString("Dataset"), setup.DatasetNames()))
	flag.Parse()
	if err := setup.Run(); err != nil {
		flag.Usage()
		log.Print(err)
		os.Exit(1)
	}
}
