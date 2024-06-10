#!/bin/sh

# Copyright 2024 The Ringli Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# install_external_metrics.sh installs the metrics that aren't directly
# linked against the C++ and Go code of Zimtohrli.
#
# It depends on various things like pyenv (for building Python virtual
# environments of old Python versions), git (for downloading particular
# versions of software published on GitHub), and Go (for building some
# wrappers around command line binaries).
#
# Usage:
# install_external_metrics.sh DESTINATION_DIRECTORY
#
# It will drop executable files in DESTINATION_DIRECTORY with the name
# pattern serve_XXX, which when called conform to the format expected
# by github.com/google/zimtohrli/go/pipe.
#

DST="${1}"

if test -z "${DST}"; then
    echo "Usage: ${0} DESTINATION_DIRECTORY"
    exit 1
fi

INSTALL_LOG="${DST}/install_log.txt"

export PYENV_DST="${DST}/pyenv"
if test -d "${PYENV_DST}"; then
    echo "pyenv v2.3.36 already checked out in ${PYENV_DST}."
else
    echo "Checking out pyenv v2.3.36 into ${PYENV_DST}/..."
    git -c advice.detachedHead=false clone --quiet --branch v2.3.36 --single-branch https://github.com/pyenv/pyenv.git "${PYENV_DST}"
fi

function install_python() {
    NAME="${1}"
    VERSION="${2}"
    VARNAME="${3}"

    export PYENV_ROOT="${PYENV_DST}/installs/${NAME}"
    local PYTHON_ROOT="${PYENV_ROOT}/versions/${VERSION}"
    if test -d "${PYTHON_ROOT}"; then
        echo "Python ${VERSION} already installed in ${PYTHON_ROOT}."
    else
        echo "Installing Python ${VERSION} into ${PYTHON_ROOT}..."
        "${PYENV_DST}/bin/pyenv" install "${VERSION}" 2> "${INSTALL_LOG}"
        if test "${?}" != "0"; then
            cat "${INSTALL_LOG}"
            exit 2
        fi
    fi
    export "${VARNAME}"="${PYTHON_ROOT}/bin/python"
}

RETRUNC_PY="${DST}/retrunc.py"
echo "Dropping library ${RETRUNC_PY}..."
cat > "${RETRUNC_PY}" << EOF
import contextlib
import numpy as np
import scipy.io.wavfile
import tempfile
import sys

def raise_if_nan(samples):
    if np.any(np.isnan(samples)):
        raise ValueError(f'nan values in {samples.shape}')

@contextlib.contextmanager
def resampled_and_truncated_to(path, want_rate = None, max_samples = None, want_samples = None):
    rate, samples = scipy.io.wavfile.read(path)
    samples = samples if np.max(samples) <= 1 else samples / float(1<<15)
    raise_if_nan(samples)
    if (want_samples and len(samples) != want_samples) or (want_rate and rate != want_rate) or max_samples:
        if want_rate and rate != want_rate:
            samples = scipy.signal.resample(samples, int(len(samples) * want_rate / rate))
            raise_if_nan(samples)
        if want_samples and len(samples) != want_samples:
            if want_samples and len(samples) < want_samples:
                raise ValueError(f'{path} is {len(samples)} long, which is less than {want_samples}s')
            if want_samples and len(samples) > want_samples:
                samples = samples[:want_samples]
                raise_if_nan(samples)
        if max_samples and len(samples) > max_samples:
            samples = samples[:max_samples]
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            scipy.io.wavfile.write(f, want_rate, samples)
            yield f.name
    else:
        yield path

@contextlib.contextmanager
def resample_and_truncate_all(want_rate = None, max_samples = None, want_samples = None, untreated_paths = [], treated_paths: list = []):
    if not untreated_paths:
        yield treated_paths
        return
    last_untreated = untreated_paths.pop()
    with resampled_and_truncated_to(path=last_untreated, want_rate=want_rate, max_samples=max_samples, want_samples=want_samples) as first_treated:
        treated_paths.insert(0, first_treated)
        with resample_and_truncate_all(want_rate=want_rate, max_samples=max_samples, want_samples=want_samples, untreated_paths=untreated_paths, treated_paths=treated_paths) as treated_paths:
            yield treated_paths

@contextlib.contextmanager
def uniform(paths: list):
    min_seconds = None
    max_rate = None
    for path in paths:
        rate, samples = scipy.io.wavfile.read(path)
        if not max_rate or rate > max_rate:
            max_rate = rate
        seconds = len(samples) / rate
        if not min_seconds or seconds < min_seconds:
            min_seconds = seconds
    want_samples = int(min_seconds * max_rate)
    with resample_and_truncate_all(want_rate=max_rate, want_samples=want_samples, untreated_paths=paths) as uniform_paths:
        yield uniform_paths
EOF

function install_warp_q() {
    local WARP_Q_ROOT="${DST}/WARP-Q"
    if test -d "${WARP_Q_ROOT}"; then
        echo "WARP-Q v1.0.0 already checked out in ${WARP_Q_ROOT}."
    else
        echo "Checking out WARP-Q v1.0.0 into ${WARP_Q_ROOT}..."
        git -c advice.detachedHead=false clone --quiet --branch v1.0.0 --single-branch https://github.com/wjassim/WARP-Q.git "${WARP_Q_ROOT}"
    fi

    install_python warpq 3.9.18 PYTHON39

    echo "Ensuring WARP-Q dependencies installed..."
    "${PYTHON39}" -m pip install -r "${WARP_Q_ROOT}/requirements.txt" > "${INSTALL_LOG}" 2>&1
    if test "${?}" != "0"; then
        cat "${INSTALL_LOG}"
        exit 3
    fi

    local WARP_Q_SCRIPT="${DST}/serve_warp_q.py"
    echo "Dropping executable ${WARP_Q_SCRIPT}..."
    cat > "${WARP_Q_SCRIPT}" <<EOF
#!${PYTHON39}

import sys
import retrunc

sys.path.insert(0,'${WARP_Q_ROOT}')
from WARPQ.WARPQmetric import warpqMetric

want_samplerate = 16000

args = dict(
    sr=want_samplerate,
    mode='predict_file',
    mapping_model='${WARP_Q_ROOT}/models/SequentialStack/Genspeech_TCDVoIP_PSup23.zip',
    apply_vad=True,
    n_mfcc=13,
    fmax=5000,
    patch_size=0.4,
    sigma=[[1,0],[0,3],[1,3]],
)

warpq = warpqMetric(args)
print('READY:WARP-Q')
try:
    while True:
        ref = input('REF\n')
        dist = input('DIST\n')
        with retrunc.resample_and_truncate_all(want_rate=want_samplerate, max_samples=10*want_samplerate, untreated_paths=[ref, dist]) as treated_paths:
            _, warpq_mappedScore = warpq.evaluate(treated_paths[0], treated_paths[1])
            print(f'SCORE={warpq_mappedScore}')
except EOFError:
    pass
EOF
    chmod +x "${WARP_Q_SCRIPT}"
}

install_warp_q

function install_dpam() {
    install_python dpam 3.7.17 PYTHON37

    echo "Installing DPAM and dependencies..."
    "${PYTHON37}" -m pip install -U dpam==0.0.4 numba==0.48.0 protobuf==3.20.3 tensorflow==1.14.0 scikit-learn==0.20.3 scipy==1.2.1 tqdm==4.32.2 resampy==0.2.2 librosa==0.7.2 > "${INSTALL_LOG}" 2>&1
    if test "${?}" != "0"; then
        cat "${INSTALL_LOG}"
        exit 3
    fi

    local DPAM_SCRIPT="${DST}/serve_dpam.py"
    echo "Dropping executable ${DPAM_SCRIPT}..."
    cat > "${DPAM_SCRIPT}" <<EOF
#!${PYTHON37}

import dpam
import retrunc

loss_fn = dpam.DPAM()
print('READY:DPAM')
try:
    while True:
        ref = input('REF\n')
        dist = input('DIST\n')
        with retrunc.uniform([ref, dist]) as uniform_paths:
            ref_wav = dpam.load_audio(uniform_paths[0])
            dist_wav = dpam.load_audio(uniform_paths[1])
            print(f'SCORE={loss_fn.forward(ref_wav, dist_wav)[0]}')
except EOFError:
    pass
EOF
    chmod +x "${DPAM_SCRIPT}"
}

install_dpam

function install_cdpam() {
    install_python cdpam 3.7.17 PYTHON37

    echo "Installing CDPAM and dependencies..."
    "${PYTHON37}" -m pip install cdpam==0.0.6 torch==1.4.0 numba==0.48.0 librosa==0.7.2 matplotlib==2.2.5 numpy==1.16.6 resampy==0.2.2 scikit-learn==0.20.4 scipy==1.2.3 tqdm==4.43.0 tensorboard==2.1.0 future > "${INSTALL_LOG}" 2>&1
    if test "${?}" != "0"; then
        cat "${INSTALL_LOG}"
        exit 3
    fi

    local CDPAM_SCRIPT="${DST}/serve_cdpam.py"
    echo "Dropping executable ${CDPAM_SCRIPT}..."
    cat > "${CDPAM_SCRIPT}" <<EOF
#!${PYTHON37}

import cdpam

loss_fn = cdpam.CDPAM(dev='cpu')
print('READY:CDPAM')
try:
    while True:
        ref = input('REF\n')
        dist = input('DIST\n')
        ref_wav = cdpam.load_audio(ref)
        dist_wav = cdpam.load_audio(dist)
        print(f'SCORE={loss_fn.forward(ref_wav, dist_wav)[0]}')
except EOFError:
    pass
EOF
    chmod +x "${CDPAM_SCRIPT}"
}

install_cdpam

( cd "${DST}" && go mod init external_metrics ) > "${INSTALL_LOG}" 2>&1

function install_peaq() {
    local PEAQ_ROOT="${DST}/peaqb-fast"
    if test -d "${PEAQ_ROOT}"; then
        echo "peaqb-fast already checked out in ${PEAQ_ROOT}."
    else
        echo "Checking out peaqb-fast into ${PEAQ_ROOT}..."
        git clone --quiet --depth 1 https://github.com/akinori-ito/peaqb-fast.git "${PEAQ_ROOT}"
        ( cd "${PEAQ_ROOT}" && git reset --hard "2fa1c8ddf6f4ee30a0ca206944fd7b180f059400" )
    fi

    local PEAQ_BIN="${PEAQ_ROOT}/src/peaqb"
    if test -f "${PEAQ_BIN}"; then
        echo "${PEAQ_BIN} already compiled."
    else
        echo "Compiling ${PEAQ_BIN}..."
        ( cd "${PEAQ_ROOT}" && ./configure && make -j 16 ) > "${INSTALL_LOG}" 2>&1
        if test "${?}" != "0"; then
            cat "${INSTALL_LOG}"
            exit 3
        fi
    fi

    local WRAPPER="${DST}/serve_peaqb"
    echo "Dropping PEAQ wrapper ${WRAPPER}..."
    cat > "${WRAPPER}.go" <<EOF
package main

import (
    "bufio"
    "fmt"
    "io"
    "log"
    "os"
    "os/exec"
    "strconv"
    "strings"
)

func main() {
    reader := bufio.NewReader(os.Stdin)
    fmt.Println("READY:PEAQB")
    for {
        fmt.Println("REF")
        ref, err := reader.ReadString('\n')
        if err == io.EOF {
            break
        } else if err != nil {
            log.Fatal(err)
        }
        fmt.Println("DIST")
        dist, err := reader.ReadString('\n')
        if err == io.EOF {
            break
        } else if err != nil {
            log.Fatal(err)
        }
        command := exec.Command("${PEAQ_BIN}", "-r", strings.TrimSpace(ref), "-t", strings.TrimSpace(dist))
        output, err := command.CombinedOutput()
        if err != nil {
            log.Fatal(fmt.Errorf("when calling %v: %v\n%s", command, err, output))
        }
        odgSum := 0.0
        count := 0
        for _, line := range strings.Split(string(output), "\n") {
            if match, found := strings.CutPrefix(line, "ODG: "); found {
                odg, err := strconv.ParseFloat(match, 64)
                if err != nil {
                    log.Fatal(err)
                }
                odgSum += odg
                count++
            }
        }
        fmt.Printf("SCORE=%f\n", odgSum / float64(count))
    }
}
EOF

    ( cd "${DST}" && go build "${WRAPPER}.go" ) > "${INSTALL_LOG}" 2>&1
    if test "${?}" != "0"; then
        cat "${INSTALL_LOG}"
        exit 3
    fi

    local WRAPPER="${DST}/serve_2f"
    echo "Dropping 2f wrapper ${WRAPPER}..."
    cat > "${WRAPPER}.go" <<EOF
package main

import (
    "bufio"
    "fmt"
    "io"
    "log"
    "math"
    "os"
    "os/exec"
    "strconv"
    "strings"
)

func extract(line string, label string) (float64, bool, error) {
    if match, found := strings.CutPrefix(line, fmt.Sprintf("%s: ", label)); found {
        f, err := strconv.ParseFloat(match, 64)
        if err != nil {
            return 0, false, err
        }
        return f, true, nil
    }
    return 0, false, nil
}

func main() {
    reader := bufio.NewReader(os.Stdin)
    fmt.Println("READY:2f")
    for {
        fmt.Println("REF")
        ref, err := reader.ReadString('\n')
        if err == io.EOF {
            break
        } else if err != nil {
            log.Fatal(err)
        }
        fmt.Println("DIST")
        dist, err := reader.ReadString('\n')
        if err == io.EOF {
            break
        } else if err != nil {
            log.Fatal(err)
        }
        command := exec.Command("${PEAQ_BIN}", "-r", strings.TrimSpace(ref), "-t", strings.TrimSpace(dist))
        output, err := command.CombinedOutput()
        if err != nil {
            log.Fatal(fmt.Errorf("when calling %v: %v\n%s", command, err, output))
        }
        adbSum := 0.0
        avgModDiff1Sum := 0.0
        count := 0
        for _, line := range strings.Split(string(output), "\n") {
            if f, found, err := extract(line, "ADBb"); err != nil {
                log.Fatal(err)
            } else if found {
                count++
                adbSum += f
            }
            if f, found, err := extract(line, "AvgModDiff1b"); err != nil {
                log.Fatal(err)
            } else if found {
                avgModDiff1Sum += f
            }
        }
        meanADB := adbSum / float64(count)
        meanAvgModDiff1 := avgModDiff1Sum / float64(count)
        fmt.Printf("SCORE=%f\n", 56.1345 /
                                 (1 + math.Pow(-0.0282 * meanAvgModDiff1 - 0.8628,
                                               2)) -27.1451 * meanADB + 86.3515)
    }
}
EOF

    ( cd "${DST}" && go build "${WRAPPER}.go" ) > "${INSTALL_LOG}" 2>&1
    if test "${?}" != "0"; then
        cat "${INSTALL_LOG}"
        exit 3
    fi
}

install_peaq

function install_pesq() {
    local PESQ_ROOT="${DST}/itu-pesq"
    if test -d "${PESQ_ROOT}"; then
        echo "ITU PESQ already checked out in ${PESQ_ROOT}."
    else
        echo "Checking out ITU PESQ into ${PESQ_ROOT}..."
        git clone --quiet --depth 1 https://github.com/dennisguse/ITU-T_pesq.git "${PESQ_ROOT}"
        ( cd "${PESQ_ROOT}" && git reset --hard "ed390e8b1c7847daa511b057ec7f8d42c387700f" )
    fi

    local PESQ_BIN="${PESQ_ROOT}/bin/itu-t-pesq2005"
    if test -f "${PESQ_BIN}"; then
        echo "${PESQ_BIN} already compiled."
    else
        echo "Compiling ${PESQ_BIN}..."
        ( cd "${PESQ_ROOT}" && CFLAGS=-fcommon make -j 16 ) > "${INSTALL_LOG}" 2>&1
        if test "${?}" != "0"; then
            cat "${INSTALL_LOG}"
            exit 3
        fi
    fi

    ( cd "${DST}" && go get github.com/google/zimtohrli@v0.1.11 )

    local WRAPPER="${DST}/serve_pesq"
    echo "Dropping PESQ wrapper ${WRAPPER}..."
    cat > "${WRAPPER}.go" <<EOF
package main

import (
    "bufio"
    "fmt"
    "io"
    "log"
    "os"
    "os/exec"
    "strconv"
    "strings"
    "regexp"

    "github.com/google/zimtohrli/go/aio"
    "github.com/google/zimtohrli/go/audio"
)

const rate = 16000

func truncateAndResample(paths []string) ([]string, error) {
    shortestLength := -1
    audios := make([]*audio.Audio, len(paths))
    for index, path := range paths {
        a, err := aio.LoadAtRate(path, rate)
        if err != nil {
            return nil, err
        }
        audios[index] = a
        if samples := len(a.Samples[0]); shortestLength == -1 || samples < shortestLength {
            shortestLength = samples
        }
    }
    result := make([]string, len(paths))
    for audioIndex, a := range audios {
        for channelIndex := range a.Samples {
            a.Samples[channelIndex] = a.Samples[channelIndex][:shortestLength]
        }
        f, err := os.CreateTemp(os.TempDir(), "zimtohrli.pesq.*.wav")
        if err != nil {
            return nil, err
        }
        f.Close()
        if err := aio.Save(a, f.Name()); err != nil {
            return nil, err
        }
        result[audioIndex] = f.Name()
    }
    return result, nil
}

var resultPattern = regexp.MustCompile("^P.862 Prediction \\\\(Raw MOS, MOS-LQO\\\\):\\\\s+=\\\\s+([-0-9.]+)\\\\s+([-0-9.]+)\\\\s*$")

func main() {
    reader := bufio.NewReader(os.Stdin)
    fmt.Println("READY:PESQ")
    for {
        fmt.Println("REF")
        ref, err := reader.ReadString('\n')
        if err == io.EOF {
            break
        } else if err != nil {
            log.Fatal(err)
        }
        fmt.Println("DIST")
        dist, err := reader.ReadString('\n')
        if err == io.EOF {
            break
        } else if err != nil {
            log.Fatal(err)
        }
        if err := func() error {
            paths, err := truncateAndResample([]string{strings.TrimSpace(ref), strings.TrimSpace(dist)})
            defer func() {
                for _, path := range paths {
                    if path != "" {
                        os.RemoveAll(path)
                    }
                }
            }()
            if err != nil {
                return err
            }
            command := exec.Command("${PESQ_BIN}", "+16000", paths[0], paths[1])
            out, err := command.CombinedOutput()
            if err != nil {
                return fmt.Errorf("when running %v: %v\n%s", command, err, out)
            }
            lines := strings.Split(string(out), "\n")
            lastLine := strings.TrimSpace(lines[len(lines)-2])
            match := resultPattern.FindStringSubmatch(lastLine)
            if len(match) == 0 {
                return fmt.Errorf("last line %q didn't match expectations (%v) in %s", lastLine, resultPattern, out)
            }
            score, err := strconv.ParseFloat(match[2], 64)
            if err != nil {
                return err
            }
            fmt.Printf("SCORE=%f\n", score)
            return nil
        }(); err != nil {
            log.Fatal(err)
        }
    }
}
EOF

    ( cd "${DST}" && go build "${WRAPPER}.go" ) > "${INSTALL_LOG}" 2>&1
    if test "${?}" != "0"; then
        cat "${INSTALL_LOG}"
        exit 3
    fi
}

install_pesq
