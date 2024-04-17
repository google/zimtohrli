# Zimtohrli Go tools

Zimtohrli has a Go wrapper, goohrli, installable via `go install`.

It works by using `cgo` to wrap the Zimtohrli C++ library.

To install it, a few dependencies are needed. To do this in a Debian-like system:

```
sudo apt install -y libc++-dev libc++abi-dev libflac-dev libogg-dev libvorbis-dev libvorbis-dev libopus-dev
```

## Goohrli wrapper

To install the wrapper library when inside a go module (a directory with a `go.mod` file):

```
go get github.com/google/zimtohrli/go/goohrli
```

For documentation about the API, see [https://pkg.go.dev/github.com/google/zimtohrli/go/goohrli](https://pkg.go.dev/github.com/google/zimtohrli/go/goohrli)

## Compare command line tool

A simple command line tool to compare WAV files is provided.

To install it:

```
go install github.com/google/zimtohrli/go/bin/compare
```

To run it (run `go env` to see your $GOPATH):

```
$GOPATH/bin/compare -path_a reference.wav -path_b distortion.wav
```