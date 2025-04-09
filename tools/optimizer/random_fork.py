#!/usr/bin/python
# Copyright (c) the Zimtohrli Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Implementation of simplex search for an external process.

The external process gets the input vector through environment variables.
Input of vector as setenv("VAR%dimension", val)
Getting the optimized function with regexp match from stdout
of the forked process.

https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method

start as ./simplex_fork.py binary dimensions amount
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range
import copy
import os
import random
import re
import subprocess
import sys

global g_offset
g_offset = int(sys.argv[4])

def Midpoint(simplex):
  """Nelder-Mead-like simplex midpoint calculation."""
  simplex.sort()
  dim = len(simplex) - 1
  retval = [None] + [0.0] * dim
  for i in range(1, dim + 1):
    for k in range(dim):
      retval[i] += simplex[k][i]
    retval[i] /= dim
  return retval


def Subtract(a, b):
  """Vector arithmetic, with [0] being ignored."""
  return [None if k == 0 else a[k] - b[k] for k in range(len(a))]

def Add(a, b):
  """Vector arithmetic, with [0] being ignored."""
  return [None if k == 0 else a[k] + b[k] for k in range(len(a))]

def Average(a, b):
  """Vector arithmetic, with [0] being ignored."""
  return [None if k == 0 else 0.5 * (a[k] + b[k]) for k in range(len(a))]


eval_hash = {}
g_best_val = None
g_sample = None
g_perceptual_sample_rate = None

def EvalCacheForget():
  global eval_hash
  eval_hash = {}
  global g_sample
  # g_sample = "zimtohrli_scores_sample" + str(random.randint(0, 10))
  # g_sample = "zimtohrli_scores2"
  g_sample = "zimtohrli_scores_2p9"
  global g_best_val
  g_best_val = None
  global g_perceptual_sample_rate
  g_perceptual_sample_rate = 97 + random.random() * 2.0


def Eval(vec, binary_name, cached=True):
  """Evaluates the objective function by forking a process.

  Args:
    vec: [0] will be set to the objective function, [1:] will
      contain the vector position for the objective function.
    binary_name: the name of the binary that evaluates the value.
  """
  global eval_hash
  global g_best_val
  global g_sample
  global g_perceptual_sample_rate
  key = ""
  # os.environ["BUTTERAUGLI_OPTIMIZE"] = "1"
  for i in range(999):
    os.environ["VAR%d" % i] = "0"
  for i in range(len(vec) - 1):
    os.environ["VAR%d" % i] = str(vec[i + 1])
    key += str(vec[i + 1]) + ":"
  if cached and (key in eval_hash):
    vec[0] = eval_hash[key]
    return

  #corpus = 'coresvnet'
  corpus = '*'
  #corpus = 'odaq'
  print("popen")
  process = subprocess.Popen(
      ('/usr/lib/google-golang/bin/go', 'run', '../go/bin/score/score.go', '--force',
       '--calculate_zimtohrli',
       '--calculate', '/usr/local/google/home/jyrki/' + g_sample + '/' + corpus,
       '--leaderboard', '/usr/local/google/home/jyrki/' + g_sample + '/' + corpus,
#       '--zimtohrli_parameters', '{"PerceptualSampleRate":%.15f}' % g_perceptual_sample_rate
       ),
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      env=dict(os.environ))

  #print("wait")
  #process.wait()
  #print("wait complete")
  found_score = False
  vec[0] = 1.0
  dct2 = 0.0
  dct4 = 0.0
  dct16 = 0.0
  dct32 = 0.0
  n = 0
  print("communicate")
  for line in process.communicate(input=None, timeout=1000)[0].splitlines():
    print("BE", line)
    sys.stdout.flush()
    linesplit = line.split(b'|')
    if len(linesplit) >= 3 and linesplit[1][:5] == b'Zimto':
      mse = float(linesplit[2])
      mean = 1.0 - float(linesplit[5])
      minval = 1.0 - float(linesplit[3])
      maxval = 1.0 - float(linesplit[4])
      vec[0] = mse # * (maxval ** 0.02)# * (mean ** 0.5) # * (minval ** 0.3) * (maxval ** 0.1)
      # vec[0] = float(linesplit[2])

      found_score = True
  print("end communicate", found_score)
  print("eval: ", vec)
  if (vec[0] <= 0.0):
    vec[0] = 1e30
  eval_hash[key] = vec[0]
  if found_score:
    if not g_best_val or vec[0] < g_best_val:
      g_best_val = vec[0]
      print("\nSaving best simplex\n")
      with open("best_simplex.txt", "w") as f:
        print(vec, file=f)
    print("wait really")
    process.wait()
    print("wait really done")
    return
  vec[0] = 1e31
  print("wait really [not score]")
  process.wait()
  print("wait really done [no score]")
  return
  # sys.exit("awful things happened")

def Reflect(simplex, binary):
  """Main iteration step of Nelder-Mead optimization. Modifies `simplex`."""
  simplex.sort()
  last = simplex[-1]
  mid = Midpoint(simplex)
  diff = Subtract(mid, last)
  mirrored = Add(mid, diff)
  Eval(mirrored, binary)
  if mirrored[0] > simplex[-2][0]:
    print("\nStill worst\n\n")
    # Still the worst, shrink towards the best.
    shrinking = Average(simplex[-1], simplex[0])
    Eval(shrinking, binary)
    print("\nshrinking...\n\n")
    simplex[-1] = shrinking
    return
  if mirrored[0] < simplex[0][0]:
    # new best
    print("\nNew Best\n\n")
    even_further = Add(mirrored, diff)
    Eval(even_further, binary)
    if even_further[0] < mirrored[0]:
      print("\nEven Further\n\n")
      mirrored = even_further
    simplex[-1] = mirrored
    # try to extend
    return
  else:
    # not a best, not a worst point
    simplex[-1] = mirrored

import copy
    
def InitialSimplex(vec, dim, amount, biases):
  """Initialize the simplex at origin."""
  EvalCacheForget()
  best = vec[:]
  Eval(best, g_binary)
  retval = [best]

  for i in range(dim):
    best = retval[0][:]
    bestcopy = copy.copy(best)
    rangelimit = random.random() < 0.4
    for k in range(dim):
      r = ((random.random() - 0.5) * 2) ** 5
      best[k + 1] += amount * r
      ave = 0
      if (k != 0 and k != dim - 1 and biases[k + 1] != 0):
        ave = (best[k] + best[k + 2]) * 0.5
      best[k + 1] += 0.001 * amount * (biases[k] + ave - best[k + 1]) * random.random()
      #if k < 90 or (k >= 130 and k < g_offset):
      if k < g_offset:
        best[k + 1] = 0.0
      if rangelimit:
        rangevals = dim - g_offset - 5
        temp_offset = random.randint(g_offset, g_offset + rangevals)
        if k < temp_offset or k >= temp_offset + 5:
          best[k + 1] = bestcopy[k + 1]
    Eval(best, g_binary)
    retval.append(best)
    if (best[0] < retval[0][0]):
      amount *= 1.25
      print("scaling up amount", amount)
    else:
      amount *= 0.98
      print("scaling down amount", amount)
    retval.sort()
  return retval


if len(sys.argv) != 5:
  print("usage: ", sys.argv[0], "binary-name number-of-dimensions simplex-size first-non-zero-dim")
  exit(1)

EvalCacheForget()


def FileLine(i):
  path = "/usr/local/google/home/jyrki/github/zimtohrli/cpp/zimt/fourier_bank.cc"
  
  linenoset = []
  valset = []
  print("trying", i)
  for lineno, line in enumerate(open(path).readlines()):
    if 'atof(getenv("VAR' in line:
      ixu = int(line.split("VAR")[1].split('"')[0])
      if not (ixu == i - 1 or ixu == i or ixu == i + 1):
        continue
      linenoset.append(lineno)
      if not '+ atof(' in line:
        return 0.0
      numstr = line[:line.index('+ atof(')].strip().split(' ')[-1]
      numstr = numstr.split("(")[-1]
      if numstr[-1] == 'f':
        numstr = numstr[0:-1]
      num = float(numstr)
      valset.append(num)
    if len(linenoset) == 3 and abs(linenoset[2] - linenoset[0]) == 2 and linenoset[0] + linenoset[2] == 2 * linenoset[1]:
      # good
      bias = 0.5 * (valset[0] + valset[2]) - valset[1]
      print("found bias for", i, bias)
      return bias

  return 0.0

g_dim = int(sys.argv[2])
g_biases = [FileLine(i) for i in range(g_dim)]


g_amount = float(sys.argv[3])
g_binary = sys.argv[1]
g_simplex = InitialSimplex([None] + [0.0] * g_dim,
                           g_dim, 7.0 * g_amount, g_biases)
best = g_simplex[0][:]
g_simplex = InitialSimplex(best, g_dim, g_amount * 2.47, g_biases)
best = g_simplex[0][:]
g_simplex = InitialSimplex(best, g_dim, g_amount, g_biases)
best = g_simplex[0][:]
g_simplex = InitialSimplex(best, g_dim, g_amount * 0.33, g_biases)
best = g_simplex[0][:]

for restarts in range(99999):
  for ii in range(g_dim * 5):
    g_simplex.sort()
    print("reflect", ii, g_simplex[0])
    Reflect(g_simplex, g_binary)

  mulli = 0.1 + 15 * random.random()**2.0
  print("\n\n\nRestart", restarts, "mulli", mulli)
  g_simplex.sort()
  best = g_simplex[0][:]
  g_simplex = InitialSimplex(best, g_dim, g_amount * mulli, g_biases)
