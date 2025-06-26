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
  g_sample = "zimtohrli_scores_sample" + str(random.randint(0, 10))
  g_sample = "zimtohrli_scores2"
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
  for i in range(300):
    os.environ["VAR%d" % i] = "0"
  for i in range(len(vec) - 1):
    os.environ["VAR%d" % i] = str(vec[i + 1])
    key += str(vec[i + 1]) + ":"
  if cached and (key in eval_hash):
    vec[0] = eval_hash[key]
    return

  #corpus = 'coresvnet'
  corpus = '*'
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
  for line in process.communicate(input=None, timeout=600)[0].splitlines():
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


def InitialSimplex(vec, dim, amount):
  """Initialize the simplex at origin."""
  EvalCacheForget()
  best = vec[:]
  Eval(best, g_binary)
  retval = [best]
  comp_order = list(range(1, dim + 1))
  random.shuffle(comp_order)

  for i in range(dim):
    index = comp_order[i]
    best = retval[0][:]
    best_vals = [None, best[0], None]
    best[index] += amount
    Eval(best, g_binary)
    retval.append(best)
    best_vals[2] = best[0]
    if (retval[0][0] < retval[-1][0]):
      print("not best, let's negate this axis")
      best = copy.copy(retval[0][:])
      best[index] -= amount
      Eval(best, g_binary)
      best_vals[0] = best[0]
      if (best[0] < retval[-1][0]):
        print("found new displacement best by negating")
        retval[-1] = best
      # perhaps one more try with shrinking amount
      best = copy.copy(retval[0][:])
      if (best_vals[1] < best_vals[0] and
          best_vals[1] < best_vals[2]):
        if (best_vals[0] < best_vals[2] and
            best_vals[2] - best_vals[0] > best_vals[0] - best_vals[1]):
          best[index] -= 0.1 * amount
          Eval(best, g_binary)
          if (best[0] < retval[-1][0]):
            print("found new best displacement by shrinking (neg)")
            retval[-1] = best
        if (best_vals[0] > best_vals[2] and
            best_vals[0] - best_vals[2] > best_vals[2] - best_vals[1]):
          best[index] += 0.1 * amount
          Eval(best, g_binary)
          if (best[0] < retval[-1][0]):
            print("found new best displacement by shrinking (neg)")
            retval[-1] = best


    retval.sort()
  return retval


if len(sys.argv) != 4:
  print("usage: ", sys.argv[0], "binary-name number-of-dimensions simplex-size")
  exit(1)

EvalCacheForget()

g_dim = int(sys.argv[2])
g_amount = float(sys.argv[3])
g_binary = sys.argv[1]
g_simplex = InitialSimplex([None] + [0.0] * g_dim,
                           g_dim, 7.0 * g_amount)
best = g_simplex[0][:]
g_simplex = InitialSimplex(best, g_dim, g_amount * 2.47)
best = g_simplex[0][:]
g_simplex = InitialSimplex(best, g_dim, g_amount)
best = g_simplex[0][:]
g_simplex = InitialSimplex(best, g_dim, g_amount * 0.33)
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
  g_simplex = InitialSimplex(best, g_dim, g_amount * mulli)
