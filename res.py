#!/usr/bin/env python3

# This file is part of CorPipe <https://github.com/ufal/crac2023-corpipe>.
#
# Copyright 2023 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=str, help="Experiment name")
parser.add_argument("epochs", default=0, nargs="?", type=int, help="Epochs to show")
parser.add_argument("-c", default=None, type=str, help="Compare to another experiment")
args = parser.parse_args()

treebanks = ["ca", "cs_pced", "cs_pdt", "de_pot", "de_par", "en_par", "en_gum", "es", "fr", "hu_k", "hu_s", "lt", "no_bok", "no_nyn", "pl", "ru", "tr"]
high_resource = ["ca", "cs_pced", "cs_pdt", "en_gum", "es", "fr", "hu_s", "no_bok", "no_nyn", "pl", "ru", "tr"]

# Load the data
def load(exp):
    exp_name, exp_suffix = exp, "eval"
    if exp_name.endswith((".e", ".s")):
        exp_name, exp_suffix = exp_name[:-2], f"eval{exp_name[-1]}"
    if "/" not in exp_name: exp_name = "logs/" + exp_name
    results = {}
    for path in sorted(glob.glob(f"{exp_name}*/*[0-9].{exp_suffix}")):
        base, epoch, *_ = os.path.basename(path)[:-len(exp_suffix)-1].split(".")
        epoch = int(epoch)
        for treebank in treebanks:
            if base.startswith(treebank):
                base = treebank
        if base not in treebanks:
            raise ValueError(f"Unknown treebank for evaluation '{base}'")
        results.setdefault(base, {})
        if epoch in results[base]:
            raise ValueError(f"Multiple evaluations for '{base}' epoch '{epoch}'")
        with open(path, "r", encoding="utf-8") as eval_file:
            for line in eval_file:
                line = line.rstrip("\r\n")
                if line.startswith("CoNLL score: "):
                    results[base][epoch] = line[13:]
    return results
results = load(args.exp)

# Print them out
def avg(callback, results):
    best_epoch = max(((sum(float(results[t][e]) for t in treebanks) / len(treebanks), e)
                      for e in results.get(treebanks[0], {}) if all(e in results.get(t, {}) for t in treebanks)), default=(None, 0))[1]
    values = [callback(results[t], best_epoch) if t in results else "" for t in treebanks]
    if all(value for t, value in zip(treebanks, values) if t in high_resource):
        values.append("{:.2f}".format(sum(float(value) for t, value in zip(treebanks, values) if t in high_resource) / (len(high_resource))))
    if all(values):
        values.append("{:.2f}".format(sum(float(value) for value in values[:-1]) / len(values[:-1])))
    return values
if args.c:
    others = load(args.c)
    def show(callback):
        xs, ys = avg(callback, results), avg(callback, others)
        return ["\033[{}m{:+.2f}\033[0m".format(32 if float(x) >= float(y) else 31,
                                                float(x) - float(y)) if x and y else ""
                for x, y in zip(xs, ys)]
else:
    show = lambda callback: avg(callback, results)
print("mode", *treebanks, "avg-hig", "avg", sep="\t")
print("last", *show(lambda res, _: list(res.values())[-1]), sep="\t")
print("best", *show(lambda res, best: res.get(best, "")), sep="\t")
print("max", *show(lambda res, _: max(list(res.values()), key=float)), sep="\t")
offset = 0 if any(0 in res for res in results.values()) else 1
for epoch in range(offset, offset + args.epochs):
    print(epoch, *show(lambda res, _: res.get(epoch, "")), sep="\t")
