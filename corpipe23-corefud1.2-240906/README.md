# The `corpipe23-corefud1.2-240906` Model Used in CRAC 2024

The `corpipe23-corefud1.2-240906` is a `mT5-large`-based multilingual model for
coreference resolution usable in CorPipe 23 <https://github.com/ufal/crac2023-corpipe>.
It is released at http://hdl.handle.net/11234/1-5673 under the CC BY-NC-SA 4.0
license.

The model is language agnostic (no _corpus id_ on input), so it can be in theory
used to predict coreference in any `mT5` language. However, the model expects
empty nodes to be already present on input, predicted by the
https://www.kaggle.com/models/ufal-mff/crac2024_zero_nodes_baseline/.

This model was present in the CorPipe 24 paper as an alternative to
a single-stage approach, where the empty nodes are predicted joinly with
coreference resolution (via https://hdl.handle.net/11234/1-5672), an approach
circa twice as fast but of slightly worse quality.

The model was trained using the following command (see the CorPipe 23 repository
for more information):
```sh
tb="ca_ancora cs_pcedt cs_pdt cu_proiel de_parcorfull de_potsdamcc en_gum en_litbank en_parcorfull es_ancora fr_democrat grc_proiel hbo_ptnk hu_korkor hu_szegedkoref lt_lcc no_bokmaalnarc no_nynorsknarc pl_pcc ru_rucor tr_itcc"
ratios_sqrt="7.3 12.3 10.3 2.8 1.2 2.1 5.2 5.2 1.2 7.7 6.1 3.0 1.1 1.8 4.0 2.2 5.7 5.3 8.4 4.5 2.7"

corpipe23.py --train --dev --treebanks $(for c in $tb; do echo data/$c/$c-corefud-train.conllu; done) --resample 10000 $ratios_sqrt --epochs=15 --batch_size=8 --adafactor --learning_rate=6e-4 --learning_rate_decay --encoder=google/mt5-large --segment=512 --right=50 --label_smoothing=0.2 --exp=corpipe23-corefud1.2
```

## CorefUD 1.2 Test Sets Results

With segment size 2560, the model achieves the following CorefUD 1.2 test set
results, assuming the empty nodes are first predicted using the
https://github.com/ufal/crac2024_zero_nodes_baseline:

| avg   | ca   | cs_pce | cs_pdt | cu   | de_par | de_pot | en_gum | en_lit | en_par | es   | fr   | grc  | hbo  | hu_kor | hu_sze | lt   | no_bok | no_nyn | pl   | ru   | tr   |
|-------|------|--------|--------|------|--------|--------|--------|--------|--------|------|------|------|------|--------|--------|------|--------|--------|------|------|------|
| 71.32 | 81.0 | 74.2   | 75.9   | 56.7 | 64.7   | 66.4   | 74.7   | 78.2   | 57.9   | 81.2 | 67.2 | 67.6 | 64.2 | 61.6   | 67.9   | 77.7 | 77.6   | 77.3   | 77.4 | 81.3 | 67.0 |


## Running the Model on Plain Text

To run the model on plain text, first the plain text needs to be tokenized and
converted to CoNLL-U (and optionally parsed if you also want mention heads),
by using for example UDPipe 2:

```sh
curl -F data="Eve came home and Peter greeted her there. Then Peter and Paul set out to a trip and Eve waved them off." \
  -F model=english -F tokenizer= -F tagger= -F parser=  https://lindat.mff.cuni.cz/services/udpipe/api/process \
  | python -X utf8 -c "import sys,json; sys.stdout.write(json.load(sys.stdin)['result'])" >input.conllu
```

Then, the empty nodes can optionally be predicted using https://github.com/ufal/crac2024_zero_nodes_baseline.

Finally, the CoNLL-U file should be processed by CorPipe 23, by using for example
```sh
python3 corpipe23.py --load corpipe23-corefud1.2-240906/model.h5 --exp . --epoch 0 --test input.conllu
```
which would generate the following predictions in `input.00.conllu`:
```
# generator = UDPipe 2, https://lindat.mff.cuni.cz/services/udpipe
# udpipe_model = english-ewt-ud-2.12-230717
# udpipe_model_licence = CC BY-NC-SA
# newdoc
# global.Entity = eid-etype-head-other
# newpar
# sent_id = 1
# text = Eve came home and Peter greeted her there.
1	Eve	Eve	PROPN	NNP	Number=Sing	2	nsubj	_	Entity=(c1--1)
2	came	come	VERB	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	0	root	_	_
3	home	home	ADV	RB	_	2	advmod	_	Entity=(c2--1)
4	and	and	CCONJ	CC	_	6	cc	_	_
5	Peter	Peter	PROPN	NNP	Number=Sing	6	nsubj	_	Entity=(c3--1)
6	greeted	greet	VERB	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	2	conj	_	_
7	her	she	PRON	PRP	Case=Acc|Gender=Fem|Number=Sing|Person=3|PronType=Prs	6	obj	_	Entity=(c1--1)
8	there	there	ADV	RB	PronType=Dem	6	advmod	_	Entity=(c2--1)|SpaceAfter=No
9	.	.	PUNCT	.	_	2	punct	_	_

# sent_id = 2
# text = Then Peter and Paul set out to a trip and Eve waved them off.
1	Then	then	ADV	RB	PronType=Dem	5	advmod	_	_
2	Peter	Peter	PROPN	NNP	Number=Sing	5	nsubj	_	Entity=(c4--1(c3--1)
3	and	and	CCONJ	CC	_	4	cc	_	_
4	Paul	Paul	PROPN	NNP	Number=Sing	2	conj	_	Entity=(c5--1)c4)
5	set	set	VERB	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	0	root	_	_
6	out	out	ADP	RP	_	5	compound:prt	_	_
7	to	to	ADP	IN	_	9	case	_	_
8	a	a	DET	DT	Definite=Ind|PronType=Art	9	det	_	Entity=(c6--2
9	trip	trip	NOUN	NN	Number=Sing	5	obl	_	Entity=c6)
10	and	and	CCONJ	CC	_	12	cc	_	_
11	Eve	Eve	PROPN	NNP	Number=Sing	12	nsubj	_	Entity=(c1--1)
12	waved	wave	VERB	VBD	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	5	conj	_	_
13	them	they	PRON	PRP	Case=Acc|Number=Plur|Person=3|PronType=Prs	12	obj	_	Entity=(c4--1)
14	off	off	ADP	RP	_	12	compound:prt	_	SpaceAfter=No
15	.	.	PUNCT	.	_	5	punct	_	SpaceAfter=No

```

## How to Cite

During the model release, only the preprint was available; please cite the
CRAC 2024 paper once published.

```
@misc{straka-2024-corpipe,
  title={{CorPipe at CRAC 2024: Predicting Zero Mentions from Raw Text}},
  author={Milan Straka},
  year={2024},
  eprint={2410.02756},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2410.02756},
}
```
