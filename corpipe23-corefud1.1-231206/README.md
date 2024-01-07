# The `corpipe23-corefud1.1-231206` Model

The `corpipe23-corefud1.1-231206` is a `mT5-large`-based multilingual model for
coreference resolution usable in CorPipe 23 <https://github.com/ufal/crac2023-corpipe>.
It is released under the CC BY-NC-SA 4.0 license.

The model is language agnostic (no _corpus id_ on input), so it can be used to
predict coreference in any `mT5` language (for zero-shot evaluation, see the
paper). However, note that the empty nodes must be present already on input,
they are not predicted (the same settings as in the CRAC23 shared task).

The model was trained using the following command (see CorPipe 23 for more
information):
```sh
tb="ca_ancora cs_pcedt cs_pdt de_parcorfull de_potsdamcc en_gum en_parcorfull es_ancora fr_democrat hu_korkor hu_szegedkoref lt_lcc no_bokmaalnarc no_nynorsknarc pl_pcc ru_rucor tr_itcc"
ratios_sqrt="8.4 14.0 11.7 1.4 2.4 5.6 1.4 8.8 6.9 2.0 4.6 2.5 6.5 6.0 9.5 5.1 3.1"

corpipe23.py --train --dev --treebanks $(for c in $tb; do echo data/$c/$c-corefud-train.conllu; done) --resample 8000 $ratios_sqrt --epochs=15 --batch_size=8 --adafactor --learning_rate=6e-4 --learning_rate_decay --encoder=google/mt5-large --segment=512 --right=50 --label_smoothing=0.2 --exp=mt5-large
```

## CorefUD 1.1 Development and Test Sets Results.

With segment size 2560, the model achieves the following CorefUD 1.1 development
set results, as evaluated by the official CRAC 2023 metric via the CorefUD scorer:

| avg   | ca    | cs-pce | cs-pdt | de-par | de-pot | en-gum | en-par | es    | fr    | hu-kor | hu-sze | lt    | no-boo | no-nyn | pl    | ru    | tr    |
|-------|-------|--------|--------|--------|--------|--------|--------|-------|-------|--------|--------|-------|--------|--------|-------|-------|-------|
| 75.19 | 81.83 | 80.73  | 79.29  | 77.92  | 74.49  | 76.45  | 64.42  | 83.64 | 70.85 | 68.73  | 68.67  | 79.79 | 81.13  | 81.28  | 77.35 | 76.90 | 54.71 |

With segment size 2560, the model was also evaluated on the CorefUD 1.1 test set
using CodaLab https://codalab.lisn.upsaclay.fr/competitions/11800 :

| avg   | ca    | cs-pce | cs-pdt | de-par | de-pot | en-gum | en-par | es    | fr    | hu-kor | hu-sze | lt    | no-boo | no-nyn | pl    | ru    | tr    |
|-------|-------|--------|--------|--------|--------|--------|--------|-------|-------|--------|--------|-------|--------|--------|-------|-------|-------|
| 73.21 | 82.39 | 77.93  | 77.85  | 69.94  | 67.93  | 75.02  | 64.79  | 82.26 | 68.22 | 67.95  | 69.16  | 75.63 | 78.94  | 77.24  | 78.93 | 80.35 | 49.97 |

## Running the Model on Plain Text

To run the model on plain text, first the plain text needs to be tokenized and
converted to CoNLL-U (and optionally parsed if you also want mention heads),
by using for example UDPipe 2:

```sh
curl -F data="Eve came home and Peter greeted her there. Then Peter and Paul set out to a trip and Eve waved them off." \
  -F model=english -F tokenizer= -F tagger= -F parser=  https://lindat.mff.cuni.cz/services/udpipe/api/process \
  | python -X utf8 -c "import sys,json; sys.stdout.write(json.load(sys.stdin)['result'])" >input.conllu
```

Then the CoNLL-U file can be processed by CorPipe 23, by using for example
```sh
python3 corpipe23.py --load corpipe23-corefud1.1-231206/model.h5 --exp . --epoch 0 --test input.conllu
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

```
@inproceedings{straka-2023-ufal,
    title = "{{\'U}FAL} {C}or{P}ipe at {CRAC} 2023: Larger Context Improves Multilingual Coreference Resolution",
    author = "Straka, Milan",
    editor = "{\v{Z}}abokrtsk{\'y}, Zden{\v{e}}k  and Ogrodniczuk, Maciej",
    booktitle = "Proceedings of the CRAC 2023 Shared Task on Multilingual Coreference Resolution",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.crac-sharedtask.4",
    doi = "10.18653/v1/2023.crac-sharedtask.4",
    pages = "41--51",
}
```
