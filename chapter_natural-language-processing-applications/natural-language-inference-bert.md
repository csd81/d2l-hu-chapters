# Természetes nyelvi inferencia: BERT finomhangolása
:label:`sec_natural-language-inference-bert`

A fejezet korábbi szakaszaiban
figyelemalapú architektúrát terveztünk
(a :numref:`sec_natural-language-inference-attention` szakaszban)
a természetes nyelvi inferencia feladathoz
az SNLI adathalmazon (ahogyan azt a :numref:`sec_natural-language-inference-and-dataset` ismertette).
Most ezt a feladatot a BERT finomhangolásával közelítjük meg újra.
Amint a :numref:`sec_finetuning-bert` szakaszban tárgyaltuk,
a természetes nyelvi inferencia szekvenciaszintű szövegpár-osztályozási feladat,
és a BERT finomhangolásához csupán egy kiegészítő MLP-alapú architektúra szükséges,
ahogyan azt a :numref:`fig_nlp-map-nli-bert` ábra szemlélteti.

![Ez a szakasz előre tanított BERT-et táplál egy MLP-alapú architektúrába a természetes nyelvi inferencia céljából.](../img/nlp-map-nli-bert.svg)
:label:`fig_nlp-map-nli-bert`

Ebben a szakaszban
letöltjük a BERT előre tanított kis verzióját,
majd finomhangoljuk azt
a természetes nyelvi inferenciához az SNLI adathalmazon.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
import json
import multiprocessing
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import json
import multiprocessing
import torch
from torch import nn
import os
```

## [**Előre tanított BERT betöltése**]

Elmagyaráztuk, hogyan tanítható elő a BERT a WikiText-2 adathalmazon
a :numref:`sec_bert-dataset` és :numref:`sec_bert-pretraining` szakaszokban
(megjegyezzük, hogy az eredeti BERT modell jóval nagyobb korpuszon lett előre tanítva).
Amint a :numref:`sec_bert-pretraining` szakaszban tárgyaltuk,
az eredeti BERT modellnek több száz millió paramétere van.
A következőkben
az előre tanított BERT két verzióját kínáljuk:
a „bert.base" nagyjából akkora, mint az eredeti BERT alapmodell, amelynek finomhangolásához sok számítási erőforrás szükséges,
míg a „bert.small" egy kis verzió a szemléltetés megkönnyítésére.

```{.python .input}
#@tab mxnet
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.zip',
                             '7b3820b35da691042e5d34c0971ac3edbd80d3f4')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.zip',
                              'a4e718a47137ccd1809c9107ab4f5edd317bae2c')
```

```{.python .input}
#@tab pytorch
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.torch.zip',
                             '225d66f04cae318b841a13d32af3acc165f253ac')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.torch.zip',
                              'c72329e68a732bef0452e4b96a1c341c8910f81f')
```

Mindkét előre tanított BERT modell tartalmaz egy „vocab.json" fájlt, amely meghatározza a szókészletet,
és egy „pretrained.params" fájlt az előre tanított paraméterekkel.
A következő `load_pretrained_model` függvényt implementáljuk az [**előre tanított BERT paraméterek betöltéséhez**].

```{.python .input}
#@tab mxnet
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_blks, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # Üres szókészletet definiálunk az előre meghatározott szókészlet betöltéséhez
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, ffn_num_hiddens, num_heads, 
                         num_blks, dropout, max_len)
    # Az előre tanított BERT paraméterek betöltése
    bert.load_parameters(os.path.join(data_dir, 'pretrained.params'),
                         ctx=devices)
    return bert, vocab
```

```{.python .input}
#@tab pytorch
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_blks, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # Üres szókészletet definiálunk az előre meghatározott szókészlet betöltéséhez
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(
        len(vocab), num_hiddens, ffn_num_hiddens=ffn_num_hiddens, num_heads=4,
        num_blks=2, dropout=0.2, max_len=max_len)
    # Az előre tanított BERT paraméterek betöltése
    bert.load_state_dict(torch.load(os.path.join(data_dir,
                                                 'pretrained.params')))
    return bert, vocab
```

A legtöbb gépen való szemléltetés megkönnyítése érdekében
ebben a szakaszban az előre tanított BERT kis verzióját („bert.small") töltjük be és hangoljuk finomra.
A feladatok között megmutatjuk, hogyan hangolható finomra a jóval nagyobb „bert.base" a tesztelési pontosság jelentős javítása érdekében.

```{.python .input}
#@tab all
devices = d2l.try_all_gpus()
bert, vocab = load_pretrained_model(
    'bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
    num_blks=2, dropout=0.1, max_len=512, devices=devices)
```

## [**Az adathalmaz a BERT finomhangolásához**]

Az SNLI adathalmazon végzett downstream feladathoz, a természetes nyelvi inferenciához
definiálunk egy testreszabott adathalmazosztályt: `SNLIBERTDataset`.
Minden egyes példában
a premissza és a hipotézis egy szövegszekvencia-párt alkotnak,
amelyeket egyetlen BERT bemeneti szekvenciába csomagolunk, ahogyan a :numref:`fig_bert-two-seqs` ábra szemlélteti.
Visszaidézve a :numref:`subsec_bert_input_rep` szakaszt, a szegmens-azonosítók
a premissza és a hipotézis megkülönböztetésére szolgálnak egy BERT bemeneti szekvencián belül.
A BERT bemeneti szekvencia előre meghatározott maximális hosszával (`max_len`),
a hosszabb bemeneti szöveg utolsó tokenjét addig távolítjuk el, amíg
a `max_len` korlát teljesül.
Az SNLI adathalmaz generálásának gyorsítása érdekében
a BERT finomhangolásához
4 munkásfolyamatot használunk a tanítási és tesztelési példák párhuzamos generálásához.

```{.python .input}
#@tab mxnet
class SNLIBERTDataset(gluon.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]
        
        self.labels = np.array(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # 4 munkásfolyamatot használunk
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (np.array(all_token_ids, dtype='int32'),
                np.array(all_segments, dtype='int32'), 
                np.array(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # Helyet tartunk fenn a '<CLS>', '<SEP>' és '<SEP>' tokeneknek a BERT
        # bemenetén
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab pytorch
class SNLIBERTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]
        
        self.labels = torch.tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # 4 munkásfolyamatot használunk
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long), 
                torch.tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # Helyet tartunk fenn a '<CLS>', '<SEP>' és '<SEP>' tokeneknek a BERT
        # bemenetén
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
```

Az SNLI adathalmaz letöltése után
[**tanítási és tesztelési példákat generálunk**]
az `SNLIBERTDataset` osztály példányosításával.
Ezeket a példákat mini-batchekben olvassuk be a természetes nyelvi inferencia
tanítása és tesztelése során.

```{.python .input}
#@tab mxnet
# Csökkentsd a `batch_size` értékét, ha memóriahiba lép fel. Az eredeti BERT
# modellben `max_len` = 512
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=num_workers)
test_iter = gluon.data.DataLoader(test_set, batch_size,
                                  num_workers=num_workers)
```

```{.python .input}
#@tab pytorch
# Csökkentsd a `batch_size` értékét, ha memóriahiba lép fel. Az eredeti BERT
# modellben `max_len` = 512
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                  num_workers=num_workers)
```

## A BERT finomhangolása

Amint a :numref:`fig_bert-two-seqs` ábra jelzi,
a BERT finomhangolása a természetes nyelvi inferenciához
csupán egy két teljesen összekötött rétegből álló kiegészítő MLP-t igényel
(lásd a `self.hidden` és `self.output` elemeket az alábbi `BERTClassifier` osztályban).
[**Ez az MLP transzformálja a speciális „&lt;cls&gt;" token
BERT-reprezentációját**],
amely mind a premissza, mind a hipotézis információját kódolja,
(**a természetes nyelvi inferencia három kimenetévé**):
következmény (entailment), ellentmondás (contradiction) és semleges (neutral).

```{.python .input}
#@tab mxnet
class BERTClassifier(nn.Block):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Dense(3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
```

```{.python .input}
#@tab pytorch
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.LazyLinear(3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
```

A következőkben
az előre tanított BERT modell `bert` kerül a `BERTClassifier` `net` példányába
a downstream alkalmazás számára.
A BERT finomhangolásának általános implementációiban
csak a kiegészítő MLP kimeneti rétegének paraméterei (`net.output`) kerülnek nulláról tanítva.
Az előre tanított BERT enkóder összes paramétere (`net.encoder`) és a kiegészítő MLP rejtett rétege (`net.hidden`) finomhangolásra kerül.

```{.python .input}
#@tab mxnet
net = BERTClassifier(bert)
net.output.initialize(ctx=devices)
```

```{.python .input}
#@tab pytorch
net = BERTClassifier(bert)
```

Visszaidézve, hogy
a :numref:`sec_bert` szakaszban
a `MaskLM` osztály és a `NextSentencePred` osztály
is rendelkezik paraméterekkel a bennük alkalmazott MLP-kben.
Ezek a paraméterek az előre tanított BERT modell `bert` paramétereinek részét képezik,
és így a `net` paramétereinek is részei.
Azonban ezek a paraméterek csak az
elfedett nyelvi modellezési veszteség
és a következő mondat előrejelzési veszteség kiszámítására szolgálnak
az előtanítás során.
Ez a két veszteségfüggvény nem releváns a downstream alkalmazások finomhangolásához,
ezért a `MaskLM` és `NextSentencePred` osztályokban alkalmazott MLP-k paraméterei nem frissülnek (elavulnak) a BERT finomhangolása során.

Az elavult gradiensekkel rendelkező paraméterek engedélyezéséhez
az `ignore_stale_grad=True` jelző van beállítva a `d2l.train_batch_ch13` `step` függvényében.
Ezt a függvényt használjuk a `net` modell tanítására és kiértékelésére az SNLI
tanítóhalmaza (`train_iter`) és tesztelési halmaza (`test_iter`) segítségével.
A korlátozott számítási erőforrások miatt, [**a tanítási**] és tesztelési pontosság
tovább javítható: megvitatásukat a feladatokra hagyjuk.

```{.python .input}
#@tab mxnet
lr, num_epochs = 1e-4, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices,
               d2l.split_batch_multi_inputs)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 1e-4, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
net(next(iter(train_iter))[0])
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## Összefoglalás

* Az előre tanított BERT modell finomhangolható downstream alkalmazásokhoz, például az SNLI adathalmazon végzett természetes nyelvi inferenciához.
* Finomhangolás során a BERT modell a downstream alkalmazás modelljének részévé válik. Kizárólag az előtanítási veszteséghez kapcsolódó paraméterek nem frissülnek finomhangolás közben.



## Feladatok

1. Ha a számítási erőforrások megengedik, hangolj finomra egy jóval nagyobb előre tanított BERT modellt, amely nagyjából akkora, mint az eredeti BERT alapmodell. Állítsd be a `load_pretrained_model` függvény argumentumait a következőképpen: cseréld le a 'bert.small'-t 'bert.base'-re, és növeld a `num_hiddens=256`, `ffn_num_hiddens=512`, `num_heads=4` és `num_blks=2` értékeket rendre 768-ra, 3072-re, 12-re és 12-re. A finomhangolási epochok növelésével (és esetleg más hiperparaméterek hangolásával) el tudsz-e érni 0,86-nál magasabb tesztelési pontosságot?
1. Hogyan csonkítható egy szekvenciapár a hosszuk aránya alapján? Hasonlítsd össze ezt a pár-csonkítási módszert az `SNLIBERTDataset` osztályban alkalmazottal. Melyek az előnyeik és hátrányaik?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/397)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1526)
:end_tab:
