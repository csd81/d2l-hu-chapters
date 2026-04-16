# A BERT előtanítás adathalmaza
:label:`sec_bert-dataset`

A :numref:`sec_bert`-ben megvalósított BERT modell előtanításához
olyan formátumban kell előállítani az adathalmazt, amely támogatja
a két előtanítási feladatot:
a maszkolt nyelvi modellezést és a következő mondat előrejelzését.
Egyrészt az eredeti BERT modell a BookCorpus és az English Wikipedia
két hatalmas korpusz összefűzésén van előtanítva (lásd: :numref:`subsec_bert_pretraining_tasks`),
ami a legtöbb olvasó számára nehezen futtatható.
Másrészt az előtanított BERT modell
nem feltétlenül alkalmas adott szakterületi alkalmazásokra, mint például az orvostudomány.
Ezért egyre népszerűbbé vált a BERT testreszabott adathalmazon való előtanítása.
A BERT előtanítás szemléltetéséhez
egy kisebb WikiText-2 korpuszt használunk :cite:`Merity.Xiong.Bradbury.ea.2016`.

A :numref:`sec_word2vec_data`-ban a word2vec előtanításához használt PTB adathalmazhoz képest
a WikiText-2: (i) megőrzi az eredeti írásjeleket, így alkalmas a következő mondat előrejelzésére; (ii) megőrzi az eredeti kis- és nagybetűket, valamint a számokat; (iii) több mint kétszer nagyobb.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import random

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import os
import random
import torch
```

[**A WikiText-2 adathalmazban**]
minden sor egy bekezdést jelöl, ahol
szóköz kerül minden írásjel és az azt megelőző token közé.
Legalább két mondatot tartalmazó bekezdések kerülnek megőrzésre.
A mondatok szétválasztásához egyszerűség kedvéért csak a pontot használjuk elválasztóként.
A bonyolultabb mondatszétválasztási technikák tárgyalását a szakasz végén
található gyakorlatokra hagyjuk.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

#@save
def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # A nagybetűket kisbetűkre alakítjuk
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs
```

## Az előtanítási feladatokhoz szükséges segédfüggvények meghatározása

A következőkben először megvalósítjuk a két BERT előtanítási feladat segédfüggvényeit:
a következő mondat előrejelzéséhez és a maszkolt nyelvi modellezéshez szükségeseket.
Ezeket a segédfüggvényeket később, a nyers szövegkorpusz
ideális formátumú BERT előtanítási adathalmazzá alakításakor hívjuk meg.

### [**A következő mondat előrejelzési feladat előállítása**]

A :numref:`subsec_nsp` leírásai alapján
a `_get_next_sentence` függvény egy tanítópéldát állít elő
a bináris osztályozási feladathoz.

```{.python .input}
#@tab all
#@save
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # A `paragraphs` listák listájának listája
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next
```

A következő függvény a következő mondat előrejelzéséhez
állít elő tanítópéldákat a bemeneti `paragraph`-ból, a `_get_next_sentence` függvény meghívásával.
Itt a `paragraph` mondatok listája, ahol minden mondat tokenek listája.
A `max_len` argumentum a BERT bemeneti szekvencia maximális hosszát adja meg az előtanítás során.

```{.python .input}
#@tab all
#@save
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # 1 '<cls>' tokent és 2 '<sep>' tokent is számításba veszünk
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph
```

### [**A maszkolt nyelvi modellezési feladat előállítása**]
:label:`subsec_prepare_mlm_data`

Ahhoz, hogy a maszkolt nyelvi modellezési feladathoz
tanítópéldákat állítsunk elő egy BERT bemeneti szekvenciából,
definiáljuk az alábbi `_replace_mlm_tokens` függvényt.
A bemenetek között a `tokens` egy BERT bemeneti szekvenciát alkotó tokenek listája,
a `candidate_pred_positions` a BERT bemeneti szekvencia token-indexeinek listája,
kizárva a speciális tokenek indexeit (a speciális tokeneket nem jósoljuk meg a maszkolt nyelvi modellezési feladatban),
a `num_mlm_preds` pedig a megjósolandó tokenek számát jelzi (visszaidézve: az összes token 15%-a).
A maszkolt nyelvi modellezési feladat :numref:`subsec_mlm`-beli definíciójának megfelelően
minden predikciós pozícióban a bemeneti token helyére kerülhet
egy speciális "&lt;mask&gt;" token, egy véletlenszerű token, vagy maradhat változatlan.
Végül a függvény visszaadja a lehetséges csere után kapott bemeneti tokeneket,
a predikciós pozíciók token-indexeit, valamint a megfelelő címkéket.

```{.python .input}
#@tab all
#@save
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    # A maszkolt nyelvi modell bemenetéhez készítünk egy másolatot a tokenekről,
    # és néhányat lecserélünk '<mask>'-re vagy véletlenszerű tokenekre
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # Véletlenszerűsítés: a maszkolt nyelvi modellezési feladatban
    # a megjósolandó 15%-nyi véletlenszerű token kiválasztásához
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80% eséllyel: a szót lecseréljük a '<mask>' tokenre
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10% eséllyel: a szó változatlan marad
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10% eséllyel: a szót egy véletlenszerű szóra cseréljük
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels
```

A fent említett `_replace_mlm_tokens` függvény meghívásával
az alábbi függvény egy BERT bemeneti szekvenciát (`tokens`) kap bemenetként,
és visszaadja a bemeneti tokenek indexeit
(a :numref:`subsec_mlm`-ben leírt lehetséges tokencserék után),
a predikciós pozíciók token-indexeit,
valamint ezekhez a predikciókhoz tartozó címke-indexeket.

```{.python .input}
#@tab all
#@save
def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # A `tokens` egy sztringek listája
    for i, token in enumerate(tokens):
        # A speciális tokeneket nem jósoljuk meg a maszkolt
        # nyelvi modellezési feladatban
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # A maszkolt nyelvi modellezési feladatban a tokenek 15%-át jósoljuk meg
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]
```

## A szöveg előtanítási adathalmazzá alakítása

Most már közel járunk ahhoz, hogy testreszabjunk egy `Dataset` osztályt a BERT előtanításához.
Előtte azonban még szükségünk van egy `_pad_bert_inputs` segédfüggvényre,
amely [**speciális "&lt;pad&gt;" tokeneket fűz a bemenetekhez.**]
Az `examples` argumentuma a `_get_nsp_data_from_paragraph` és a `_get_mlm_data_from_tokens`
segédfüggvények kimeneteit tartalmazza a két előtanítási feladathoz.

```{.python .input}
#@tab mxnet
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(np.array(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype='int32'))
        all_segments.append(np.array(segments + [0] * (
            max_len - len(segments)), dtype='int32'))
        # A `valid_lens` nem számolja a '<pad>' tokeneket
        valid_lens.append(np.array(len(token_ids), dtype='float32'))
        all_pred_positions.append(np.array(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype='int32'))
        # A kitöltő tokenekre vonatkozó predikciók a veszteségből
        # 0 súlyokkal való szorzás útján szűrjük ki
        all_mlm_weights.append(
            np.array([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)), dtype='float32'))
        all_mlm_labels.append(np.array(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype='int32'))
        nsp_labels.append(np.array(is_next))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

```{.python .input}
#@tab pytorch
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # A `valid_lens` nem számolja a '<pad>' tokeneket
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # A kitöltő tokenekre vonatkozó predikciók a veszteségből
        # 0 súlyokkal való szorzás útján szűrjük ki
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

A két előtanítási feladat tanítópéldáit előállító segédfüggvényeket
és a bemeneteket kitöltő segédfüggvényt összefogva
az alábbi `_WikiTextDataset` osztályt hozzuk létre, mint [**a BERT előtanításhoz használt WikiText-2 adathalmaz**].
A `__getitem__` függvény implementálásával
tetszőlegesen elérhetjük a WikiText-2 korpusz mondatpárjaiból
generált előtanítási (maszkolt nyelvi modellezési és következő mondat előrejelzési) példákat.

Az eredeti BERT modell WordPiece beágyazásokat használ, amelyek szókészletének mérete 30 000 :cite:`Wu.Schuster.Chen.ea.2016`.
A WordPiece tokenizálási módszer az eredeti bájt-pár kódolási algoritmus :numref:`subsec_Byte_Pair_Encoding`-beli enyhe módosítása.
Az egyszerűség kedvéért a `d2l.tokenize` függvényt használjuk tokenizáláshoz.
Az ötnél kevesebbszer előforduló ritka tokeneket kiszűrjük.

```{.python .input}
#@tab mxnet
#@save
class _WikiTextDataset(gluon.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # A bemeneti `paragraphs[i]` mondatsztringek listája, amely egy bekezdést
        # jelöl; a kimeneti `paragraphs[i]` szintén bekezdést jelöl, de mondatok
        # listájaként, ahol minden mondat tokenek listája
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Adatok előállítása a következő mondat előrejelzési feladathoz
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Adatok előállítása a maszkolt nyelvi modellezési feladathoz
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Bemenetek kitöltése
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab pytorch
#@save
class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # A bemeneti `paragraphs[i]` mondatsztringek listája, amely egy bekezdést
        # jelöl; a kimeneti `paragraphs[i]` szintén bekezdést jelöl, de mondatok
        # listájaként, ahol minden mondat tokenek listája
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Adatok előállítása a következő mondat előrejelzési feladathoz
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Adatok előállítása a maszkolt nyelvi modellezési feladathoz
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Bemenetek kitöltése
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

A `_read_wiki` függvény és a `_WikiTextDataset` osztály felhasználásával
definiáljuk az alábbi `load_data_wiki` függvényt, amely [**letölti a WikiText-2 adathalmazt
és előállítja belőle az előtanítási példákat**].

```{.python .input}
#@tab mxnet
#@save
def load_data_wiki(batch_size, max_len):
    """A WikiText-2 adathalmaz betöltése."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    return train_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_wiki(batch_size, max_len):
    """A WikiText-2 adathalmaz betöltése."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab
```

A batch méretét 512-re, a BERT bemeneti szekvencia maximális hosszát 64-re állítva
[**kiírjuk egy BERT előtanítási minibatch alakjait**].
Fontos megjegyezni, hogy minden BERT bemeneti szekvenciában
$10$ ($64 \times 0.15$) pozíció kerül megjóslásra a maszkolt nyelvi modellezési feladatban.

```{.python .input}
#@tab all
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)

for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
     mlm_Y, nsp_y) in train_iter:
    print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
          pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
          nsp_y.shape)
    break
```

Végül nézzük meg a szókészlet méretét.
Még a ritka tokenek kiszűrése után is
több mint kétszer nagyobb a PTB adathalmazénál.

```{.python .input}
#@tab all
len(vocab)
```

## Összefoglalás

* A PTB adathalmazhoz képest a WikiText-2 adathalmaz megőrzi az eredeti írásjeleket, kis- és nagybetűket, valamint számokat, és több mint kétszer nagyobb.
* Tetszőlegesen elérhetjük a WikiText-2 korpusz mondatpárjaiból generált előtanítási (maszkolt nyelvi modellezési és következő mondat előrejelzési) példákat.


## Gyakorlatok

1. Az egyszerűség kedvéért csak a pontot használjuk mondatszétválasztóként. Próbálj ki más mondatszétválasztási technikákat, például a spaCy-t vagy az NLTK-t. Vedd az NLTK-t példaként. Először telepítsd: `pip install nltk`. A kódban először `import nltk`. Majd töltsd le a Punkt mondattokenizálót: `nltk.download('punkt')`. A `sentences = 'This is great ! Why not ?'` mondatok szétválasztásához az `nltk.tokenize.sent_tokenize(sentences)` meghívása két mondatból álló listát ad vissza: `['This is great !', 'Why not ?']`.
1. Mekkora a szókészlet mérete, ha egyetlen ritka tokent sem szűrsz ki?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/389)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1496)
:end_tab:
