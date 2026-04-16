# A BERT előtanítása
:label:`sec_bert-pretraining`

A :numref:`sec_bert`-ben megvalósított BERT modellel
és a :numref:`sec_bert-dataset`-ben a WikiText-2 adathalmazból előállított előtanítási példákkal
ebben a szakaszban előtanítjuk a BERT-et a WikiText-2 adathalmazon.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

Először betöltjük a WikiText-2 adathalmazt a maszkolt nyelvi modellezés és a következő mondat előrejelzése
előtanítási példáinak mini-batch-jeként.
A batch mérete 512, a BERT bemeneti szekvencia maximális hossza 64.
Megjegyezzük, hogy az eredeti BERT modellben a maximális hossz 512.

```{.python .input}
#@tab all
batch_size, max_len = 512, 64
train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)
```

## A BERT előtanítása

Az eredeti BERT-nek két, eltérő modellméretű változata létezik :cite:`Devlin.Chang.Lee.ea.2018`.
Az alap modell ($\textrm{BERT}_{\textrm{BASE}}$) 12 réteget (transformer kódoló blokkot) használ
768 rejtett egységgel (rejtett méret) és 12 önfigyelmi fejjel.
A nagy modell ($\textrm{BERT}_{\textrm{LARGE}}$) 24 réteget használ
1024 rejtett egységgel és 16 önfigyelmi fejjel.
Az előbbi 110 millió, az utóbbi 340 millió paraméterrel rendelkezik.
A szemléltetés egyszerűsége érdekében
[**egy kis BERT-et definiálunk, 2 réteggel, 128 rejtett egységgel és 2 önfigyelmi fejjel**].

```{.python .input}
#@tab mxnet
net = d2l.BERTModel(len(vocab), num_hiddens=128, ffn_num_hiddens=256,
                    num_heads=2, num_blks=2, dropout=0.2)
devices = d2l.try_all_gpus()
net.initialize(init.Xavier(), ctx=devices)
loss = gluon.loss.SoftmaxCELoss()
```

```{.python .input}
#@tab pytorch
net = d2l.BERTModel(len(vocab), num_hiddens=128, 
                    ffn_num_hiddens=256, num_heads=2, num_blks=2, dropout=0.2)
devices = d2l.try_all_gpus()
loss = nn.CrossEntropyLoss()
```

A tanítási ciklus definiálása előtt
meghatározzuk a `_get_batch_loss_bert` segédfüggvényt.
A tanítópéldák egy szeletét kapva
ez a függvény [**kiszámítja a maszkolt nyelvi modellezési és a következő mondat előrejelzési feladatok veszteségét**].
Megjegyezzük, hogy a BERT előtanítás végső veszteége
csupán a maszkolt nyelvi modellezési veszteség
és a következő mondat előrejelzési veszteség összege.

```{.python .input}
#@tab mxnet
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X_shards,
                         segments_X_shards, valid_lens_x_shards,
                         pred_positions_X_shards, mlm_weights_X_shards,
                         mlm_Y_shards, nsp_y_shards):
    mlm_ls, nsp_ls, ls = [], [], []
    for (tokens_X_shard, segments_X_shard, valid_lens_x_shard,
         pred_positions_X_shard, mlm_weights_X_shard, mlm_Y_shard,
         nsp_y_shard) in zip(
        tokens_X_shards, segments_X_shards, valid_lens_x_shards,
        pred_positions_X_shards, mlm_weights_X_shards, mlm_Y_shards,
        nsp_y_shards):
        # Előre menet
        _, mlm_Y_hat, nsp_Y_hat = net(
            tokens_X_shard, segments_X_shard, valid_lens_x_shard.reshape(-1),
            pred_positions_X_shard)
        # A maszkolt nyelvi modellezési veszteség kiszámítása
        mlm_l = loss(
            mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y_shard.reshape(-1),
            mlm_weights_X_shard.reshape((-1, 1)))
        mlm_l = mlm_l.sum() / (mlm_weights_X_shard.sum() + 1e-8)
        # A következő mondat előrejelzési veszteség kiszámítása
        nsp_l = loss(nsp_Y_hat, nsp_y_shard)
        nsp_l = nsp_l.mean()
        mlm_ls.append(mlm_l)
        nsp_ls.append(nsp_l)
        ls.append(mlm_l + nsp_l)
        npx.waitall()
    return mlm_ls, nsp_ls, ls
```

```{.python .input}
#@tab pytorch
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # Előre menet
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # A maszkolt nyelvi modellezési veszteség kiszámítása
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # A következő mondat előrejelzési veszteség kiszámítása
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l
```

A fent említett két segédfüggvényt meghívva
az alábbi `train_bert` függvény
definiálja a [**BERT (`net`) előtanítási eljárását a WikiText-2 (`train_iter`) adathalmazon**].
A BERT tanítása nagyon hosszú ideig tarthat.
Ahelyett, hogy a tanítási korszakok számát adnánk meg
a `train_ch13` függvényhez hasonlóan (lásd: :numref:`sec_image_augmentation`),
az alábbi függvény `num_steps` bemeneti paramétere
a tanítási iterációs lépések számát adja meg.

```{.python .input}
#@tab mxnet
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': 0.01})
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # A maszkolt nyelvi modellezési veszteségek összege, a következő mondat
    # előrejelzési veszteségek összege, mondatpárok száma, darabszám
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for batch in train_iter:
            (tokens_X_shards, segments_X_shards, valid_lens_x_shards,
             pred_positions_X_shards, mlm_weights_X_shards,
             mlm_Y_shards, nsp_y_shards) = [gluon.utils.split_and_load(
                elem, devices, even_split=False) for elem in batch]
            timer.start()
            with autograd.record():
                mlm_ls, nsp_ls, ls = _get_batch_loss_bert(
                    net, loss, vocab_size, tokens_X_shards, segments_X_shards,
                    valid_lens_x_shards, pred_positions_X_shards,
                    mlm_weights_X_shards, mlm_Y_shards, nsp_y_shards)
            for l in ls:
                l.backward()
            trainer.step(1)
            mlm_l_mean = sum([float(l) for l in mlm_ls]) / len(mlm_ls)
            nsp_l_mean = sum([float(l) for l in nsp_ls]) / len(nsp_ls)
            metric.add(mlm_l_mean, nsp_l_mean, batch[0].shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```

```{.python .input}
#@tab pytorch
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net(*next(iter(train_iter))[:4])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # A maszkolt nyelvi modellezési veszteségek összege, a következő mondat
    # előrejelzési veszteségek összege, mondatpárok száma, darabszám
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```

A BERT előtanítása során megjeleníthetjük mind a maszkolt nyelvi modellezési, mind a következő mondat előrejelzési veszteséget.

```{.python .input}
#@tab all
train_bert(train_iter, net, loss, len(vocab), devices, 50)
```

## [**Szöveg reprezentálása BERT-tel**]

A BERT előtanítása után
felhasználhatjuk egyetlen szöveg, szövegpárok, vagy azok bármelyik tokenjének reprezentálására.
Az alábbi függvény visszaadja a BERT (`net`) reprezentációit
a `tokens_a` és `tokens_b` összes tokenjére.

```{.python .input}
#@tab mxnet
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = np.expand_dims(np.array(vocab[tokens], ctx=devices[0]),
                               axis=0)
    segments = np.expand_dims(np.array(segments, ctx=devices[0]), axis=0)
    valid_len = np.expand_dims(np.array(len(tokens), ctx=devices[0]), axis=0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

```{.python .input}
#@tab pytorch
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

[**Vegyük az "a crane is flying" mondatot.**]
Idézzük fel a BERT bemeneti reprezentációját, ahogy azt a :numref:`subsec_bert_input_rep` tárgyalja.
A speciális "&lt;cls&gt;" (osztályozáshoz használt)
és "&lt;sep&gt;" (szétválasztáshoz használt) tokenek beillesztése után
a BERT bemeneti szekvencia hossza hat.
Mivel a nulla a "&lt;cls&gt;" token indexe,
az `encoded_text[:, 0, :]` a teljes bemeneti mondat BERT reprezentációja.
A „crane" többjelentésű token kiértékeléséhez
kiírjuk a token BERT reprezentációjának első három elemét is.

```{.python .input}
#@tab all
tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
# Tokenek: '<cls>', 'a', 'crane', 'is', 'flying', '<sep>'
encoded_text_cls = encoded_text[:, 0, :]
encoded_text_crane = encoded_text[:, 2, :]
encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3]
```

[**Most vegyük az "a crane driver came" és "he just left" mondatpárt.**]
Hasonlóan, az `encoded_pair[:, 0, :]` az egész mondatpár kódolt eredménye az előtanított BERT-től.
Fontos, hogy a „crane" többjelentésű token első három eleme eltér attól, ami más kontextusban szerepel.
Ez igazolja, hogy a BERT reprezentációk kontextusérzékenyek.

```{.python .input}
#@tab all
tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
# Tokenek: '<cls>', 'a', 'crane', 'driver', 'came', '<sep>', 'he', 'just',
# 'left', '<sep>'
encoded_pair_cls = encoded_pair[:, 0, :]
encoded_pair_crane = encoded_pair[:, 2, :]
encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3]
```

A :numref:`chap_nlp_app`-ban egy előtanított BERT modellt fogunk finomhangolni
természetes nyelvfeldolgozási downstream alkalmazásokhoz.

## Összefoglalás

* Az eredeti BERT-nek két változata van: az alap modellnek 110 millió, a nagy modellnek 340 millió paramétere van.
* A BERT előtanítása után felhasználhatjuk egyetlen szöveg, szövegpárok, vagy azok bármelyik tokenjének reprezentálására.
* A kísérletben ugyanannak a tokennek eltérő BERT reprezentációja van, ha a kontextusa különböző. Ez igazolja, hogy a BERT reprezentációk kontextusérzékenyek.


## Gyakorlatok

1. A kísérletben látható, hogy a maszkolt nyelvi modellezési veszteség lényegesen nagyobb a következő mondat előrejelzési veszteségnél. Miért?
2. Állítsuk a BERT bemeneti szekvencia maximális hosszát 512-re (ugyanannyi, mint az eredeti BERT modellben). Használjuk az eredeti BERT modell konfigurációját, például a $\textrm{BERT}_{\textrm{LARGE}}$-ot. Tapasztalunk-e hibát a szakasz futtatásakor? Miért?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/390)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1497)
:end_tab:
