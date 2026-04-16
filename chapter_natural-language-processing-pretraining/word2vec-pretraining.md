# A word2vec Előtanítása
:label:`sec_word2vec_pretraining`


Folytatjuk a :numref:`sec_word2vec` fejezetben definiált
skip-gram modell implementálásával.
Ezután
előtanítjuk a word2vec-et negatív mintavételezéssel
a PTB adathalmazon.
Mindenekelőtt
hozzuk létre az adatiteratort és a szókincset ehhez az adathalmazhoz
a `d2l.load_data_ptb` függvény meghívásával,
amelyet a :numref:`sec_word2vec_data` fejezetben ismertettünk.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

## A Skip-Gram Modell

A skip-gram modellt
beágyazó rétegek és batch mátrix szorzások segítségével valósítjuk meg.
Először nézzük át,
hogyan működnek a beágyazó rétegek.


### Beágyazó Réteg

Ahogy a :numref:`sec_seq2seq` fejezetben leírtuk,
egy beágyazó réteg
leképezi egy token indexét a jellemzővektorára.
Ennek a rétegnek a súlya
egy mátrix, amelynek soreinek száma egyenlő
a szótár méretével (`input_dim`), és
oszlopainak száma egyenlő
az egyes tokenek vektoros dimenziójával (`output_dim`).
A szóbeágyazási modell betanítása után
ez a súly az, amire szükségünk van.

```{.python .input}
#@tab mxnet
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
```

```{.python .input}
#@tab pytorch
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, '
      f'dtype={embed.weight.dtype})')
```

Egy beágyazó réteg bemenete egy token (szó) indexe.
Bármely $i$ tokenindexhez
a vektoros reprezentációja
a beágyazó réteg súlymátrixának $i$-ik sorából kapható meg.
Mivel a vektor dimenzió (`output_dim`)
4-re van beállítva,
a beágyazó réteg
(2, 3, 4) alakú vektorokat ad vissza
(2, 3) alakú tokenindexekből álló mini-batch esetén.

```{.python .input}
#@tab all
x = d2l.tensor([[1, 2, 3], [4, 5, 6]])
embed(x)
```

### Az Előrefelé Terjedés Definiálása

Az előrefelé terjedés során
a skip-gram modell bemenete tartalmazza
a (batch méret, 1) alakú középső szó indexeket (`center`)
és
a (batch méret, `max_len`) alakú összefűzött kontextus- és zajszó indexeket (`contexts_and_negatives`),
ahol `max_len`
a :numref:`subsec_word2vec-mini-batch-loading` fejezetben van definiálva.
Ez a két változó először a token indexekből vektorokká lesz átalakítva a beágyazó rétegen keresztül,
majd ezek batch mátrix szorzása
(leírva a :numref:`subsec_batch_dot` fejezetben)
egy (batch méret, 1, `max_len`) alakú kimenetet ad.
A kimenetben minden elem egy középső szóvektor és egy kontextus- vagy zajszóvektor skaláris szorzata.

```{.python .input}
#@tab mxnet
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = npx.batch_dot(v, u.swapaxes(1, 2))
    return pred
```

```{.python .input}
#@tab pytorch
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred
```

Nyomtassuk ki a `skip_gram` függvény kimenetének alakját néhány példabemenethez.

```{.python .input}
#@tab mxnet
skip_gram(np.ones((2, 1)), np.ones((2, 4)), embed, embed).shape
```

```{.python .input}
#@tab pytorch
skip_gram(torch.ones((2, 1), dtype=torch.long),
          torch.ones((2, 4), dtype=torch.long), embed, embed).shape
```

## Tanítás

A skip-gram modell negatív mintavételezéssel való tanítása előtt
először definiáljuk a veszteségfüggvényt.


### Bináris Kereszt-entrópia Veszteség

A negatív mintavételezés veszteségfüggvényének definíciója szerint a :numref:`subsec_negative-sampling` fejezetben
a bináris kereszt-entrópia veszteséget fogjuk alkalmazni.

```{.python .input}
#@tab mxnet
loss = gluon.loss.SigmoidBCELoss()
```

```{.python .input}
#@tab pytorch
class SigmoidBCELoss(nn.Module):
    # Bináris kereszt-entrópia veszteség maszkkal
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

loss = SigmoidBCELoss()
```

Idézzük fel a maszk változó
és a címke változó leírását a
:numref:`subsec_word2vec-mini-batch-loading` fejezetből.
A következő
kiszámítja a
bináris kereszt-entrópia veszteséget
az adott változókhoz.

```{.python .input}
#@tab all
pred = d2l.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = d2l.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = d2l.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
```

Az alábbiakban bemutatjuk,
hogyan számítják ki a fenti eredményeket
(kevésbé hatékony módon)
a sigmoid aktivációs függvény alkalmazásával
a bináris kereszt-entrópia veszteségben.
A két kimenetet tekinthetjük
két normalizált veszteségnek,
amelyeket a nem maszolt előrejelzések átlagolásával kapunk.

```{.python .input}
#@tab all
def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

print(f'{(sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4:.4f}')
print(f'{(sigmd(-1.1) + sigmd(-2.2)) / 2:.4f}')
```

### Modellparaméterek Inicializálása

Definiálunk két beágyazó réteget
a szókincs összes szavához,
amelyeket középső szóként és kontextusszóként is használunk.
A szóvektor dimenzió
`embed_size` értéke 100-ra van beállítva.

```{.python .input}
#@tab mxnet
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size))
```

```{.python .input}
#@tab pytorch
embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size))
```

### A Tanítási Ciklus Definiálása

A tanítási ciklus az alábbi módon van definiálva. A tömítés jelenléte miatt a veszteségfüggvény kiszámítása kissé eltér a korábbi tanítási függvényekhez képest.

```{.python .input}
#@tab mxnet
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    net.initialize(ctx=device, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Normalizált veszteségek összege, normalizált veszteségek száma
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            center, context_negative, mask, label = [
                data.as_in_ctx(device) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

```{.python .input}
#@tab pytorch
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(module):
        if type(module) == nn.Embedding:
            nn.init.xavier_uniform_(module.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Normalizált veszteségek összege, normalizált veszteségek száma
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

Most taníthatunk egy skip-gram modellt negatív mintavételezéssel.

```{.python .input}
#@tab all
lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)
```

## Szóbeágyazások Alkalmazása
:label:`subsec_apply-word-embed`


A word2vec modell betanítása után
a betanított modell szóvektorainak koszinusz-hasonlóságával
megtalálhatjuk a szótárból
azokat a szavakat, amelyek szemantikailag a leginkább hasonlítanak
egy bemeneti szóhoz.

```{.python .input}
#@tab mxnet
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[vocab[query_token]]
    # Számítsuk ki a koszinusz-hasonlóságot. 1e-9 hozzáadása numerikus stabilitásért
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # A bemeneti szavak eltávolítása
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

```{.python .input}
#@tab pytorch
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # Számítsuk ki a koszinusz-hasonlóságot. 1e-9 hozzáadása numerikus stabilitásért
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # A bemeneti szavak eltávolítása
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

## Összefoglalás

* A skip-gram modell negatív mintavételezéssel tanítható beágyazó rétegek és bináris kereszt-entrópia veszteség segítségével.
* A szóbeágyazások alkalmazásai közé tartozik a szemantikailag hasonló szavak megkeresése egy adott szóhoz a szóvektorok koszinusz-hasonlósága alapján.


## Gyakorló feladatok

1. A betanított modell segítségével keressünk szemantikailag hasonló szavakat más bemeneti szavakhoz. Javíthatók-e az eredmények a hiperparaméterek hangolásával?
1. Ha a tanítókorpusz hatalmas, a modellparaméterek frissítésekor gyakran mintavételezzük a kontextus- és zajszavakat az aktuális mini-batch középső szavaihoz. Más szóval, ugyanaz a középső szó különböző tanítási korszakokban különböző kontextus- vagy zajszavakkal rendelkezhet. Mik ennek a módszernek az előnyei? Próbáld meg megvalósítani ezt a tanítási módszert.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/384)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1335)
:end_tab:
