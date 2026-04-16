```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['pytorch', 'jax'])
```

# Transformerek a számítógépes látásban
:label:`sec_vision-transformer`

A Transformer architektúrát kezdetben
szekvencia-szekvencia tanulásra javasolták,
a gépi fordítást helyezve előtérbe.
Ezt követően a Transformerek a modellválasztás alapjává váltak
különféle természetes nyelvfeldolgozási feladatokban :cite:`Radford.Narasimhan.Salimans.ea.2018,Radford.Wu.Child.ea.2019,brown2020language,Devlin.Chang.Lee.ea.2018,raffel2020exploring`.
Azonban a számítógépes látás területén
az uralkodó architektúra a
CNN maradt (:numref:`chap_modern_cnn`).
Természetesen a kutatók elkezdtek azon gondolkodni,
hogy jobb eredményeket lehetne-e elérni
Transformer modellek képadatokhoz való adaptálásával.
Ez a kérdés hatalmas érdeklődést keltett
a számítógépes látás közösségében.
Nemrég :citet:`ramachandran2019stand` javasolt
egy sémát a konvolúció önfigyelemre való cseréjéhez.
Azonban a figyelemben speciális minták alkalmazása
megnehezíti a modellek skálázását hardveres gyorsítókon.
Ezután :citet:`cordonnier2020relationship` elméletileg bizonyította,
hogy az önfigyelem megtanulhat hasonlóan viselkedni, mint a konvolúció.
Empirikusan $2 \times 2$ méretű képfoltokat vettünk bemeneti adatként,
de a kis foltméret a modellt
csak alacsony felbontású képadatokra teszi alkalmazhatóvá.

A foltméret-korlátozások nélkül
a *vision Transformerek* (ViT-ek)
foltokat vonnak ki a képekből
és betáplálják azokat egy Transformer kódolóba
egy globális reprezentáció megszerzéséhez,
amelyet végül osztályozáshoz transzformálnak :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`.
Figyelemre méltó, hogy a Transformerek jobb skálázhatóságot mutatnak, mint a CNN-ek:
és nagyobb adathalmazokon nagyobb modellek tanításakor
a vision Transformerek jelentős különbséggel felülmúlják a ResNet-eket.
A természetes nyelvfeldolgozás hálózatarchitektúra-tervezési tájképéhez hasonlóan
a Transformerek is játékváltóvá váltak a számítógépes látásban.

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## Modell

A :numref:`fig_vit` mutatja
a vision Transformerek modellarchitektúráját.
Ez az architektúra egy törzsből áll, amely foltokra bontja a képeket,
egy többrétegű Transformer kódolón alapuló főrészből,
és egy fejből, amely a globális reprezentációt
a kimeneti címkévé transzformálja.

![A vision Transformer architektúra. Ebben a példában egy képet kilenc foltra osztanak. Egy speciális "&lt;cls&gt;" token és a kilenc laposított képfolt patch-beágyazáson és $\mathit{n}$ Transformer kódoló blokkon keresztül tíz reprezentációvá transzformálódnak. A "&lt;cls&gt;" reprezentáció tovább transzformálódik a kimeneti címkévé.](../img/vit.svg)
:label:`fig_vit`

Tekintsünk egy $h$ magasságú, $w$ szélességű
és $c$ csatornájú bemeneti képet.
A folt magasságát és szélességét egyaránt $p$-re megadva,
a kép $m = hw/p^2$ foltból álló szekvenciára bomlik szét,
ahol minden folt $cp^2$ hosszúságú vektorrá lapítódik.
Így a képfoltok hasonlóan kezelhetők, mint a szöveges szekvenciák tokenjei a Transformer kódolók által.
Egy speciális "&lt;cls&gt;" (osztály) token és
az $m$ laposított képfolt lineárisan vetül
$m+1$ vektorból álló szekvenciává,
amelyeket tanítható pozícióbeágyazásokkal összegznek.
A többrétegű Transformer kódoló
$m+1$ bemeneti vektort transzformál
azonos számú azonos hosszúságú kimeneti vektoros reprezentációvá.
Pontosan ugyanúgy működik, mint az eredeti Transformer kódoló a :numref:`fig_transformer`-ben,
csak a normalizálás helyzetében tér el.
Mivel a "&lt;cls&gt;" token az összes képfoltra figyel
az önfigyelmen keresztül (lásd :numref:`fig_cnn-rnn-self-attention`),
a Transformer kódoló kimenetéből kapott reprezentációja
tovább transzformálódik a kimeneti címkévé.

## Patch-beágyazás

Egy vision Transformer implementálásához kezdjük
a :numref:`fig_vit`-ben látható patch-beágyazással.
Egy kép foltokra osztása
és e laposított foltok lineáris vetítése
egyetlen konvolúciós műveletté egyszerűsíthető,
ahol mind a kernel-méret, mind a lépésköz mérete a foltméretre van beállítva.

```{.python .input}
%%tab pytorch
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # A kimenet alakja: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)
```

```{.python .input}
%%tab jax
class PatchEmbedding(nn.Module):
    img_size: int = 96
    patch_size: int = 16
    num_hiddens: int = 512

    def setup(self):
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(self.img_size), _make_tuple(self.patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.Conv(self.num_hiddens, kernel_size=patch_size,
                            strides=patch_size, padding='SAME')

    def __call__(self, X):
        # A kimenet alakja: (batch size, no. of patches, no. of channels)
        X = self.conv(X)
        return X.reshape((X.shape[0], -1, X.shape[3]))
```

A következő példában `img_size` magasságú és szélességű képeket véve bemenetként,
a patch-beágyazás `(img_size//patch_size)**2` foltot ad ki,
amelyek lineárisan `num_hiddens` hosszúságú vektorokba vetülnek.

```{.python .input}
%%tab pytorch
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = d2l.zeros(batch_size, 3, img_size, img_size)
d2l.check_shape(patch_emb(X),
                (batch_size, (img_size//patch_size)**2, num_hiddens))
```

```{.python .input}
%%tab jax
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = d2l.zeros((batch_size, img_size, img_size, 3))
output, _ = patch_emb.init_with_output(d2l.get_key(), X)
d2l.check_shape(output, (batch_size, (img_size//patch_size)**2, num_hiddens))
```

## Vision Transformer kódoló
:label:`subsec_vit-encoder`

A vision Transformer kódoló MLP-je kissé eltér
az eredeti Transformer kódoló pozíciószerinti FFN-jétől
(lásd :numref:`subsec_positionwise-ffn`).
Először itt az aktiválási függvény a Gauss-hiba lineáris egységet (GELU) használja,
amelyet a ReLU simított változatának tekinthetünk :cite:`Hendrycks.Gimpel.2016`.
Másodszor, dropout-ot alkalmaznak az MLP-ben minden teljes összeköttetésű réteg kimenetére regularizáció céljából.

```{.python .input}
%%tab pytorch
class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(
            self.dense1(x)))))
```

```{.python .input}
%%tab jax
class ViTMLP(nn.Module):
    mlp_num_hiddens: int
    mlp_num_outputs: int
    dropout: float = 0.5

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(self.mlp_num_hiddens)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        x = nn.Dense(self.mlp_num_outputs)(x)
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        return x
```

A vision Transformer kódoló blokk implementációja
a :numref:`fig_vit`-ben lévő elő-normalizálási tervezést követi,
ahol a normalizálás közvetlenül *a* többfejű figyelem vagy az MLP *előtt* kerül alkalmazásra.
Ellentétben az utó-normalizálással (az „add & norm" a :numref:`fig_transformer`-ben),
ahol a normalizálás közvetlenül *a* maradékkapcsolatok *után* kerül,
az elő-normalizálás hatékonyabb vagy jobb tanítást eredményez a Transformereknél :cite:`baevski2018adaptive,wang2019learning,xiong2020layer`.

```{.python .input}
%%tab pytorch
class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens,
                 num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, X, valid_lens=None):
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))
```

```{.python .input}
%%tab jax
class ViTBlock(nn.Module):
    num_hiddens: int
    mlp_num_hiddens: int
    num_heads: int
    dropout: float
    use_bias: bool = False

    def setup(self):
        self.attention = d2l.MultiHeadAttention(self.num_hiddens, self.num_heads,
                                                self.dropout, self.use_bias)
        self.mlp = ViTMLP(self.mlp_num_hiddens, self.num_hiddens, self.dropout)

    @nn.compact
    def __call__(self, X, valid_lens=None, training=False):
        X = X + self.attention(*([nn.LayerNorm()(X)] * 3),
                               valid_lens, training=training)[0]
        return X + self.mlp(nn.LayerNorm()(X), training=training)
```

Ahogy a :numref:`subsec_transformer-encoder`-ban,
egyetlen vision Transformer kódoló blokk sem változtatja meg a bemenet alakját.

```{.python .input}
%%tab pytorch
X = d2l.ones((2, 100, 24))
encoder_blk = ViTBlock(24, 24, 48, 8, 0.5)
encoder_blk.eval()
d2l.check_shape(encoder_blk(X), X.shape)
```

```{.python .input}
%%tab jax
X = d2l.ones((2, 100, 24))
encoder_blk = ViTBlock(24, 48, 8, 0.5)
d2l.check_shape(encoder_blk.init_with_output(d2l.get_key(), X)[0], X.shape)
```

## Mindent összehozva

A vision Transformerek előreterjesztése az alábbiakban egyszerű.
Először a bemeneti képek betáplálódnak egy `PatchEmbedding` példányba,
amelynek kimenete össze van fűzve a "&lt;cls&gt;" token beágyazásával.
Tanítható pozícióbeágyazásokkal összeadva, mielőtt dropout-ot alkalmaznak.
Ezután a kimenet betáplálódik a `ViTBlock` osztály `num_blks` példányát egymásra rakó Transformer kódolóba.
Végül a "&lt;cls&gt;" token reprezentációja a hálózat feje által vetítődik.

```{.python .input}
%%tab pytorch
class ViT(d2l.Classifier):
    """Vision Transformer."""
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout, lr=0.1,
                 use_bias=False, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(d2l.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias))
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens),
                                  nn.Linear(num_hiddens, num_classes))

    def forward(self, X):
        X = self.patch_embedding(X)
        X = d2l.concat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])
```

```{.python .input}
%%tab jax
class ViT(d2l.Classifier):
    """Vision Transformer."""
    img_size: int
    patch_size: int
    num_hiddens: int
    mlp_num_hiddens: int
    num_heads: int
    num_blks: int
    emb_dropout: float
    blk_dropout: float
    lr: float = 0.1
    use_bias: bool = False
    num_classes: int = 10
    training: bool = False

    def setup(self):
        self.patch_embedding = PatchEmbedding(self.img_size, self.patch_size,
                                              self.num_hiddens)
        self.cls_token = self.param('cls_token', nn.initializers.zeros,
                                    (1, 1, self.num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = self.param('pos_embed', nn.initializers.normal(),
                                        (1, num_steps, self.num_hiddens))
        self.blks = [ViTBlock(self.num_hiddens, self.mlp_num_hiddens,
                              self.num_heads, self.blk_dropout, self.use_bias)
                    for _ in range(self.num_blks)]
        self.head = nn.Sequential([nn.LayerNorm(), nn.Dense(self.num_classes)])

    @nn.compact
    def __call__(self, X):
        X = self.patch_embedding(X)
        X = d2l.concat((jnp.tile(self.cls_token, (X.shape[0], 1, 1)), X), 1)
        X = nn.Dropout(emb_dropout, deterministic=not self.training)(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X, training=self.training)
        return self.head(X[:, 0])
```

## Tanítás

A vision Transformer tanítása a Fashion-MNIST adathalmazon pontosan olyan, mint ahogy a CNN-eket tanítják a :numref:`chap_modern_cnn`-ben.

```{.python .input}
%%tab all
img_size, patch_size = 96, 16
num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
            num_blks, emb_dropout, blk_dropout, lr)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(img_size, img_size))
trainer.fit(model, data)
```

## Összefoglalás és Vita

Észreveheted, hogy kis adathalmazok esetén, mint a Fashion-MNIST,
az implementált vision Transformerünk
nem múlja felül a :numref:`sec_resnet`-ben lévő ResNet-et.
Hasonló megfigyelések az ImageNet adathalmazon is tehetők (1,2 millió kép).
Ennek oka az, hogy a Transformerek *nélkülözik* azokat a hasznos elveket a konvolúcióban,
mint az eltolás-invariancia és a lokalitás (:numref:`sec_why-conv`).
Azonban a kép megváltozik, ha nagyobb modelleket tanítunk nagyobb adathalmazokon (pl. 300 millió kép),
ahol a vision Transformerek nagy különbséggel felülmúlják a ResNet-eket a képosztályozásban, demonstrálva
a Transformerek belső fölényét a skálázhatóság szempontjából :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`.
A vision Transformerek bevezetése
megváltoztatta a képadatok modellezésének hálózattervezési tájképét.
Hamarosan hatékonynak bizonyultak az ImageNet adathalmazon
a DeiT adathatékony tanítási stratégiáival :cite:`touvron2021training`.
Azonban az önfigyelem négyzetes bonyolultsága
(:numref:`sec_self-attention-and-positional-encoding`)
a Transformer architektúrát
kevésbé alkalmassá teszi a magasabb felbontású képekhez.
Az általános célú gerinchálózat irányában a számítógépes látásban,
a Swin Transformerek megoldották a képmérettel kapcsolatos négyzetes számítási bonyolultságot
(:numref:`subsec_cnn-rnn-self-attention`)
és visszaállították a konvolúció-szerű előfeltevéseket,
kiterjesztve a Transformerek alkalmazhatóságát a számítógépes látási feladatok széles körére
a képosztályozáson túl, legkorszerűbb eredményekkel :cite:`liu2021swin`.

## Feladatok

1. Hogyan befolyásolja az `img_size` értéke a tanítási időt?
1. A "&lt;cls&gt;" token reprezentáció kimenetté vetítése helyett hogyan vetítenéd az átlagolt foltreprezentációkat? Implementáld ezt a változtatást, és nézd meg, hogyan befolyásolja a pontosságot.
1. Módosíthatod-e a hiperparamétereket a vision Transformer pontosságának javítása érdekében?

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/8943)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18032)
:end_tab:
