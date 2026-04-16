# Kétirányú Encoder Reprezentációk Transformerekből (BERT)
:label:`sec_bert`

Bemutattunk több szóbeágyazási modellt a természetes nyelv megértéséhez.
Az előtanítás után a kimenet egy mátrixként képzelhető el,
amelynek minden sora egy vektort képvisel, ami a szótár egy szavát írja le.
Valójában ezek a szóbeágyazási modellek mind *kontextusfüggetlenek*.
Kezdjük ennek a tulajdonságnak a szemléltetésével.


## A Kontextusfüggetlentől a Kontextusérzékenyig

Idézzük fel a :numref:`sec_word2vec_pretraining` és :numref:`sec_synonyms` kísérleteit.
Például a word2vec és a GloVe is ugyanazt az előtanított vektort rendeli ugyanahhoz a szóhoz,
függetlenül a szó kontextusától (ha van ilyen).
Formálisan, bármely $x$ token kontextusfüggetlen reprezentációja
egy $f(x)$ függvény, amely csak $x$-et veszi bemenetként.
A természetes nyelvek gazdag poliszémiájára és összetett szemantikájára tekintettel
a kontextusfüggetlen reprezentációknak nyilvánvaló korlátai vannak.
Például a „crane" szó a „a crane is flying" és a „a crane driver came" kontextusokban
teljesen különböző jelentéssel bír;
így ugyanannak a szónak különböző reprezentációkat lehet rendelni a kontextustól függően.

Ez motiválta a *kontextusérzékeny* szóreprezentációk fejlesztését,
ahol a szavak reprezentációi függnek a kontextusuktól.
Ezért az $x$ token kontextusérzékeny reprezentációja egy $f(x, c(x))$ függvény,
amely mind $x$-től, mind annak $c(x)$ kontextusától függ.
Népszerű kontextusérzékeny reprezentációk:
TagLM (language-model-augmented sequence tagger) :cite:`Peters.Ammar.Bhagavatula.ea.2017`,
CoVe (Context Vectors) :cite:`McCann.Bradbury.Xiong.ea.2017`,
és ELMo (Embeddings from Language Models) :cite:`Peters.Neumann.Iyyer.ea.2018`.

Például, az ELMo a teljes sorozatot veszi bemenetként,
és minden szóhoz egy reprezentációt rendel a bemeneti sorozatból.
Konkrétan az ELMo az előtanított kétirányú LSTM összes közbenső rétegreprezentációját
kombinálja kimeneti reprezentációként.
Az ELMo reprezentáció ezután hozzáadódik egy downstream feladat meglévő felügyelt modelljéhez
kiegészítő jellemzőként, például az ELMo reprezentáció és a tokenek eredeti reprezentációjának
(pl. GloVe) összefűzésével a meglévő modellben.
Egyrészt az előtanított kétirányú LSTM modell összes súlya rögzített,
miután az ELMo reprezentációkat hozzáadták.
Másrészt a meglévő felügyelt modell kifejezetten az adott feladathoz van testre szabva.
Az akkori különböző legjobb modellek felhasználásával az ELMo hozzáadása
hat természetesnyelv-feldolgozási feladatban javította az élvonalbeli eredményeket:
hangulatelemzés, természetes nyelvi következtetés,
szemantikus szerepcímkézés, koreferencia-feloldás,
névelem-felismerés és kérdés-megválaszolás.


## A Feladatspecifikustól a Feladatfüggetlenig

Bár az ELMo jelentősen javította a természetesnyelv-feldolgozási feladatok megoldásait,
minden megoldás még mindig *feladatspecifikus* architektúrán alapul.
Azonban nem magától értetődő egy adott architektúra kialakítása minden egyes
természetesnyelv-feldolgozási feladathoz.
A GPT (Generative Pre-Training) modell egy olyan általános, *feladatfüggetlen* modell
tervezésére irányuló törekvést képvisel, amely kontextusérzékeny reprezentációkat nyújt :cite:`Radford.Narasimhan.Salimans.ea.2018`.
Egy Transformer decoder alapján építkezve a GPT egy nyelvmodellt tanít elő,
amelyet szövegsorozatok reprezentálására használnak.
Amikor a GPT-t egy downstream feladatra alkalmazzák,
a nyelvmodell kimenete egy hozzáadott lineáris kimeneti rétegbe kerül,
hogy megjósolja a feladat címkéjét.
Az ELMo-val ellentétben, amely rögzíti az előtanított modell paramétereit,
a GPT az előtanított Transformer decoder *összes* paraméterét finomhangolja
a downstream feladat felügyelt tanulása során.
A GPT-t természetes nyelvi következtetés, kérdés-megválaszolás,
mondatok hasonlósága és osztályozás tizenkét feladatán értékelték,
és ezek közül kilencben javította az élvonalbeli eredményeket
minimális architektúraváltoztatásokkal.

Azonban a nyelvmodellek autoregresszív jellegéből adódóan
a GPT csak előre tekint (balról jobbra).
Az „i went to the bank to deposit cash" és „i went to the bank to sit down" kontextusokban,
mivel a „bank" érzékeny a bal oldali kontextusra,
a GPT ugyanazt a reprezentációt adja a „bank"-hoz,
holott eltérő jelentésű.


## BERT: A Két Világ Legjobbjainak Kombinálása

Amint láttuk,
az ELMo kétirányúan kódolja a kontextust, de feladatspecifikus architektúrákat használ;
míg a GPT feladatfüggetlen, de balról jobbra kódolja a kontextust.
A két világ legjobbjait ötvözve,
a BERT (Bidirectional Encoder Representations from Transformers)
kétirányúan kódolja a kontextust, és minimális architektúraváltoztatásokat igényel
a természetesnyelv-feldolgozási feladatok széles köréhez :cite:`Devlin.Chang.Lee.ea.2018`.
Egy előtanított Transformer encoder segítségével
a BERT képes bármely tokent annak kétirányú kontextusa alapján reprezentálni.
A downstream feladatok felügyelt tanulása során
a BERT két szempontból hasonlít a GPT-re.
Először is, a BERT reprezentációk egy hozzáadott kimeneti rétegbe kerülnek,
minimális architektúraváltoztatásokkal a feladatok jellegétől függően,
például minden tokenre való jóslás vs. a teljes sorozatra való jóslás.
Másodszor, az előtanított Transformer encoder összes paraméterét finomhangolják,
míg a kiegészítő kimeneti réteg nulláról kerül betanításra.
A :numref:`fig_elmo-gpt-bert` ábra az ELMo, a GPT és a BERT közötti különbségeket mutatja be.

![Az ELMo, a GPT és a BERT összehasonlítása.](../img/elmo-gpt-bert.svg)
:label:`fig_elmo-gpt-bert`


A BERT tovább javította az élvonalbeli eredményeket tizenegy természetesnyelv-feldolgozási feladatban
az alábbi kategóriákban: (i) egyszöveges osztályozás (pl. hangulatelemzés),
(ii) szövegpár-osztályozás (pl. természetes nyelvi következtetés),
(iii) kérdés-megválaszolás, (iv) szövegcímkézés (pl. névelem-felismerés).
Mind 2018-ban javasolva, a kontextusérzékeny ELMo-tól a feladatfüggetlen GPT-ig és BERT-ig,
a természetes nyelvek mély reprezentációinak fogalmilag egyszerű, mégis empirikusan erőteljes előtanítása
forradalmasította a különféle természetesnyelv-feldolgozási feladatok megoldásait.

A fejezet hátralévő részében
a BERT előtanításával foglalkozunk.
Amikor a természetesnyelv-feldolgozási alkalmazásokat a :numref:`chap_nlp_app` fejezetben tárgyaljuk,
bemutatjuk a BERT finomhangolását downstream alkalmazásokhoz.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## [**Bemeneti Reprezentáció**]
:label:`subsec_bert_input_rep`

A természetesnyelv-feldolgozásban
egyes feladatok (pl. hangulatelemzés) egyetlen szöveget vesznek bemenetként,
míg más feladatokban (pl. természetes nyelvi következtetés)
a bemenet szövegsorozatok párja.
A BERT bemeneti sorozat egyértelműen képviseli mind az egyszöveges, mind a szövegpár eseteket.
Az előbbi esetben a BERT bemeneti sorozat az
„&lt;cls&gt;" speciális osztályozási token,
a szövegsorozat tokenjei,
és a „&lt;sep&gt;" speciális elválasztó token összefűzése.
Az utóbbi esetben a BERT bemeneti sorozat a
„&lt;cls&gt;", az első szövegsorozat tokenjei,
„&lt;sep&gt;", a második szövegsorozat tokenjei és „&lt;sep&gt;" összefűzése.
Következetesen megkülönböztetjük a „BERT bemeneti sorozat" terminológiát
a többi „sorozat" típustól.
Például egy *BERT bemeneti sorozat* tartalmazhat egy *szövegsorozatot* vagy két *szövegsorozatot*.

A szövegpárok megkülönböztetéséhez
a tanult szegmensbeágyazásokat $\mathbf{e}_A$ és $\mathbf{e}_B$
hozzáadják az első, illetve a második sorozat tokenbeágyazásaihoz.
Egyszöveges bemenet esetén csak $\mathbf{e}_A$ kerül felhasználásra.

Az alábbi `get_tokens_and_segments` függvény egy vagy két mondatot vesz bemenetként,
majd visszaadja a BERT bemeneti sorozat tokenjeit
és a megfelelő szegmens-azonosítókat.

```{.python .input}
#@tab all
#@save
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """A BERT bemeneti sorozat tokenjeit és szegmens-azonosítóit adja vissza."""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # A 0 és 1 jelöli az A és a B szegmenst
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments
```

A BERT a Transformer encodert választja kétirányú architektúraként.
A Transformer encoderben megszokott módon
pozícionális beágyazásokat adnak hozzá a BERT bemeneti sorozat minden pozíciójához.
Az eredeti Transformer encodertől eltérően azonban
a BERT *tanulható* pozícionális beágyazásokat alkalmaz.
Összefoglalva, a :numref:`fig_bert-input` ábra mutatja, hogy
a BERT bemeneti sorozat beágyazásai a tokenbeágyazások,
a szegmensbeágyazások és a pozícionális beágyazások összege.

![A BERT bemeneti sorozat beágyazásai a tokenbeágyazások, a szegmensbeágyazások
és a pozícionális beágyazások összege.](../img/bert-input.svg)
:label:`fig_bert-input`

Az alábbi [**`BERTEncoder` osztály**] hasonló a :numref:`sec_transformer` fejezetben
megvalósított `TransformerEncoder` osztályhoz.
A `TransformerEncoder`-től eltérően a `BERTEncoder`
szegmensbeágyazásokat és tanulható pozícionális beágyazásokat használ.

```{.python .input}
#@tab mxnet
#@save
class BERTEncoder(nn.Block):
    """BERT kódoló."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout, max_len=1000, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for _ in range(num_blks):
            self.blks.add(d2l.TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, True))
        # A BERT-ben a pozícionális beágyazások tanulhatók, ezért elegendően
        # hosszú pozícionális beágyazás paramétert hozunk létre
        self.pos_embedding = self.params.get('pos_embedding',
                                             shape=(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Az `X` alakja változatlan marad a következő kódrészletben:
        # (kötegméret, maximális sorozathossz, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data(ctx=X.ctx)[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

```{.python .input}
#@tab pytorch
#@save
class BERTEncoder(nn.Module):
    """BERT kódoló."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout, max_len=1000, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", d2l.TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, True))
        # A BERT-ben a pozícionális beágyazások tanulhatók, ezért elegendően
        # hosszú pozícionális beágyazás paramétert hozunk létre
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Az `X` alakja változatlan marad a következő kódrészletben:
        # (kötegméret, maximális sorozathossz, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

Tegyük fel, hogy a szótár mérete 10000.
A `BERTEncoder` előre irányú [**inferenciájának**] szemléltetéséhez
hozzunk létre egy példányt, és inicializáljuk a paramétereit.

```{.python .input}
#@tab mxnet
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
num_blks, dropout = 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                      num_blks, dropout)
encoder.initialize()
```

```{.python .input}
#@tab pytorch
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
ffn_num_input, num_blks, dropout = 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                      num_blks, dropout)
```

A `tokens` változót 2 db, 8 hosszúságú BERT bemeneti sorozatként definiáljuk,
ahol minden token a szótár egy indexe.
A `BERTEncoder` előre irányú inferenciája a `tokens` bemenettel
a kódolt eredményt adja vissza, ahol minden tokent egy vektor reprezentál,
amelynek hosszát a `num_hiddens` hiperparaméter határozza meg előre.
Ezt a hiperparamétert általában a Transformer encoder *rejtett méretének*
(rejtett egységek száma) nevezik.

```{.python .input}
#@tab mxnet
tokens = np.random.randint(0, vocab_size, (2, 8))
segments = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

```{.python .input}
#@tab pytorch
tokens = torch.randint(0, vocab_size, (2, 8))
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

## Előtanítási Feladatok
:label:`subsec_bert_pretraining_tasks`

A `BERTEncoder` előre irányú inferenciája megadja a BERT reprezentációját
a bemeneti szöveg minden tokenjéhez, beleértve a beillesztett
„&lt;cls&gt;" és „&lt;seq&gt;" speciális tokeneket is.
Ezután ezeket a reprezentációkat fogjuk felhasználni
a BERT előtanításának veszteségfüggvényének kiszámításához.
Az előtanítás az alábbi két feladatból áll:
maszolt nyelvmodellezés és következő mondat jóslása.

### [**Maszolt Nyelvmodellezés**]
:label:`subsec_mlm`

Amint azt a :numref:`sec_language-model` fejezetben bemutattuk,
egy nyelvmodell a bal oldali kontextus alapján jósol meg egy tokent.
A kontextus kétirányú kódolásához az egyes tokenek reprezentálása céljából
a BERT véletlenszerűen maszol tokeneket, és a kétirányú kontextus tokenjeit használja
a maszolt tokenek önfelügyelt módon való jóslásához.
Ezt a feladatot *maszolt nyelvmodellnek* nevezzük.

Ebben az előtanítási feladatban a tokenek 15%-át választják ki véletlenszerűen
maszolt tokenként jóslás céljára.
Ahhoz, hogy a maszolt tokent a címke felhasználása nélkül jósolják meg,
egy egyszerű megközelítés az lenne, hogy mindig egy speciális „&lt;mask&gt;" tokenre
cserélik a BERT bemeneti sorozatban.
Azonban a mesterséges speciális „&lt;mask&gt;" token soha nem jelenik meg
a finomhangolás során.
Az előtanítás és a finomhangolás közötti ilyen eltérés elkerülése érdekében,
ha egy tokent maszolnak jóslás céljára (pl. a „great" szó van kiválasztva
maszolásra és jóslásra a „this movie is great" mondatban),
a bemenetben az alábbiak valamelyikére cserélik:

* speciális „&lt;mask&gt;" tokenre az esetek 80%-ában (pl. „this movie is great" → „this movie is &lt;mask&gt;");
* véletlenszerű tokenre az esetek 10%-ában (pl. „this movie is great" → „this movie is drink");
* a változatlan eredeti tokenre az esetek 10%-ában (pl. „this movie is great" → „this movie is great").

Megjegyezzük, hogy a 15% 10%-ában véletlenszerű tokent illesztenek be.
Ez az alkalmi zaj arra ösztönzi a BERT-et, hogy kevésbé legyen elfogult a maszolt tokennel szemben
(különösen akkor, amikor a címketoken változatlan marad) a kétirányú kontextuskódolás során.

Az alábbi `MaskLM` osztályt valósítjuk meg a maszolt tokenek jóslásához
a BERT előtanítás maszolt nyelvmodell feladatában.
A jóslás egy egyrétegű MLP-t (`self.mlp`) használ.
Az előre irányú inferencia során két bemenetet vesz:
a `BERTEncoder` kódolt eredményét és a jósláshoz szükséges tokenek pozícióit.
A kimenet a jóslási eredmények ezeken a pozíciókon.

```{.python .input}
#@tab mxnet
#@save
class MaskLM(nn.Block):
    """A BERT maszolt nyelvmodell feladata."""
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential()
        self.mlp.add(
            nn.Dense(num_hiddens, flatten=False, activation='relu'))
        self.mlp.add(nn.LayerNorm())
        self.mlp.add(nn.Dense(vocab_size, flatten=False))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = np.arange(0, batch_size)
        # Tegyük fel, hogy `batch_size` = 2, `num_pred_positions` = 3, ekkor
        # `batch_idx` értéke `np.array([0, 0, 0, 1, 1, 1])`
        batch_idx = np.repeat(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

```{.python .input}
#@tab pytorch
#@save
class MaskLM(nn.Module):
    """A BERT maszolt nyelvmodell feladata."""
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.LazyLinear(num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.LazyLinear(vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Tegyük fel, hogy `batch_size` = 2, `num_pred_positions` = 3, ekkor
        # `batch_idx` értéke `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

A [**`MaskLM` előre irányú inferenciájának**] bemutatásához
hozzuk létre a `mlm` példányát és inicializáljuk.
Emlékeztetőül: a `BERTEncoder` előre irányú inferenciájából kapott `encoded_X`
2 BERT bemeneti sorozatot reprezentál.
Az `mlm_positions` változót 3 indexként definiáljuk, amelyeket
az `encoded_X` bármelyik BERT bemeneti sorozatában kell jósolni.
Az `mlm` előre irányú inferenciája visszaadja a `mlm_Y_hat` jóslási eredményeket
az `encoded_X` összes `mlm_positions` maszolt pozíciójára.
Minden jósláshoz az eredmény mérete egyenlő a szótár méretével.

```{.python .input}
#@tab mxnet
mlm = MaskLM(vocab_size, num_hiddens)
mlm.initialize()
mlm_positions = np.array([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

```{.python .input}
#@tab pytorch
mlm = MaskLM(vocab_size, num_hiddens)
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

A maszolt `mlm_Y_hat` jósolt tokenek valódi `mlm_Y` címkéivel
kiszámíthatjuk a maszolt nyelvmodell feladat keresztentrópia-veszteségét
a BERT előtanításban.

```{.python .input}
#@tab mxnet
mlm_Y = np.array([[7, 8, 9], [10, 20, 30]])
loss = gluon.loss.SoftmaxCrossEntropyLoss()
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

```{.python .input}
#@tab pytorch
mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

### [**Következő Mondat Jóslása**]
:label:`subsec_nsp`

Bár a maszolt nyelvmodellezés képes kétirányúan kódolni a kontextust
a szavak reprezentálásához, nem modellezi explicit módon
a szövegpárok logikai kapcsolatát.
A két szövegsorozat közötti kapcsolat megértésének segítésére
a BERT egy bináris osztályozási feladatot, a *következő mondat jóslását* veszi figyelembe
az előtanítás során.
Mondatpárok generálásakor az előtanításhoz
az esetek felében valóban egymást követő mondatok szerepelnek „True" (igaz) címkével;
míg az esetek másik felében a második mondat véletlenszerűen kerül mintavételezésre
a korpuszból „False" (hamis) címkével.

Az alábbi `NextSentencePred` osztály egyrétegű MLP-t használ
annak jóslásához, hogy a második mondat az első mondat következő mondatja-e
a BERT bemeneti sorozatban.
A Transformer encoder önfigyelme miatt
a „&lt;cls&gt;" speciális token BERT reprezentációja
mindkét bemeneti mondatot kódolja.
Ezért az MLP osztályozó kimeneti rétege (`self.output`) az `X`-et veszi bemenetként,
ahol `X` az MLP rejtett rétegének kimenete, amelynek bemenete a kódolt „&lt;cls&gt;" token.

```{.python .input}
#@tab mxnet
#@save
class NextSentencePred(nn.Block):
    """A BERT következő mondat jóslási feladata."""
    def __init__(self, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Dense(2)

    def forward(self, X):
        # `X` alakja: (kötegméret, `num_hiddens`)
        return self.output(X)
```

```{.python .input}
#@tab pytorch
#@save
class NextSentencePred(nn.Module):
    """A BERT következő mondat jóslási feladata."""
    def __init__(self, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.LazyLinear(2)

    def forward(self, X):
        # `X` alakja: (kötegméret, `num_hiddens`)
        return self.output(X)
```

Láthatjuk, hogy [**egy `NextSentencePred` példány előre irányú inferenciája**]
bináris jóslásokat ad vissza minden BERT bemeneti sorozathoz.

```{.python .input}
#@tab mxnet
nsp = NextSentencePred()
nsp.initialize()
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

```{.python .input}
#@tab pytorch
# A PyTorch alapértelmezés szerint nem lapítja ki a tenzort, ellentétben
# az mxnet-tel, ahol flatten=True esetén az összes tengely (az első
# kivételével) összeolvad
encoded_X = torch.flatten(encoded_X, start_dim=1)
# NSP bemeneti alakja: (kötegméret, `num_hiddens`)
nsp = NextSentencePred()
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

A 2 bináris osztályozás keresztentrópia-vesztesége szintén kiszámítható.

```{.python .input}
#@tab mxnet
nsp_y = np.array([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

```{.python .input}
#@tab pytorch
nsp_y = torch.tensor([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

Megjegyzendő, hogy a fent említett előtanítási feladatok mindkét feladatának
összes címkéje triviálisan megszerezhető az előtanítási korpuszból
manuális címkézési erőfeszítés nélkül.
Az eredeti BERT-et a BookCorpus :cite:`Zhu.Kiros.Zemel.ea.2015`
és az angol Wikipédia összefűzésén tanították elő.
Ez a két szöveges korpusz hatalmas méretű:
800 millió, illetve 2,5 milliárd szót tartalmaz.


## [**Az Egész Összerakása**]

A BERT előtanításakor a végső veszteségfüggvény
a maszolt nyelvmodellezés és a következő mondat jóslásának
veszteségfüggvényeinek lineáris kombinációja.
Most definiálhatjuk a `BERTModel` osztályt a három osztály,
a `BERTEncoder`, a `MaskLM` és a `NextSentencePred` példányosításával.
Az előre irányú inferencia visszaadja a kódolt BERT reprezentációkat `encoded_X`,
a maszolt nyelvmodellezés jóslásait `mlm_Y_hat`,
és a következő mondat jóslásait `nsp_Y_hat`.

```{.python .input}
#@tab mxnet
#@save
class BERTModel(nn.Block):
    """A BERT modell."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout, max_len=1000):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_blks, dropout, max_len)
        self.hidden = nn.Dense(num_hiddens, activation='tanh')
        self.mlm = MaskLM(vocab_size, num_hiddens)
        self.nsp = NextSentencePred()

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # Az MLP osztályozó rejtett rétege a következő mondat jóslásához.
        # A 0 a '<cls>' token indexe
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

```{.python .input}
#@tab pytorch
#@save
class BERTModel(nn.Module):
    """A BERT modell."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, 
                 num_heads, num_blks, dropout, max_len=1000):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_blks, dropout,
                                   max_len=max_len)
        self.hidden = nn.Sequential(nn.LazyLinear(num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens)
        self.nsp = NextSentencePred()

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # Az MLP osztályozó rejtett rétege a következő mondat jóslásához.
        # A 0 a '<cls>' token indexe
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

## Összefoglalás

* Az olyan szóbeágyazási modellek, mint a word2vec és a GloVe, kontextusfüggetlenek. Ugyanazt az előtanított vektort rendelik ugyanahhoz a szóhoz, függetlenül a kontextustól (ha van ilyen). Nehezen kezelik a poliszémiát vagy a természetes nyelvek összetett szemantikáját.
* A kontextusérzékeny szóreprezentációk esetén, mint az ELMo és a GPT, a szavak reprezentációi függnek a kontextusuktól.
* Az ELMo kétirányúan kódolja a kontextust, de feladatspecifikus architektúrákat alkalmaz (ugyanakkor nem egyszerű minden egyes természetesnyelv-feldolgozási feladathoz specifikus architektúrát kialakítani); míg a GPT feladatfüggetlen, de balról jobbra kódolja a kontextust.
* A BERT ötvözi a két világ legjobbjait: kétirányúan kódolja a kontextust, és minimális architektúraváltoztatásokat igényel a természetesnyelv-feldolgozási feladatok széles köréhez.
* A BERT bemeneti sorozat beágyazásai a tokenbeágyazások, a szegmensbeágyazások és a pozícionális beágyazások összege.
* A BERT előtanítása két feladatból áll: maszolt nyelvmodellezés és következő mondat jóslása. Az előbbi képes kétirányúan kódolni a kontextust a szavak reprezentálásához, míg az utóbbi explicit módon modellezi a szövegpárok logikai kapcsolatát.


## Feladatok

1. Minden más körülmény azonossága esetén több vagy kevesebb előtanítási lépésre lesz szüksége egy maszolt nyelvmodellnek a konvergenciához, mint egy balról jobbra haladó nyelvmodellnek? Miért?
1. Az eredeti BERT implementációban a `BERTEncoder` pozíciónkénti előrecsatoló hálózata (a `d2l.TransformerEncoderBlock`-on keresztül) és a `MaskLM` teljesen összekötött rétege egyaránt a Gauss-hibás lineáris egységet (GELU) :cite:`Hendrycks.Gimpel.2016` használja aktivációs függvényként. Kutass a GELU és a ReLU közötti különbségek után.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/388)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1490)
:end_tab:
