# Természetes nyelvi inferencia és az adathalmaz
:label:`sec_natural-language-inference-and-dataset`

A :numref:`sec_sentiment` szakaszban a szentimentelemzés problémáját tárgyaltuk. Ez a feladat egyetlen szövegsorozatot sorol előre meghatározott kategóriákba, például hangulati polaritások halmazába. Azonban ha azt kell eldönteni, hogy egy mondat levezethető-e egy másikból, vagy szemantikailag egyenértékű mondatok azonosításával csökkenteni a redundanciát, akkor egyetlen szövegsorozat osztályozásának ismerete nem elegendő. Ehelyett szövegsorozat-párok felett kell tudnunk következtetni.


## Természetes nyelvi inferencia

A *természetes nyelvi inferencia* azt vizsgálja, hogy egy *hipotézis* levezethető-e egy *premisszából*, ahol mindkettő szövegsorozat. Más szóval a természetes nyelvi inferencia egy szövegsorozat-pár logikai viszonyát határozza meg. Az ilyen viszonyok általában három típusba sorolhatók:

* *Következmény*: a hipotézis levezethető a premisszából.
* *Ellentmondás*: a hipotézis tagadása levezethető a premisszából.
* *Semleges*: minden egyéb eset.

A természetes nyelvi inferencia szöveges következtetés felismerése (recognizing textual entailment) feladatként is ismert. Például a következő pár *következményként* lesz megjelölve, mivel a hipotézisben szereplő „showing affection" kifejezés levezethető a premisszában lévő „hugging one another" fordulatból.

> Premissza: Two women are hugging each other.

> Hipotézis: Two women are showing affection.

A következő példa *ellentmondásra* mutat, mivel a „running the coding example" azt jelenti, hogy „not sleeping", nem pedig „sleeping".

> Premissza: A man is running the coding example from Dive into Deep Learning.

> Hipotézis: The man is sleeping.

A harmadik példa *semleges* viszonyt mutat, mivel a „are performing for us" tényből sem a „famous", sem a „not famous" nem vezethető le.

> Premissza: The musicians are performing for us.

> Hipotézis: The musicians are famous.

A természetes nyelvi inferencia a természetes nyelv megértésének központi témája lett. Alkalmazásai széles körűek: az információkereséstől a nyílt tartományú kérdés-megválaszolásig terjednek. E probléma tanulmányozásához egy népszerű természetes nyelvi inferencia benchmark adathalmaz vizsgálatával kezdünk.


## A Stanford Natural Language Inference (SNLI) adathalmaz

[**A Stanford Natural Language Inference (SNLI) Corpus**] több mint 500 000 megjelölt angol mondatpár gyűjteménye :cite:`Bowman.Angeli.Potts.ea.2015`. Az SNLI adathalmazt letöltjük és a `../data/snli_1.0` útvonalon tároljuk.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import re

npx.set_np()

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
import re

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

### [**Az adathalmaz beolvasása**]

Az eredeti SNLI adathalmaz sokkal gazdagabb információt tartalmaz annál, amire a kísérleteinkben szükségünk van. Ezért definiálunk egy `read_snli` függvényt, amely csak az adathalmaz egy részét nyeri ki, majd premisszák, hipotézisek és azok címkéinek listáit adja vissza.

```{.python .input}
#@tab all
#@save
def read_snli(data_dir, is_train):
    """Az SNLI adathalmazt premisszákba, hipotézisekbe és címkékbe olvassa."""
    def extract_text(s):
        # Eltávolítja az általunk nem használt információt
        s = re.sub('\\(', '', s) 
        s = re.sub('\\)', '', s)
        # Két vagy több egymást követő szóközt egyetlen szóközre cserél
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels
```

Most [**nyomtassuk ki az első 3 premissza–hipotézis párt**] és azok címkéit (a „0", „1" és „2" rendre a „entailment", „contradiction" és „neutral" értékeknek felel meg).

```{.python .input}
#@tab all
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('premise:', x0)
    print('hypothesis:', x1)
    print('label:', y)
```

A tanítóhalmaz körülbelül 550 000 párt tartalmaz, a teszthalmaz pedig körülbelül 10 000 párt. Az alábbi eredmény azt mutatja, hogy [**az „entailment", „contradiction" és „neutral" három címke kiegyensúlyozott**] mind a tanítóhalmazban, mind a teszthalmazban.

```{.python .input}
#@tab all
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```

### [**Osztály definiálása az adathalmaz betöltéséhez**]

Az alábbiakban definiálunk egy osztályt az SNLI adathalmaz betöltéséhez a Gluon `Dataset` osztályából örökölve. Az osztály konstruktorának `num_steps` argumentuma adja meg egy szövegsorozat hosszát, hogy minden minibatch sorozata azonos alakú legyen. Más szóval a hosszabb sorozatokban az első `num_steps` tokenen túli tokeneket levágjuk, míg a rövidebb sorozatokhoz speciális „&lt;pad&gt;" tokeneket fűzünk hozzá, amíg hosszuk el nem éri a `num_steps` értéket. A `__getitem__` függvény implementálásával az `idx` indexszel tetszőlegesen hozzáférhetünk a premisszához, hipotézishez és a címkéhez.

```{.python .input}
#@tab mxnet
#@save
class SNLIDataset(gluon.data.Dataset):
    """Egyéni adathalmaz-osztály az SNLI adathalmaz betöltéséhez."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = np.array(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return np.array([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

```{.python .input}
#@tab pytorch
#@save
class SNLIDataset(torch.utils.data.Dataset):
    """Egyéni adathalmaz-osztály az SNLI adathalmaz betöltéséhez."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

### [**Összerakva az egészet**]

Most meghívhatjuk a `read_snli` függvényt és az `SNLIDataset` osztályt az SNLI adathalmaz letöltéséhez, és `DataLoader` példányokat adunk vissza mind a tanítási, mind a tesztelési készlethez, a tanítóhalmaz szókincsével együtt. Fontos megjegyezni, hogy a teszthalmazhoz is a tanítóhalmazból felépített szókincset kell használni. Ennek következtében a teszthalmazban szereplő bármely új token ismeretlen lesz a tanítóhalmazon tanított modell számára.

```{.python .input}
#@tab mxnet
#@save
def load_data_snli(batch_size, num_steps=50):
    """Letölti az SNLI adathalmazt, és visszaadja az adatiterátorokat és a szókincset."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(test_set, batch_size, shuffle=False,
                                      num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_snli(batch_size, num_steps=50):
    """Letölti az SNLI adathalmazt, és visszaadja az adatiterátorokat és a szókincset."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

Itt a batch méretét 128-ra, a sorozathosszt 50-re állítjuk, és meghívjuk a `load_data_snli` függvényt az adatiterátorok és a szókincs megszerzéséhez. Ezt követően kiírjuk a szókincs méretét.

```{.python .input}
#@tab all
train_iter, test_iter, vocab = load_data_snli(128, 50)
len(vocab)
```

Most kiírjuk az első minibatch alakját. A szentimentelemzéssel ellentétben két bemenetünk van: az `X[0]` és az `X[1]`, amelyek premissza–hipotézis párokat reprezentálnak.

```{.python .input}
#@tab all
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```

## Összefoglalás

* A természetes nyelvi inferencia azt vizsgálja, hogy egy hipotézis levezethető-e egy premisszából, ahol mindkettő szövegsorozat.
* A természetes nyelvi inferenciában a premisszák és hipotézisek közötti viszonyok lehetnek következmény, ellentmondás és semleges jellegűek.
* A Stanford Natural Language Inference (SNLI) Corpus a természetes nyelvi inferencia egyik népszerű benchmark adathalmaza.


## Gyakorlatok

1. A gépi fordítást régóta felszíni $n$-gram-egyezés alapján értékelik a fordítás kimenete és a referencia-fordítás között. Tudnál-e olyan mértéket tervezni a gépi fordítási eredmények értékelésére, amely természetes nyelvi inferenciát használ?
1. Hogyan változtathatjuk meg a hiperparamétereket a szókincs méretének csökkentése érdekében?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/394)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1388)
:end_tab:
