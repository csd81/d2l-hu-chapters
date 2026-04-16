# Jellemzőgazdag ajánlórendszerek

Az interakciós adatok a felhasználói preferenciák és érdeklődés legalapvetőbb jelzései. Az előzőekben bevezetett modellekben kritikus szerepet játszanak. Ugyanakkor az interakciós adatok rendszerint rendkívül ritkák, és néha zajosak is lehetnek. Ennek kezelésére a jellemzők közötti mellékinformációkat, például az elemek jellemzőit, a felhasználói profilokat, sőt még azt is, hogy milyen környezetben történt az interakció, beépíthetjük az ajánlási modellbe. Ezeknek a jellemzőknek a használata segíthet az ajánlásban, mivel különösen akkor lehetnek jó előrejelzői a felhasználói érdeklődésnek, amikor kevés az interakciós adat. Ezért fontos, hogy az ajánlási modellek képesek legyenek ezeket a jellemzőket kezelni, és bizonyos tartalom-/kontextustudatosságot nyújtsanak a modellnek. Az ilyen ajánlási modellek bemutatására bevezetjük az online hirdetési ajánlások kattintási arányával (CTR) foglalkozó feladatot :cite:`McMahan.Holt.Sculley.ea.2013`, és egy anonim hirdetési adathalmazt mutatunk be. A célzott hirdetési szolgáltatások széles körű figyelmet kaptak, és gyakran ajánlórendszerként tekintünk rájuk. Az olyan hirdetések ajánlása, amelyek illeszkednek a felhasználók személyes ízléséhez és érdeklődéséhez, fontos a kattintási arány javításához.


A digitális marketingesek online hirdetéseket használnak, hogy reklámokat jelenítsenek meg az ügyfeleknek. A kattintási arány egy olyan mérőszám, amely azt méri, hogy egy hirdetés hány kattintást kap a megjelenések számához képest, és százalékban fejezzük ki a következő képlettel:

$$ \textrm{CTR} = \frac{\#\textrm{Clicks}} {\#\textrm{Impressions}} \times 100 \% .$$

A kattintási arány fontos jelzés az előrejelző algoritmusok hatékonyságáról. A kattintási arány előrejelzése annak becslését jelenti, hogy egy weboldalon valamit meg fognak-e kattintani. A CTR-előrejelző modellek nemcsak célzott hirdetési rendszerekben, hanem általános elem-ajánló rendszerekben is használhatók, például filmek, hírek vagy termékek esetén, továbbá e-mail kampányokban és még keresőmotorokban is. Emellett szorosan kapcsolódik a felhasználói elégedettséghez és a konverziós arányhoz is, ezért hasznos lehet kampánycélok meghatározásában, mivel segít reális elvárásokat felállítani a hirdetők számára.

```{.python .input}
#@tab mxnet
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
```

## Egy online hirdetési adathalmaz

Az internetes és mobiltechnológiai fejlődésnek köszönhetően az online hirdetés fontos bevételi forrássá vált, és az internetes iparág bevételének túlnyomó részét adja. Fontos olyan releváns hirdetéseket megjeleníteni, amelyek felkeltik a felhasználók érdeklődését, hogy az alkalmi látogatók fizető ügyfelekké válhassanak. Az általunk bemutatott adathalmaz egy online hirdetési adathalmaz. 34 mezőből áll, ahol az első oszlop a célváltozó, és azt jelzi, hogy a hirdetésre kattintottak-e (1) vagy sem (0). Az összes többi oszlop kategorikus jellemző. Az oszlopok jelenthetnek hirdetésazonosítót, oldal- vagy alkalmazásazonosítót, eszközazonosítót, időt, felhasználói profilokat és hasonlókat. A jellemzők valódi jelentése anonimizálás és adatvédelmi okok miatt nem ismert.

A következő kód letölti az adathalmazt a szerverünkről, és elmenti a helyi adatkönyvtárba.

```{.python .input  n=15}
#@tab mxnet
#@save
d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip',
                       'e18327c48c8e8e5c23da714dd614e390d369843f')

data_dir = d2l.download_extract('ctr')
```

Van egy tanító- és egy teszthalmaz, amelyek rendre 15000 és 3000 mintából/sorból állnak.

## Adathalmaz-burkoló

Az adatbetöltés kényelme érdekében megvalósítunk egy `CTRDataset` osztályt, amely a CSV-fájlból tölti be a hirdetési adathalmazt, és `DataLoader`-rel használható.

```{.python .input  n=13}
#@tab mxnet
#@save
class CTRDataset(gluon.data.Dataset):
    def __init__(self, data_path, feat_mapper=None, defaults=None,
                 min_threshold=4, num_feat=34):
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = np.zeros(self.NUM_FEATS, dtype=np.int64)
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                label = np.float32([0, 0])
                label[int(values[0])] = 1
                instance['y'] = [np.float32(values[0])]
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
                    instance.setdefault('x', []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {i: {feat for feat, c in cnt.items() if c >=
                               min_threshold} for i, cnt in feat_cnts.items()}
            self.feat_mapper = {i: {feat_v: idx for idx, feat_v in enumerate(feat_values)}
                                for i, feat_values in feat_mapper.items()}
            self.defaults = {i: len(feat_values) for i, feat_values in feat_mapper.items()}
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        self.offsets = np.array((0, *np.cumsum(self.field_dims).asnumpy()
                                 [:-1]))
        
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        feat = np.array([self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
                         for i, v in enumerate(self.data[idx]['x'])])
        return feat + self.offsets, self.data[idx]['y']
```

Az alábbi példa betölti a tanítóadatokat, és kiírja az első rekordot.

```{.python .input  n=16}
#@tab mxnet
train_data = CTRDataset(os.path.join(data_dir, 'train.csv'))
train_data[0]
```

Látható, hogy mind a 34 mező kategorikus jellemző. Minden érték a megfelelő elem one-hot indexét jelenti. A $0$ címke azt jelenti, hogy nem történt kattintás. Ez a `CTRDataset` más adathalmazok betöltésére is használható, például a Criteo display advertising challenge [adathalmazára](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) és az Avazu click-through rate prediction [adathalmazára](https://www.kaggle.com/c/avazu-ctr-prediction).  

## Összefoglalás 
* A kattintási arány fontos mérőszám a hirdetési rendszerek és ajánlórendszerek hatékonyságának mérésére.
* A kattintási arány előrejelzése általában bináris osztályozási feladattá alakul. A cél annak előrejelzése, hogy egy hirdetés/elem kattintást kap-e a megadott jellemzők alapján.

## Gyakorlatok

* Képes vagy betölteni a Criteo és Avazu adathalmazokat a megadott `CTRDataset` segítségével? Érdemes megjegyezni, hogy a Criteo adathalmaz valós értékű jellemzőket tartalmaz, így lehet, hogy egy kicsit módosítanod kell a kódot.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/405)
:end_tab:
