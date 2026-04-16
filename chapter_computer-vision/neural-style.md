# Neurális stílusátvitel

Ha fotográfiai rajongó vagy,
talán ismered a szűrőket.
Ezek meg tudják változtatni a fotók színstílusát,
így a tájképek élesebbé válhatnak,
vagy a portrék bőrtónusát fehéríthetik.
Egy szűrő azonban általában csak
a fotó egyetlen aspektusát változtatja meg.
Ideális stílus alkalmazásához
egy fotóra
valószínűleg sok különböző
szűrőkombinációt kell kipróbálni.
Ez a folyamat
ugyanolyan összetett, mint egy modell hiperparamétereinek hangolása.



Ebben a fejezetben egy CNN rétegenként kinyert reprezentációit fogjuk kihasználni
arra, hogy az egyik kép stílusát automatikusan alkalmazzuk
egy másik képre, azaz *stílusátvitel* :cite:`Gatys.Ecker.Bethge.2016`.
Ehhez a feladathoz két bemeneti kép szükséges:
az egyik a *tartalomkép*,
a másik a *stíluskép*.
Neurális hálózatokat fogunk használni
a tartalomkép módosításához,
hogy az stílusában közel kerüljön a stílusképhez.
Például
a :numref:`fig_style_transfer` tartalomképe egy általunk készített tájfotó
a seattle-i agglomerációban lévő Mount Rainier Nemzeti Parkban, míg a stíluskép
őszi tölgyfákat ábrázoló olajfestmény.
A kimeneti szintetizált képen
a stíluskép olajfestékes ecsetvonásai
érvényesülnek, ami élénkebb színeket eredményez,
miközben megőrzi a tartalomkép
objektumainak fő alakját.

![Az adott tartalom- és stíluskép alapján a stílusátvitel szintetizált képet állít elő.](../img/style-transfer.svg)
:label:`fig_style_transfer`

## Módszer

A :numref:`fig_style_transfer_model` egy egyszerűsített példán keresztül szemlélteti
a CNN-alapú stílusátviteli módszert.
Először inicializáljuk a szintetizált képet,
például a tartalomkép alapján.
Ez a szintetizált kép az egyetlen változó, amelyet a stílusátviteli folyamat során frissíteni kell,
vagyis a tanítás során frissítendő modellparaméter.
Ezután kiválasztunk egy előtanított CNN-t
a képjellemzők kinyeréséhez, és lefagyasztjuk annak
modellparamétereit a tanítás során.
Ez a mély CNN több réteget használ
a képek hierarchikus jellemzőinek
kinyerésére.
Ezeknek a rétegeknek néhány kimenetét tartalomjellemzőként vagy stílusjellemzőként választhatjuk.
Vegyük példaként a :numref:`fig_style_transfer_model` ábrát.
Az itt szereplő előtanított neurális hálózatnak 3 konvolúciós rétege van,
ahol a második réteg adja a tartalomjellemzőket,
az első és a harmadik réteg pedig a stílusjellemzőket.

![CNN-alapú stílusátviteli folyamat. A folytonos vonalak az előrecsatornázás irányát, a szaggatott vonalak a visszaterjedés irányát jelölik. ](../img/neural-style.svg)
:label:`fig_style_transfer_model`

Ezután előrecsatornázással (folytonos nyilak irányában) kiszámítjuk a stílusátvitel veszteségfüggvényét, és visszaterjedéssel (szaggatott nyilak irányában) frissítjük a modellparamétereket (a kimeneti szintetizált képet).
A stílusátvitelben általánosan használt veszteségfüggvény három részből áll:
(i) a *tartalomveszteség* a szintetizált képet és a tartalomképet a tartalomjellemzők tekintetében közelíti;
(ii) a *stílusveszteség* a szintetizált képet és a stílusképet a stílusjellemzők tekintetében közelíti;
(iii) a *teljes variáció veszteség* segít csökkenteni a zajt a szintetizált képen.
Végül, amikor a modell tanítása befejeződik, a stílusátvitel modellparamétereit kiírjuk,
hogy előállítsuk a végső szintetizált képet.



A következőkben egy konkrét kísérlettel fogjuk megmagyarázni a stílusátvitel technikai részleteit.


## [**A tartalom- és stíluskép beolvasása**]

Először beolvassuk a tartalomképet és a stílusképet.
A kiírt koordinátatengelyekből látható,
hogy ezeknek a képeknek eltérő méretük van.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()

d2l.set_figsize()
content_img = image.imread('../img/rainier.jpg')
d2l.plt.imshow(content_img.asnumpy());
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn

d2l.set_figsize()
content_img = d2l.Image.open('../img/rainier.jpg')
d2l.plt.imshow(content_img);
```

```{.python .input}
#@tab mxnet
style_img = image.imread('../img/autumn-oak.jpg')
d2l.plt.imshow(style_img.asnumpy());
```

```{.python .input}
#@tab pytorch
style_img = d2l.Image.open('../img/autumn-oak.jpg')
d2l.plt.imshow(style_img);
```

## [**Előfeldolgozás és utófeldolgozás**]

Az alábbiakban két függvényt definiálunk a képek előfeldolgozásához és utófeldolgozásához.
A `preprocess` függvény normalizálja
a bemeneti kép mindhárom RGB-csatornáját, és az eredményeket a CNN bemeneti formátumába alakítja.
A `postprocess` függvény visszaállítja a kimeneti kép pixelértékeit az eredeti, normalizálás előtti értékekre.
Mivel a képkiíró függvény megköveteli, hogy minden pixel 0 és 1 közötti lebegőpontos értékkel rendelkezzen,
a 0-nál kisebb vagy az 1-nél nagyobb értékeket 0-val, illetve 1-gyel helyettesítjük.

```{.python .input}
#@tab mxnet
rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32') / 255 - rgb_mean) / rgb_std
    return np.expand_dims(img.transpose(2, 0, 1), axis=0)

def postprocess(img):
    img = img[0].as_in_ctx(rgb_std.ctx)
    return (img.transpose(1, 2, 0) * rgb_std + rgb_mean).clip(0, 1)
```

```{.python .input}
#@tab pytorch
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))
```

## [**Jellemzők kinyerése**]

Az ImageNet adathalmazon előtanított VGG-19 modellt használjuk a képjellemzők kinyeréséhez :cite:`Gatys.Ecker.Bethge.2016`.

```{.python .input}
#@tab mxnet
pretrained_net = gluon.model_zoo.vision.vgg19(pretrained=True)
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.vgg19(pretrained=True)
```

A kép tartalomjellemzőinek és stílusjellemzőinek kinyeréséhez a VGG hálózat bizonyos rétegeinek kimenetét választhatjuk.
Általánosságban: minél közelebb vagyunk a bemeneti réteghez, annál könnyebb kinyerni a kép részleteit, és fordítva, minél közelebb a kimeneti réteghez, annál könnyebb a globális információkat kinyerni. Annak érdekében, hogy ne tartson meg túl sok részletet a tartalomképből a szintetizált képen,
olyan VGG réteget választunk, amely közelebb van a kimenethez, mint *tartalomréteg* a kép tartalomjellemzőinek kinyeréséhez.
A helyi és globális stílusjellemzők kinyeréséhez különböző VGG rétegek kimenetét is kiválasztjuk.
Ezeket a rétegeket *stílusrétegeknek* nevezzük.
Ahogy a :numref:`sec_vgg`-ben is elhangzott,
a VGG hálózat 5 konvolúciós blokkot használ.
A kísérletben a negyedik konvolúciós blokk utolsó konvolúciós rétegét választjuk tartalomrétegként, az egyes konvolúciós blokkok első konvolúciós rétegét pedig stílusrétegként.
Ezeknek a rétegeknek az indexeit a `pretrained_net` példány kiíratásával kapjuk meg.

```{.python .input}
#@tab all
style_layers, content_layers = [0, 5, 10, 19, 28], [25]
```

Amikor VGG rétegekkel nyerünk ki jellemzőket,
csak a bemeneti rétegtől a kimeneti réteghez legközelebb eső tartalomrétegig vagy stílusrétegig terjedő összes réteget kell felhasználnunk.
Hozzunk létre egy új `net` hálózatpéldányt, amely csak a jellemző-kinyeréshez szükséges VGG rétegeket tartja meg.

```{.python .input}
#@tab mxnet
net = nn.Sequential()
for i in range(max(content_layers + style_layers) + 1):
    net.add(pretrained_net.features[i])
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(*[pretrained_net.features[i] for i in
                      range(max(content_layers + style_layers) + 1)])
```

Az `X` bemenet esetén, ha egyszerűen meghívjuk az
előrecsatornázást `net(X)`, csak az utolsó réteg kimenetét kapjuk.
Mivel a közbenső rétegek kimenetére is szükségünk van,
rétegenként kell elvégeznünk a számítást, és meg kell tartanunk
a tartalom- és stílusrétegek kimenetét.

```{.python .input}
#@tab all
def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles
```

Az alábbiakban két függvényt definiálunk:
a `get_contents` függvény kinyeri a tartalomjellemzőket a tartalomképből,
a `get_styles` függvény pedig a stílusjellemzőket nyeri ki a stílusképből.
Mivel a tanítás során nem kell frissíteni az előtanított VGG modellparamétereit,
a tartalom- és stílusjellemzőket akár a tanítás megkezdése előtt is kinyerhetjük.
Mivel a szintetizált kép
a stílusátvitelhez frissítendő modellparaméterek halmaza,
a szintetizált kép tartalom- és stílusjellemzőit csak a tanítás során hívhatjuk ki az `extract_features` függvénnyel.

```{.python .input}
#@tab mxnet
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).copyto(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).copyto(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y
```

```{.python .input}
#@tab pytorch
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y
```

## [**A veszteségfüggvény meghatározása**]

Most leírjuk a stílusátvitel veszteségfüggvényét. A veszteségfüggvény
a tartalomveszteségből, a stílusveszteségből és a teljes variáció veszteségéből áll.

### Tartalomveszteség

A lineáris regresszió veszteségfüggvényéhez hasonlóan
a tartalomveszteség a szintetizált kép és a tartalomkép
közötti tartalomjellemzőbeli különbséget méri
a négyzetes veszteségfüggvénnyel.
A négyzetes veszteségfüggvény két bemenete
mindkettő az `extract_features` függvény által kiszámított
tartalomréteg kimenet.

```{.python .input}
#@tab mxnet
def content_loss(Y_hat, Y):
    return np.square(Y_hat - Y).mean()
```

```{.python .input}
#@tab pytorch
def content_loss(Y_hat, Y):
    # A cél tartalmat leválasztjuk a gradiens dinamikus számításához
    # használt fáról: ez egy rögzített érték, nem változó. Ellenkező
    # esetben a veszteség hibát dobna.
    return torch.square(Y_hat - Y.detach()).mean()
```

### Stílusveszteség

A stílusveszteség, a tartalomveszteséghez hasonlóan,
szintén a négyzetes veszteségfüggvénnyel méri a szintetizált kép és a stíluskép közötti stílusbeli különbséget.
Bármely stílusréteg stíluskimenetének kifejezéséhez
először az `extract_features` függvénnyel
számítjuk ki a stílusréteg kimenetét.
Tegyük fel, hogy a kimenet
1 példányból, $c$ csatornából,
$h$ magasságból és $w$ szélességből áll,
akkor ezt a kimenetet átalakíthatjuk
$\mathbf{X}$ mátrixszá $c$ sorral és $hw$ oszloppal.
Ez a mátrix felfogható
$c$ vektor, $\mathbf{x}_1, \ldots, \mathbf{x}_c$
összefűzéseként,
amelyek mindegyike $hw$ hosszú.
Itt a $\mathbf{x}_i$ vektor az $i$-edik csatorna stílusjellemzőjét jelöli.

Az ezen vektorok *Gram-mátrixában* $\mathbf{X}\mathbf{X}^\top \in \mathbb{R}^{c \times c}$, az $i$-edik sorban és $j$-edik oszlopban lévő $x_{ij}$ elem a $\mathbf{x}_i$ és $\mathbf{x}_j$ vektorok skaláris szorzata.
Ez az $i$ és $j$ csatornák stílusjellemzőinek korrelációját jelöli.
Ezt a Gram-mátrixot használjuk bármely stílusréteg stíluskimenetének ábrázolásához.
Vegyük észre, hogy ha a $hw$ értéke nagyobb,
akkor valószínűleg nagyobb értékek szerepelnek a Gram-mátrixban.
Vegyük észre azt is, hogy a Gram-mátrix magassága és szélessége egyaránt a csatornák $c$ száma.
Annak érdekében, hogy a stílusveszteség ne legyen érzékeny
ezekre az értékekre,
az alábbi `gram` függvény elosztja
a Gram-mátrixot elemeinek számával, azaz $chw$-vel.

```{.python .input}
#@tab all
def gram(X):
    num_channels, n = X.shape[1], d2l.size(X) // X.shape[1]
    X = d2l.reshape(X, (num_channels, n))
    return d2l.matmul(X, X.T) / (num_channels * n)
```

Nyilvánvalóan
a stílusveszteség négyzetes veszteségfüggvényének két Gram-mátrix bemenete
a szintetizált kép és a stíluskép
stílusréteg kimenetein alapul.
Feltételezzük, hogy a `gram_Y` Gram-mátrix a stíluskép alapján előre ki van számítva.

```{.python .input}
#@tab mxnet
def style_loss(Y_hat, gram_Y):
    return np.square(gram(Y_hat) - gram_Y).mean()
```

```{.python .input}
#@tab pytorch
def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()
```

### Teljes variáció veszteség

Néha a tanult szintetizált képen
sok nagyfrekvenciás zaj jelenik meg,
azaz különösen fényes vagy sötét pixelek.
Az egyik elterjedt zajcsökkentési módszer a
*teljes variáció zajcsökkentés*.
Jelöljük $x_{i, j}$-vel az $(i, j)$ koordinátájú pixel értékét.
A teljes variáció veszteségének csökkentése

$$\sum_{i, j} \left|x_{i, j} - x_{i+1, j}\right| + \left|x_{i, j} - x_{i, j+1}\right|$$

közelíti egymáshoz a szomszédos pixelértékeket a szintetizált képen.

```{.python .input}
#@tab all
def tv_loss(Y_hat):
    return 0.5 * (d2l.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  d2l.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())
```

### Veszteségfüggvény

[**A stílusátvitel veszteségfüggvénye a tartalomveszteség, a stílusveszteség és a teljes variáció veszteségének súlyozott összege**].
Ezen súlyhiperparaméterek beállításával
egyensúlyt teremthetünk
a tartalom megőrzése,
a stílusátvitel
és a zajcsökkentés között a szintetizált képen.

```{.python .input}
#@tab all
content_weight, style_weight, tv_weight = 1, 1e4, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # A tartalom-, stílus- és teljes variáció veszteség kiszámítása
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # Az összes veszteség összegzése
    l = sum(styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l
```

## [**A szintetizált kép inicializálása**]

A stílusátvitel során
a szintetizált kép az egyetlen változó, amelyet tanítás közben frissíteni kell.
Ezért definiálhatunk egy egyszerű modellt, a `SynthesizedImage`-t, és a szintetizált képet modellparaméterként kezelhetjük.
Ebben a modellben az előrecsatornázás csak a modellparamétereket adja vissza.

```{.python .input}
#@tab mxnet
class SynthesizedImage(nn.Block):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=img_shape)

    def forward(self):
        return self.weight.data()
```

```{.python .input}
#@tab pytorch
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight
```

Ezután definiáljuk a `get_inits` függvényt.
Ez a függvény létrehoz egy szintetizált kép modellpéldányt, és inicializálja azt az `X` képre.
A stíluskép Gram-mátrixai a különböző stílusrétegeken, `styles_Y_gram`, a tanítás előtt kerülnek kiszámításra.

```{.python .input}
#@tab mxnet
def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape)
    gen_img.initialize(init.Constant(X), ctx=device, force_reinit=True)
    trainer = gluon.Trainer(gen_img.collect_params(), 'adam',
                            {'learning_rate': lr})
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer
```

```{.python .input}
#@tab pytorch
def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer
```

## [**Tanítás**]


A stílusátviteli modell tanítása során
folyamatosan kinyerjük
a szintetizált kép tartalomjellemzőit és stílusjellemzőit, és kiszámítjuk a veszteségfüggvényt.
Az alábbiakban definiáljuk a tanítási ciklust.

```{.python .input}
#@tab mxnet
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs], ylim=[0, 20],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        with autograd.record():
            contents_Y_hat, styles_Y_hat = extract_features(
                X, content_layers, style_layers)
            contents_l, styles_l, tv_l, l = compute_loss(
                X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step(1)
        if (epoch + 1) % lr_decay_epoch == 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.8)
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X).asnumpy())
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X
```

```{.python .input}
#@tab pytorch
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X
```

Most [**elkezdjük a modell tanítását**].
A tartalom- és stíluskép magasságát és szélességét 300 x 450 pixelre méretezzük.
A szintetizált kép inicializálásához a tartalomképet használjuk.

```{.python .input}
#@tab mxnet
device, image_shape = d2l.try_gpu(), (450, 300)
net.collect_params().reset_ctx(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.9, 500, 50)
```

```{.python .input}
#@tab pytorch
device, image_shape = d2l.try_gpu(), (300, 450)  # PIL kép (m, sz)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)
```

Láthatjuk, hogy a szintetizált kép
megőrzi a tartalomkép tájképét és objektumait,
miközben átveszi a stíluskép színvilágát.
Például
a szintetizált képen olyan színfoltok jelennek meg,
mint a stílusképen.
Néhány ilyen folt még az ecsetvonások finom textúráját is megőrzi.




## Összefoglalás

* A stílusátvitelben általánosan használt veszteségfüggvény három részből áll: (i) a tartalomveszteség a szintetizált képet és a tartalomképet a tartalomjellemzők tekintetében közelíti; (ii) a stílusveszteség a szintetizált képet és a stílusképet a stílusjellemzők tekintetében közelíti; (iii) a teljes variáció veszteség segít csökkenteni a zajt a szintetizált képen.
* Egy előtanított CNN-nel kinyerhetjük a képjellemzőket, és a veszteségfüggvény minimalizálásával folyamatosan frissíthetjük a szintetizált képet mint modellparamétert a tanítás során.
* Gram-mátrixokkal ábrázoljuk a stílusrétegek stíluskimeneteit.


## Feladatok

1. Hogyan változik a kimenet, ha különböző tartalom- és stílusrétegeket választunk?
1. Módosítsd a veszteségfüggvény súlyhiperparamétereit! A kimenet több tartalmat őriz meg, vagy kevesebb zajt tartalmaz?
1. Használj különböző tartalom- és stílusképeket! Tudod-e érdekesebb szintetizált képeket létrehozni?
1. Alkalmazható-e a stílusátvitel szövegre is? Tipp: nézd meg a :citet:`10.1145/3544903.3544906` áttekintő tanulmányt.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/378)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1476)
:end_tab:
