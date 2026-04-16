# Teljesen Konvolúciós Hálózatok
:label:`sec_fcn`

Amint azt :numref:`sec_semantic_segmentation`-ban tárgyaltuk,
a szemantikai szegmentálás
pixelszinten osztályozza a képeket.
A teljesen konvolúciós hálózat (FCN)
egy konvolúciós neurális hálózatot használ
a képpontok osztályokba való transzformálásához :cite:`Long.Shelhamer.Darrell.2015`.
A korábban megismert CNN-ektől eltérően,
amelyeket képosztályozáshoz
vagy objektumdetektáláshoz használtunk,
a teljesen konvolúciós hálózat
visszatranszformálja
a közbenső jellemzőtérképek magasságát és szélességét
a bemeneti kép méretére:
ezt a transzponált konvolúciós réteg valósítja meg,
amelyet :numref:`sec_transposed_conv`-ban vezettünk be.
Ennek eredményeképpen
az osztályozási kimenet
és a bemeneti kép
pixelszinten egy-az-egyhez megfeleltetésben áll egymással:
bármely kimeneti pixel csatorna dimenziója
a bemeneti kép ugyanolyan térbeli pozíciójában lévő pixel
osztályozási eredményét tartalmazza.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
```

## A Modell

Itt leírjuk a teljesen konvolúciós hálózat modelljének alapvető tervezési elveit.
Ahogy :numref:`fig_fcn`-ben látható,
ez a modell először egy CNN segítségével kinyeri a kép jellemzőit,
majd a csatornák számát az osztályok számára
transzformálja egy $1\times 1$-es konvolúciós rétegen keresztül,
végül pedig a jellemzőtérképek magasságát és szélességét
a bemeneti kép méretére transzformálja
a :numref:`sec_transposed_conv`-ban bemutatott transzponált konvolúció segítségével.
Ennek eredményeképpen
a modell kimenete ugyanolyan magasságú és szélességű, mint a bemeneti kép,
ahol a kimeneti csatorna a bemeneti kép
ugyanolyan térbeli pozíciójában lévő pixel előre jelzett osztályait tartalmazza.


![Teljesen konvolúciós hálózat.](../img/fcn.svg)
:label:`fig_fcn`

Az alábbiakban [**az ImageNet adathalmazon előre tanított ResNet-18 modellt használjuk a képjellemzők kinyerésére**],
és a modellpéldányt `pretrained_net`-nek nevezzük.
Ennek a modellnek az utolsó néhány rétege
tartalmaz egy globális átlagos összevonó réteget
és egy teljesen összekötött réteget:
ezekre nincs szükség
a teljesen konvolúciós hálózatban.

```{.python .input}
#@tab mxnet
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
pretrained_net.features[-3:], pretrained_net.output
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
list(pretrained_net.children())[-3:]
```

Ezután [**létrehozzuk a `net` teljesen konvolúciós hálózat példányt**].
Ez átmásolja a ResNet-18 összes előre tanított rétegét,
kivéve a kimenethez legközelebb eső
végső globális átlagos összevonó réteget
és a teljesen összekötött réteget.

```{.python .input}
#@tab mxnet
net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(*list(pretrained_net.children())[:-2])
```

Adott egy 320 és 480 magasságú és szélességű bemenet esetén,
a `net` előre terjesztése
az eredeti méret 1/32-ére csökkenti a bemeneti magasságot és szélességet, azaz 10-re és 15-re.

```{.python .input}
#@tab mxnet
X = np.random.uniform(size=(1, 3, 320, 480))
net(X).shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 3, 320, 480))
net(X).shape
```

Ezután [**egy $1\times 1$-es konvolúciós réteggel transzformáljuk a kimeneti csatornák számát a Pascal VOC2012 adathalmaz osztályainak számára (21).**]
Végül (**32-szeresére kell növelnünk a jellemzőtérképek magasságát és szélességét**), hogy visszaállítsuk azokat a bemeneti kép magasságára és szélességére.
Idézzük fel, hogyan számítjuk ki
egy konvolúciós réteg kimeneti alakját :numref:`sec_padding`-ban.
Mivel $(320-64+16\times2+32)/32=10$ és $(480-64+16\times2+32)/32=15$, egy transzponált konvolúciós réteget konstruálunk $32$-es lépésközzel,
a kernel magasságát és szélességét
$64$-re, a kitöltést $16$-ra állítva.
Általában
láthatjuk, hogy
$s$ lépésköz,
$s/2$ kitöltés (feltéve, hogy $s/2$ egész szám),
és $2s$ kernelmagasság és -szélesség esetén,
a transzponált konvolúció $s$-szorosára növeli
a bemenet magasságát és szélességét.

```{.python .input}
#@tab mxnet
num_classes = 21
net.add(nn.Conv2D(num_classes, kernel_size=1),
        nn.Conv2DTranspose(
            num_classes, kernel_size=64, padding=16, strides=32))
```

```{.python .input}
#@tab pytorch
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))
```

## [**Transzponált Konvolúciós Rétegek Inicializálása**]


Már tudjuk, hogy
a transzponált konvolúciós rétegek növelhetik
a jellemzőtérképek
magasságát és szélességét.
A képfeldolgozásban szükség lehet
egy kép felnagyítására, azaz *felülmintavételezésre* (*upsampling*).
A *bilineáris interpoláció*
az egyik leggyakrabban használt felülmintavételezési technika.
Gyakran használják transzponált konvolúciós rétegek inicializálásához is.

A bilineáris interpoláció magyarázataképpen,
tegyük fel, hogy
adott egy bemeneti kép,
és a felülmintavételezett kimeneti kép
minden pixelét ki kell számítanunk.
A kimeneti kép $(x, y)$ koordinátájú pixelének kiszámításához
először képezzük le az $(x, y)$-t a bemeneti kép $(x', y')$ koordinátájára, például a bemeneti és kimeneti méret arányának megfelelően.
Megjegyezzük, hogy a leképezett $x'$ és $y'$ valós számok.
Ezután keressük meg a bemeneti képen
az $(x', y')$ koordinátához legközelebb eső négy pixelt.
Végül a kimeneti kép $(x, y)$ koordinátájú pixelét e négy legközelebbi pixel
és az $(x', y')$-tól való relatív távolságuk alapján számítjuk ki.

A bilineáris interpoláció felülmintavételezése
megvalósítható transzponált konvolúciós réteggel,
amelynek kernelét a következő `bilinear_kernel` függvény állítja elő.
Helyhiány miatt az alábbiakban csak a `bilinear_kernel` függvény implementációját adjuk meg,
az algoritmus tervezésének részletezése nélkül.

```{.python .input}
#@tab mxnet
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (np.arange(kernel_size).reshape(-1, 1),
          np.arange(kernel_size).reshape(1, -1))
    filt = (1 - np.abs(og[0] - center) / factor) * \
           (1 - np.abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return np.array(weight)
```

```{.python .input}
#@tab pytorch
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight
```

Kísérletezzünk [**a bilineáris interpolációval végzett felülmintavételezéssel**],
amelyet egy transzponált konvolúciós réteg valósít meg.
Felépítünk egy transzponált konvolúciós réteget, amely
megduplázza a magasságot és szélességet,
és inicializáljuk a kernelét a `bilinear_kernel` függvénnyel.

```{.python .input}
#@tab mxnet
conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize(init.Constant(bilinear_kernel(3, 3, 4)))
```

```{.python .input}
#@tab pytorch
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4));
```

Olvassuk be az `X` képet, és rendeljük hozzá a felülmintavételezési kimenetet `Y`-hoz. A kép megjelenítéséhez be kell állítanunk a csatorna dimenzió pozícióját.

```{.python .input}
#@tab mxnet
img = image.imread('../img/catdog.jpg')
X = np.expand_dims(img.astype('float32').transpose(2, 0, 1), axis=0) / 255
Y = conv_trans(X)
out_img = Y[0].transpose(1, 2, 0)
```

```{.python .input}
#@tab pytorch
img = torchvision.transforms.ToTensor()(d2l.Image.open('../img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()
```

Amint látható, a transzponált konvolúciós réteg a kép magasságát és szélességét is kétszeresére növeli.
A koordinátákban lévő eltérő méretektől eltekintve
a bilineáris interpolációval felnagyított kép és a :numref:`sec_bbox`-ban megjelenített eredeti kép azonosnak tűnik.

```{.python .input}
#@tab mxnet
d2l.set_figsize()
print('input image shape:', img.shape)
d2l.plt.imshow(img.asnumpy());
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0));
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img);
```

A teljesen konvolúciós hálózatban [**a transzponált konvolúciós réteget bilineáris interpoláció felülmintavételezésével inicializáljuk. Az $1\times 1$-es konvolúciós réteghez Xavier-inicializálást alkalmazunk.**]

```{.python .input}
#@tab mxnet
W = bilinear_kernel(num_classes, num_classes, 64)
net[-1].initialize(init.Constant(W))
net[-2].initialize(init=init.Xavier())
```

```{.python .input}
#@tab pytorch
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W);
```

## [**Az Adathalmaz Beolvasása**]

Beolvassuk
a :numref:`sec_semantic_segmentation`-ban bemutatott
szemantikai szegmentálási adathalmazt.
A véletlen vágás kimeneti képmérete
$320\times 480$-ra van megadva: mind a magasság, mind a szélesség osztható $32$-vel.

```{.python .input}
#@tab all
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)
```

## [**Tanítás**]


Most betaníthatjuk az elkészített
teljesen konvolúciós hálózatot.
Az itt alkalmazott veszteségfüggvény és pontossági számítás
lényegében nem különbözik a korábbi fejezetek képosztályozásánál alkalmazottaktól.
Mivel a transzponált konvolúciós réteg kimeneti csatornáját
használjuk az egyes pixelek osztályának előrejelzésére,
a veszteség kiszámításakor a csatorna dimenziót kell megadni.
Emellett a pontosságot
az összes pixel helyesen előrejelzett osztálya
alapján számítjuk ki.

```{.python .input}
#@tab mxnet
num_epochs, lr, wd, devices = 5, 0.1, 1e-3, d2l.try_all_gpus()
loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
net.collect_params().reset_ctx(devices)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'wd': wd})
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## [**Előrejelzés**]


Az előrejelzés során az egyes csatornákban szabványosítani kell a bemeneti képet,
és a képet a CNN által megkövetelt négydimenzós bemeneti formátumra kell átalakítani.

```{.python .input}
#@tab mxnet
def predict(img):
    X = test_iter._dataset.normalize_image(img)
    X = np.expand_dims(X.transpose(2, 0, 1), axis=0)
    pred = net(X.as_in_ctx(devices[0])).argmax(axis=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

```{.python .input}
#@tab pytorch
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

Az egyes pixelek [**előrejelzett osztályának megjelenítéséhez**] az előrejelzett osztályt visszaképezzük az adathalmazban szereplő megfelelő címkeszínre.

```{.python .input}
#@tab mxnet
def label2image(pred):
    colormap = np.array(d2l.VOC_COLORMAP, ctx=devices[0], dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]
```

```{.python .input}
#@tab pytorch
def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]
```

A tesztadathalmazban lévő képek mérete és alakja eltérő.
Mivel a modell 32-es lépésközű transzponált konvolúciós réteget alkalmaz,
ha egy bemeneti kép magassága vagy szélessége nem osztható 32-vel,
a transzponált konvolúciós réteg kimeneti magassága vagy szélessége eltér a bemeneti kép alakjától.
E probléma kezelésére
a képből több téglalap alakú területet vághatunk ki, amelyek magassága és szélessége 32 egész számszorosai,
és ezeken a területeken külön-külön végzünk előre terjesztést.
Megjegyezzük, hogy
e téglalap alakú területek uniójának teljesen le kell fednie a bemeneti képet.
Ha egy pixelt több téglalap alakú terület is lefed,
az adott pixelre vonatkozó, különböző területeken kapott transzponált konvolúciós kimenetek átlaga
adható be
a softmax-műveletnek
az osztály előrejelzéséhez.


Az egyszerűség kedvéért csupán néhány nagyobb tesztképet olvasunk be,
és a kép bal felső sarkától kezdve $320\times480$ méretű területet vágunk ki az előrejelzéshez.
E tesztképeknél a kivágott területeket,
az előrejelzési eredményeket
és az elvárt kimeneteket soronként jelenítjük meg.

```{.python .input}
#@tab mxnet
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 480, 320)
    X = image.fixed_crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X, pred, image.fixed_crop(test_labels[i], *crop_rect)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

```{.python .input}
#@tab pytorch
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

## Összefoglalás

* A teljesen konvolúciós hálózat először egy CNN segítségével nyeri ki a kép jellemzőit, majd egy $1\times 1$-es konvolúciós rétegen keresztül transzformálja a csatornák számát az osztályok számára, végül pedig transzponált konvolúcióval alakítja át a jellemzőtérképek magasságát és szélességét a bemeneti kép méretére.
* Egy teljesen konvolúciós hálózatban a transzponált konvolúciós réteget bilineáris interpoláció felülmintavételezésével inicializálhatjuk.


## Gyakorlatok

1. Ha a kísérletben Xavier-inicializálást alkalmazunk a transzponált konvolúciós réteghez, hogyan változik az eredmény?
1. Tovább javítható-e a modell pontossága a hiperparaméterek hangolásával?
1. Jósold meg a tesztképek összes pixelének osztályát!
1. Az eredeti teljesen konvolúciós hálózatot bemutató cikk egyes közbenső CNN-rétegek kimenetét is felhasználja :cite:`Long.Shelhamer.Darrell.2015`. Próbáld megvalósítani ezt az ötletet!

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/377)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1582)
:end_tab:
