# objektumdetektálás és befoglaló téglalapok
:label:`sec_bbox`


A korábbi fejezetekben (pl. :numref:`sec_alexnet`--:numref:`sec_googlenet`) különféle képosztályozási modelleket mutattunk be.
A képosztályozási feladatokban feltételezzük, hogy a képen *egyetlen* fő objektum van, és csak arra összpontosítunk, hogyan ismerjük fel annak kategóriáját.
Azonban a vizsgált képeken gyakran *több* objektum is megtalálható.
Nemcsak a kategóriájukat szeretnénk tudni, hanem a képen belüli konkrét pozíciójukat is.
A számítógépes látásban az ilyen feladatokat *objektumdetektálásnak* (vagy *objektumdetektálásnak*) nevezzük.

Az objektumdetektálást számos területen széles körben alkalmazzák.
Például az önvezető autóknak az útvonalat kell megtervezniük az elfogott videóképeken lévő járművek, gyalogosok, utak és akadályok pozíciójának felismerésével.
Emellett a robotok e technikát alkalmazhatják az érdeklődési objektumok felismerésére és helymeghatározására, miközben a környezetükben navigálnak.
Továbbá a biztonsági rendszereknek szükségük lehet rendellenes objektumok, például betolakodók vagy bombák felismerésére.

A következő néhány fejezetben bemutatunk néhány deep learning alapú objektumdetektálási módszert.
Az objektumok *pozíciójának* (vagy *helyzetének*) bemutatásával kezdjük.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, npx, np

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

Betöltjük az ebben a fejezetben használt mintaképet. Láthatjuk, hogy a kép bal oldalán egy kutya, jobb oldalán egy macska található.
Ez a kép két fő objektuma.

```{.python .input}
#@tab mxnet
d2l.set_figsize()
img = image.imread('../img/catdog.jpg').asnumpy()
d2l.plt.imshow(img);
```

```{.python .input}
#@tab pytorch, tensorflow
d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
d2l.plt.imshow(img);
```

## Befoglaló téglalapok

Az objektumdetektálásban általában *befoglaló téglalapot* használunk az objektum térbeli elhelyezkedésének leírására.
A befoglaló téglalap téglalap alakú, amelyet a téglalap bal felső sarkának $x$ és $y$ koordinátái, valamint a jobb alsó sarok koordinátái határoznak meg.
Egy másik általánosan használt befoglaló téglalap ábrázolás a befoglaló téglalap középpontjának $(x, y)$-tengelykoordinátái, valamint a doboz szélessége és magassága.

[**Itt definiálunk függvényeket a**] (**két ábrázolás közötti konverzióhoz**):
a `box_corner_to_center` a két sarokponttal megadott ábrázolásból közép-szélesség-magasság ábrázolásba konvertál,
a `box_center_to_corner` pedig fordítva.
A `boxes` bemeneti argumentumnak egy $(n, 4)$ alakú kétdimenziós tenzornak kell lennie, ahol $n$ a befoglaló téglalapok száma.

```{.python .input}
#@tab all
#@save
def box_corner_to_center(boxes):
    """Konvertálás (bal felső, jobb alsó) formátumból (közép, szélesség, magasság) formátumba."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """Konvertálás (közép, szélesség, magasság) formátumból (bal felső, jobb alsó) formátumba."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

[**Definiáljuk a kutya és a macska befoglaló téglalapjait a képen**] a koordinátainformációk alapján.
A koordináták origója a kép bal felső sarka, jobbra és lefelé haladva az $x$ és $y$ tengelyek pozitív irányai.

```{.python .input}
#@tab all
# A `bbox` a bounding box (befoglaló téglalap) rövidítése
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
```

Kétszeri konverzióval ellenőrizhetjük a két befoglaló téglalap konverziós függvény helyességét.

```{.python .input}
#@tab all
boxes = d2l.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes
```

[**Rajzoljuk fel a befoglaló téglalapokat a képen**], hogy ellenőrizzük pontosságukat.
A rajzolás előtt definiálunk egy `bbox_to_rect` segédfüggvényt, amely a befoglaló téglalapot a `matplotlib` csomag befoglaló téglalap formátumában ábrázolja.

```{.python .input}
#@tab all
#@save
def bbox_to_rect(bbox, color):
    """Befoglaló téglalap konvertálása matplotlib formátumba."""
    # A befoglaló téglalap (bal felső x, bal felső y, jobb alsó x,
    # jobb alsó y) formátumának konvertálása matplotlib formátumba:
    # ((bal felső x, bal felső y), szélesség, magasság)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

A befoglaló téglalapok képre való felvitele után láthatjuk, hogy a két objektum fő körvonala alapvetően a két dobozba esik.

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```

## Összefoglalás

* Az objektumdetektálás nemcsak felismeri a képen lévő összes érdeklődési objektumot, hanem azok pozícióját is meghatározza. A pozíciót általában téglalap alakú befoglaló téglalappal ábrázolják.
* Két általánosan használt befoglaló téglalap ábrázolás között konvertálhatunk.

## Feladatok

1. Keress egy másik képet, és próbálj meg egy befoglaló téglalapot felcímkézni, amely tartalmazza az objektumot. Hasonlítsd össze a befoglaló téglalapok és a kategóriák felcímkézését: melyik szokott tovább tartani?
1. Miért mindig 4 a `box_corner_to_center` és `box_center_to_corner` bemeneti `boxes` argumentumának legbelső dimenziója?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/369)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1527)
:end_tab:
