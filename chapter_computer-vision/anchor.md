# Horgonydobozok
:label:`sec_anchor`


Az objektumfelismerési algoritmusok általában
nagyszámú régiót mintavételeznek a bemeneti képből, meghatározzák, hogy ezek a régiók tartalmaznak-e érdeklődési objektumokat, és a régiók határait úgy igazítják, hogy pontosabban megjósolják az objektumok *valós befoglaló téglalapjait*.
A különböző modellek eltérő régió-mintavételezési sémákat alkalmazhatnak.
Itt az egyik ilyen módszert mutatjuk be:
ez minden pixelt középpontként véve változó méretű és képarányú befoglaló téglalapokat generál.
Ezeket a befoglaló téglalapokat *horgonydobozoknak* nevezzük.
A :numref:`sec_ssd` fejezetben horgonydobozokon alapuló objektumfelismerési modellt tervezünk.

Először módosítsuk a nyomtatási pontosságot a tömörebb kimenet érdekében.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx

np.set_printoptions(2)  # A nyomtatási pontosság egyszerűsítése
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

torch.set_printoptions(2)  # A nyomtatási pontosság egyszerűsítése
```

## Több horgonydoboz generálása

Tegyük fel, hogy a bemeneti kép magassága $h$ és szélessége $w$.
A kép minden pixelét középpontként véve különböző alakú horgonydobozokat generálunk.
Legyen a *méret* $s\in (0, 1]$ és a *képarány* (szélesség és magasság aránya) $r > 0$.
Ekkor **a horgonydoboz szélessége és magassága rendre $ws\sqrt{r}$ és $hs/\sqrt{r}$.**
Megjegyezzük, hogy ha a középpont adott, akkor az ismert szélességű és magasságú horgonydoboz meghatározott.

Több különböző alakú horgonydoboz generálásához állítsunk be méretsorozatot:
$s_1,\ldots, s_n$ és
képaránysorozatot: $r_1,\ldots, r_m$.
Ha minden pixelt középpontként használjuk ezen méretek és képarányok összes kombinációjával,
a bemeneti képen összesen $whnm$ horgonydoboz keletkezik. Bár ezek a horgonydobozok lefedhetnék az összes valódi befoglaló téglalapot, a számítási komplexitás könnyen túl nagy lenne.
A gyakorlatban **csak azokat a kombinációkat vesszük figyelembe, amelyek** tartalmazzák $s_1$-et vagy $r_1$-et:

$$(s_1, r_1), (s_1, r_2), \ldots, (s_1, r_m), (s_2, r_1), (s_3, r_1), \ldots, (s_n, r_1).$$

Azaz az ugyanazon pixelt középpontként vevő horgonydobozok száma $n+m-1$. A teljes bemeneti képre összesen $wh(n+m-1)$ horgonydobozt generálunk.

A horgonydobozok fenti generálási módszerét a következő `multibox_prior` függvény valósítja meg. Megadjuk a bemeneti képet, a méretek listáját és a képarányok listáját, majd ez a függvény visszaadja az összes horgonydobozt.

```{.python .input}
#@tab mxnet
#@save
def multibox_prior(data, sizes, ratios):
    """Különböző alakú, minden pixel közepére helyezett horgonydobozok generálása."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.ctx, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, ctx=device)
    ratio_tensor = d2l.tensor(ratios, ctx=device)
    # Eltolások szükségesek a horgony pixel közepére helyezéséhez. Mivel
    # egy pixel magassága=1 és szélessége=1, a középpontokat 0.5-tel toljuk el
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Skálázott lépések az y tengely mentén
    steps_w = 1.0 / in_width  # Skálázott lépések az x tengely mentén

    # Az összes horgonydoboz-középpont generálása
    center_h = (d2l.arange(in_height, ctx=device) + offset_h) * steps_h
    center_w = (d2l.arange(in_width, ctx=device) + offset_w) * steps_w
    shift_x, shift_y = d2l.meshgrid(center_w, center_h)
    shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)

    # 'boxes_per_pixel' számú magasság és szélesség generálása, amelyeket később
    # a horgonydobozok sarokkoordinátáinak (xmin, xmax, ymin, ymax) létrehozásához használunk
    w = np.concatenate((size_tensor * np.sqrt(ratio_tensor[0]),
                        sizes[0] * np.sqrt(ratio_tensor[1:]))) \
                        * in_height / in_width  # Téglalap alakú bemenetek kezelése
    h = np.concatenate((size_tensor / np.sqrt(ratio_tensor[0]),
                        sizes[0] / np.sqrt(ratio_tensor[1:])))
    # Felezés a fél magasság és fél szélesség megkapásához
    anchor_manipulations = np.tile(np.stack((-w, -h, w, h)).T,
                                   (in_height * in_width, 1)) / 2

    # Minden középponthoz 'boxes_per_pixel' számú horgonydoboz tartozik, ezért
    # az összes horgonydoboz-középpont rácsát 'boxes_per_pixel' ismétléssel generáljuk
    out_grid = d2l.stack([shift_x, shift_y, shift_x, shift_y],
                         axis=1).repeat(boxes_per_pixel, axis=0)
    output = out_grid + anchor_manipulations
    return np.expand_dims(output, axis=0)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_prior(data, sizes, ratios):
    """Különböző alakú, minden pixel közepére helyezett horgonydobozok generálása."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, device=device)
    ratio_tensor = d2l.tensor(ratios, device=device)
    # Eltolások szükségesek a horgony pixel közepére helyezéséhez. Mivel
    # egy pixel magassága=1 és szélessége=1, a középpontokat 0.5-tel toljuk el
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Skálázott lépések az y tengely mentén
    steps_w = 1.0 / in_width  # Skálázott lépések az x tengely mentén

    # Az összes horgonydoboz-középpont generálása
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 'boxes_per_pixel' számú magasság és szélesség generálása, amelyeket később
    # a horgonydobozok sarokkoordinátáinak (xmin, xmax, ymin, ymax) létrehozásához használunk
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # Téglalap alakú bemenetek kezelése
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # Felezés a fél magasság és fél szélesség megkapásához
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # Minden középponthoz 'boxes_per_pixel' számú horgonydoboz tartozik, ezért
    # az összes horgonydoboz-középpont rácsát 'boxes_per_pixel' ismétléssel generáljuk
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)
```

Láthatjuk, hogy **a visszaadott `Y` horgonydoboz változó alakja** (batch méret, horgonydobozok száma, 4).

```{.python .input}
#@tab mxnet
img = image.imread('../img/catdog.jpg').asnumpy()
h, w = img.shape[:2]

print(h, w)
X = np.random.uniform(size=(1, 3, h, w))  # Bemeneti adatok létrehozása
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

```{.python .input}
#@tab pytorch
img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]

print(h, w)
X = torch.rand(size=(1, 3, h, w))  # Bemeneti adatok létrehozása
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

A `Y` horgonydoboz változó alakjának (képmagasság, képszélesség, az ugyanazon pixelt középpontként vevő horgonydobozok száma, 4) értékre módosítása után
megkaphatjuk az összes horgonydobozt, amelyek egy adott pixelpozíciót vesznek középpontként.
A következőkben **elérjük a (250, 250) középpontú első horgonydobozt**. Négy elemből áll: a horgonydoboz bal felső sarkának $(x, y)$-tengelykoordinátái és a jobb alsó sarok $(x, y)$-tengelykoordinátái.
Mindkét tengely koordinátaértékeit elosztják a kép szélességével és magasságával.

```{.python .input}
#@tab all
boxes = Y.reshape(h, w, 5, 4)
boxes[250, 250, 0, :]
```

Ahhoz, hogy **megmutatjuk az összes horgonydobozt, amelyek egy pixelt vesznek középpontként a képen**,
definiáljuk a következő `show_bboxes` függvényt, amely több befoglaló téglalapot rajzol a képre.

```{.python .input}
#@tab all
#@save
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes."""

    def make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = make_list(labels)
    colors = make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(d2l.numpy(bbox), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
```

Amint láttuk, a `boxes` változóban az $x$ és $y$ tengelyek koordinátaértékeit elosztják a kép szélességével és magasságával.
A horgonydobozok rajzolásakor vissza kell állítani az eredeti koordinátaértékeket;
ezért definiálunk egy `bbox_scale` változót az alábbiakban.
Most megrajzolhatjuk az összes (250, 250) középpontú horgonydobozt a képen.
Ahogy látható, a 0.75-ös méretű és 1-es képarányú kék horgonydoboz jól körülveszi a képen lévő kutyát.

```{.python .input}
#@tab all
d2l.set_figsize()
bbox_scale = d2l.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
```

## **Metszet az unióval (IoU)**

Éppen azt mondtuk, hogy egy horgonydoboz „jól" körülveszi a képen lévő kutyát.
Ha ismert az objektum valódi befoglaló téglalapja, hogyan lehet ezt a „jóságot" számszerűsíteni?
Intuitívan megmérhetjük a horgonydoboz és a valódi befoglaló téglalap hasonlóságát.
Tudjuk, hogy a *Jaccard-index* mérheti két halmaz hasonlóságát. Adott $\mathcal{A}$ és $\mathcal{B}$ halmazok esetén Jaccard-indexük a metszetük méretének és uniójuk méretének hányadosa:

$$J(\mathcal{A},\mathcal{B}) = \frac{\left|\mathcal{A} \cap \mathcal{B}\right|}{\left| \mathcal{A} \cup \mathcal{B}\right|}.$$


Valójában bármely befoglaló téglalap pixelterületét pixelhalmaznak tekinthetjük.
Ily módon megmérhetjük a két befoglaló téglalap hasonlóságát pixelhalmazaik Jaccard-indexével. Két befoglaló téglalap esetén Jaccard-indexüket általában *metszet az unióval* (*IoU*) névvel illetjük, ami a metszeti terület és az uniós terület arányát jelenti, ahogy a :numref:`fig_iou` ábrán látható.
Az IoU értékkészlete 0 és 1 közé esik:
0 azt jelenti, hogy a két befoglaló téglalap egyáltalán nem fed át,
1 pedig azt jelenti, hogy a két befoglaló téglalap egyenlő.

![Az IoU a két befoglaló téglalap metszeti területének és uniós területének aránya.](../img/iou.svg)
:label:`fig_iou`

E fejezet hátralévő részében az IoU-t fogjuk használni a horgonydobozok és a valódi befoglaló téglalapok, valamint a különböző horgonydobozok hasonlóságának mérésére.
Adott két horgony- vagy befoglaló téglalap listából az alábbi `box_iou` kiszámítja páronkénti IoU-jukat e két lista között.

```{.python .input}
#@tab mxnet
#@save
def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Alakja: `boxes1`, `boxes2`, `areas1`, `areas2`: (db boxes1, 4),
    # (db boxes2, 4), (db boxes1,), (db boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Alakja: `inter_upperlefts`, `inter_lowerrights`, `inters`: (db
    # boxes1, db boxes2, 2)
    inter_upperlefts = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clip(min=0)
    # Alakja: `inter_areas` és `union_areas`: (db boxes1, db boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

```{.python .input}
#@tab pytorch
#@save
def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Alakja: `boxes1`, `boxes2`, `areas1`, `areas2`: (db boxes1, 4),
    # (db boxes2, 4), (db boxes1,), (db boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Alakja: `inter_upperlefts`, `inter_lowerrights`, `inters`: (db
    # boxes1, db boxes2, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # Alakja: `inter_areas` és `union_areas`: (db boxes1, db boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

## Horgonydobozok felcímkézése tanítóadatokban
:label:`subsec_labeling-anchor-boxes`


A tanítóadathalmazban minden horgonydobozt tanítópéldának tekintünk.
Az objektumfelismerési modell tanításához minden horgonydobozhoz *osztály*- és *eltolás*-címkékre van szükségünk,
ahol az előbbi a horgonydobozhoz kapcsolódó objektum osztálya,
az utóbbi pedig a valódi befoglaló téglalapnak a horgonydobozhoz viszonyított eltolása.
A jóslás során
minden képhez több horgonydobozt generálunk,
megjósoljuk az összes horgonydoboz osztályait és eltolásait,
pozícióikat a jósolt eltolások szerint igazítjuk a jósolt befoglaló téglalapok megkapásához,
és végül csak azokat a jósolt befoglaló téglalapokat adjuk ki, amelyek bizonyos feltételeknek megfelelnek.


Ahogy tudjuk, egy objektumfelismerési tanítóhalmaz a *valódi befoglaló téglalapok* helyzetéhez és a körülöttük lévő objektumok osztályaihoz tartozó címkéket tartalmaz.
Bármely generált *horgonydoboz* felcímkézéséhez a hozzá legközelebb eső *hozzárendelt* valódi befoglaló téglalap felcímkézett helyzetére és osztályára hivatkozunk.
A következőkben leírunk egy algoritmust a legközelebbi valódi befoglaló téglalapok hozzárendeléséhez a horgonydobozokhoz.

### **Valódi befoglaló téglalapok hozzárendelése a horgonydobozokhoz**

Adott egy kép,
tegyük fel, hogy a horgonydobozok $A_1, A_2, \ldots, A_{n_a}$ és a valódi befoglaló téglalapok $B_1, B_2, \ldots, B_{n_b}$, ahol $n_a \geq n_b$.
Definiáljuk az $\mathbf{X} \in \mathbb{R}^{n_a \times n_b}$ mátrixot, amelynek $i$-edik sorában és $j$-edik oszlopában lévő $x_{ij}$ eleme az $A_i$ horgonydoboz és a $B_j$ valódi befoglaló téglalap IoU-ja. Az algoritmus a következő lépésekből áll:

1. Keressük meg az $\mathbf{X}$ mátrix legnagyobb elemét, és jelöljük sor- és oszlopindexét rendre $i_1$-nek és $j_1$-nek. Ekkor a $B_{j_1}$ valódi befoglaló téglalapot hozzárendeljük az $A_{i_1}$ horgonydobozhoz. Ez intuitív, mivel $A_{i_1}$ és $B_{j_1}$ a legközelebb vannak egymáshoz az összes horgonydoboz–valódi befoglaló téglalap pár közül. Az első hozzárendelés után töröljük az $\mathbf{X}$ mátrix ${i_1}$-edik sorának és ${j_1}$-edik oszlopának összes elemét.
1. Keressük meg az $\mathbf{X}$ mátrix maradék elemeinek legnagyobb értékét, és jelöljük sor- és oszlopindexét rendre $i_2$-nek és $j_2$-nek. Hozzárendeljük a $B_{j_2}$ valódi befoglaló téglalapot az $A_{i_2}$ horgonydobozhoz, és töröljük az $\mathbf{X}$ mátrix ${i_2}$-edik sorának és ${j_2}$-edik oszlopának összes elemét.
1. Ezen a ponton két sor és két oszlop elemei törlődtek az $\mathbf{X}$ mátrixból. Folytatjuk, amíg az $\mathbf{X}$ mátrix $n_b$ oszlopának összes eleme törlődik. Ekkorra minden $n_b$ horgonydobozhoz valódi befoglaló téglalapot rendeltünk.
1. Csak a maradék $n_a - n_b$ horgonydobozon haladunk át. Például adott bármely $A_i$ horgonydoboz esetén keressük meg azt a $B_j$ valódi befoglaló téglalapot, amelynek IoU-ja $A_i$-vel a legnagyobb az $\mathbf{X}$ mátrix $i$-edik sorában, és csak akkor rendeljük $B_j$-t $A_i$-hez, ha ez az IoU nagyobb egy előre meghatározott küszöbnél.

Szemléltessük a fenti algoritmust egy konkrét példán.
Ahogy a :numref:`fig_anchor_label` (bal) ábrán látható, feltéve, hogy az $\mathbf{X}$ mátrix maximális értéke $x_{23}$, hozzárendeljük a $B_3$ valódi befoglaló téglalapot az $A_2$ horgonydobozhoz.
Ezután töröljük a mátrix 2. sorának és 3. oszlopának összes elemét, megkeressük a maradék elemek (árnyékolt terület) legnagyobb $x_{71}$ értékét, és hozzárendeljük a $B_1$ valódi befoglaló téglalapot az $A_7$ horgonydobozhoz.
Következőként, ahogy a :numref:`fig_anchor_label` (közép) ábrán látható, töröljük a mátrix 7. sorának és 1. oszlopának összes elemét, megkeressük a maradék elemek (árnyékolt terület) legnagyobb $x_{54}$ értékét, és hozzárendeljük a $B_4$ valódi befoglaló téglalapot az $A_5$ horgonydobozhoz.
Végül, ahogy a :numref:`fig_anchor_label` (jobb) ábrán látható, töröljük a mátrix 5. sorának és 4. oszlopának összes elemét, megkeressük a maradék elemek (árnyékolt terület) legnagyobb $x_{92}$ értékét, és hozzárendeljük a $B_2$ valódi befoglaló téglalapot az $A_9$ horgonydobozhoz.
Ezután csak a maradék $A_1, A_3, A_4, A_6, A_8$ horgonydobozokon kell áthaladni, és a küszöbérték alapján eldönteni, hogy hozzárendelünk-e hozzájuk valódi befoglaló téglalapot.

![Valódi befoglaló téglalapok hozzárendelése a horgonydobozokhoz.](../img/anchor-label.svg)
:label:`fig_anchor_label`

Ezt az algoritmust valósítja meg a következő `assign_anchor_to_bbox` függvény.

```{.python .input}
#@tab mxnet
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Az i-edik sor j-edik oszlopának x_ij eleme az i-edik horgonydoboz
    # és a j-edik valódi befoglaló téglalap IoU-ja
    jaccard = box_iou(anchors, ground_truth)
    # A tenzor inicializálása, amely minden horgonyhoz a hozzárendelt
    # valódi befoglaló téglalapot tárolja
    anchors_bbox_map = np.full((num_anchors,), -1, dtype=np.int32, ctx=device)
    # Valódi befoglaló téglalapok hozzárendelése a küszöbérték szerint
    max_ious, indices = np.max(jaccard, axis=1), np.argmax(jaccard, axis=1)
    anc_i = np.nonzero(max_ious >= iou_threshold)[0]
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = np.full((num_anchors,), -1)
    row_discard = np.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = np.argmax(jaccard)  # A legnagyobb IoU megkeresése
        box_idx = (max_idx % num_gt_boxes).astype('int32')
        anc_idx = (max_idx / num_gt_boxes).astype('int32')
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

```{.python .input}
#@tab pytorch
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Az i-edik sor j-edik oszlopának x_ij eleme az i-edik horgonydoboz
    # és a j-edik valódi befoglaló téglalap IoU-ja
    jaccard = box_iou(anchors, ground_truth)
    # A tenzor inicializálása, amely minden horgonyhoz a hozzárendelt
    # valódi befoglaló téglalapot tárolja
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # Valódi befoglaló téglalapok hozzárendelése a küszöbérték szerint
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # A legnagyobb IoU megkeresése
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

### Osztályok és eltolások felcímkézése

Most felcímkézhetjük minden horgonydoboz osztályát és eltolását. Tegyük fel, hogy egy $A$ horgonydobozhoz hozzá van rendelve egy $B$ valódi befoglaló téglalap.
Egyrészt az $A$ horgonydoboz osztályát $B$ osztályaként jelölik.
Másrészt az $A$ horgonydoboz eltolását $B$ és $A$ középső koordinátái közötti relatív pozíció, valamint e két doboz relatív mérete alapján jelölik.
Tekintettel az adathalmazban lévő különböző dobozok eltérő pozícióira és méreteire, transformációkat alkalmazhatunk ezekre a relatív pozíciókra és méretekre, amelyek egyenletesebben elosztott eltolásokat eredményezhetnek, amelyek könnyebben illeszthetők.
Itt leírunk egy általánosan alkalmazott transzformációt.

Adott az $A$ és $B$ középső koordinátái $(x_a, y_a)$ és $(x_b, y_b)$,
szélességük $w_a$ és $w_b$,
és magasságuk $h_a$ és $h_b$, rendre.
Felcímkézhetjük $A$ eltolását a következőképpen:

$$\left( \frac{ \frac{x_b - x_a}{w_a} - \mu_x }{\sigma_x},
\frac{ \frac{y_b - y_a}{h_a} - \mu_y }{\sigma_y},
\frac{ \log \frac{w_b}{w_a} - \mu_w }{\sigma_w},
\frac{ \log \frac{h_b}{h_a} - \mu_h }{\sigma_h}\right),$$


ahol az állandók alapértelmezett értékei: $\mu_x = \mu_y = \mu_w = \mu_h = 0, \sigma_x=\sigma_y=0.1$ és $\sigma_w=\sigma_h=0.2$.
Ezt a transzformációt valósítja meg az alábbi `offset_boxes` függvény.

```{.python .input}
#@tab all
#@save
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets."""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * d2l.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = d2l.concat([offset_xy, offset_wh], axis=1)
    return offset
```

Ha egy horgonydobozhoz nincs valódi befoglaló téglalap hozzárendelve, egyszerűen „háttér" osztályként jelöljük a horgonydoboz osztályát.
A háttér osztályú horgonydobozokat gyakran *negatív* horgonydobozoknak nevezik, a többit *pozitív* horgonydobozoknak.
A következő `multibox_target` függvényt valósítjuk meg, hogy **felcímkézzük a horgonydobozok osztályait és eltolásait** (az `anchors` argumentum) a valódi befoglaló téglalapok alapján (a `labels` argumentum).
Ez a függvény a háttér osztályt nullára állítja, és egy új osztály egész indexét eggyel növeli.

```{.python .input}
#@tab mxnet
#@save
def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.ctx, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = np.tile((np.expand_dims((anchors_bbox_map >= 0),
                                            axis=-1)), (1, 4)).astype('int32')
        # Osztálycímkék és hozzárendelt befoglaló téglalap koordináták
        # inicializálása nullákkal
        class_labels = d2l.zeros(num_anchors, dtype=np.int32, ctx=device)
        assigned_bb = d2l.zeros((num_anchors, 4), dtype=np.float32,
                                ctx=device)
        # A horgonydobozok osztálycímkézése a hozzárendelt valódi befoglaló
        # téglalapok alapján. Ha egy horgonydobozhoz nincs hozzárendelve semmi,
        # az osztálya háttér marad (az érték nulla marad)
        indices_true = np.nonzero(anchors_bbox_map >= 0)[0]
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].astype('int32') + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Eltolás-transzformáció
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = d2l.stack(batch_offset)
    bbox_mask = d2l.stack(batch_mask)
    class_labels = d2l.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # Osztálycímkék és hozzárendelt befoglaló téglalap koordináták
        # inicializálása nullákkal
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # A horgonydobozok osztálycímkézése a hozzárendelt valódi befoglaló
        # téglalapok alapján. Ha egy horgonydobozhoz nincs hozzárendelve semmi,
        # az osztálya háttér marad (az érték nulla marad)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Eltolás-transzformáció
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

### Egy példa

Szemléltessük a horgonydoboz-felcímkézést egy konkrét példán.
Megadjuk a betöltött képen lévő kutya és macska valódi befoglaló téglalapjait,
ahol az első elem az osztály (0 a kutyának, 1 a macskának), a maradék négy elem pedig a bal felső sarok és a jobb alsó sarok $(x, y)$-tengelykoordinátái (értékkészlet 0 és 1 között).
Öt felcímkézendő horgonydobozt is felépítünk a bal felső sarok és a jobb alsó sarok koordinátái alapján:
$A_0, \ldots, A_4$ (az index 0-tól kezdődik).
Ezután **megrajzoljuk ezeket a valódi befoglaló téglalapokat és horgonydobozokat a képen.**

```{.python .input}
#@tab all
ground_truth = d2l.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = d2l.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);
```

A fent definiált `multibox_target` függvény segítségével
**felcímkézhetjük ezeknek a horgonydobozoknak az osztályait és eltolásait
a valódi befoglaló téglalapok alapján** a kutya és a macska esetén.
Ebben a példában a háttér, a kutya és a macska osztályok indexe rendre 0, 1 és 2.
Az alábbiakban hozzáadunk egy dimenziót a horgonydobozok és a valódi befoglaló téglalapok példányaihoz.

```{.python .input}
#@tab mxnet
labels = multibox_target(np.expand_dims(anchors, axis=0),
                         np.expand_dims(ground_truth, axis=0))
```

```{.python .input}
#@tab pytorch
labels = multibox_target(anchors.unsqueeze(dim=0),
                         ground_truth.unsqueeze(dim=0))
```

A visszaadott eredményben három elem van, amelyek mindegyike tenzor formátumú.
A harmadik elem tartalmazza a bemeneti horgonydobozok felcímkézett osztályait.

Elemezzük az alábbi visszaadott osztálycímkéket a képen lévő horgonydoboz és valódi befoglaló téglalap pozíciók alapján.
Először az összes horgonydoboz–valódi befoglaló téglalap pár közül az $A_4$ horgonydoboz és a macska valódi befoglaló téglalapja IoU-ja a legnagyobb.
Így $A_4$ osztályát macskának jelöljük.
Az $A_4$-et vagy a macska valódi befoglaló téglalapját tartalmazó párokat kivéve, a maradékból az $A_1$ horgonydoboz és a kutya valódi befoglaló téglalap párja rendelkezik a legnagyobb IoU-val.
Tehát $A_1$ osztályát kutyának jelöljük.
Ezután végig kell haladni a maradék három felcímkézetlen horgonydobozon: $A_0$, $A_2$ és $A_3$.
Az $A_0$ esetén a legnagyobb IoU-jú valódi befoglaló téglalap osztálya a kutya, de az IoU az előre meghatározott küszöb (0.5) alatt van, ezért az osztályt háttérként jelöljük;
$A_2$ esetén a legnagyobb IoU-jú valódi befoglaló téglalap osztálya a macska, és az IoU meghaladja a küszöböt, ezért az osztályt macskának jelöljük;
$A_3$ esetén a legnagyobb IoU-jú valódi befoglaló téglalap osztálya a macska, de az érték a küszöb alatt van, ezért az osztályt háttérként jelöljük.

```{.python .input}
#@tab all
labels[2]
```

A második visszaadott elem egy (batch méret, a horgonydobozok számának négyszerese) alakú maszk változó.
A maszk változó minden négy eleme minden horgonydoboz négy eltolásértékének felel meg.
Mivel nem törődünk a háttér felismerésével, ennek a negatív osztálynak az eltolásai nem befolyásolhatják a célfüggvényt.
Az elemenkénti szorzásokon keresztül a maszk változó nullái kiszűrik a negatív osztály eltolásait a célfüggvény kiszámítása előtt.

```{.python .input}
#@tab all
labels[1]
```

Az első visszaadott elem tartalmazza a minden horgonydobozhoz felcímkézett négy eltolásértéket.
Megjegyezzük, hogy a negatív osztályú horgonydobozok eltolásait nullaként jelöljük.

```{.python .input}
#@tab all
labels[0]
```

## Befoglaló téglalapok jóslása nem-maximum szuppresszióval
:label:`subsec_predicting-bounding-boxes-nms`

A jóslás során
a képhez több horgonydobozt generálunk, és mindegyikükre megjósolunk osztályokat és eltolásokat.
A *jósolt befoglaló téglalapot* tehát a horgonydobozból és a jósolt eltolásából kapjuk meg.
Az alábbiakban megvalósítjuk az `offset_inverse` függvényt,
amely horgonydobozokat és eltolás-jóslásokat vesz bemenetként, és **inverz eltolástranszformációkat alkalmaz a jósolt befoglaló téglalap koordinátáinak visszaadásához**.

```{.python .input}
#@tab all
#@save
def offset_inverse(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets."""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = d2l.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = d2l.concat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox
```

Ha sok horgonydoboz van,
sok hasonló (jelentős átfedéssel bíró)
jósolt befoglaló téglalap adható ki potenciálisan
ugyanazon objektum köré.
A kimenet egyszerűsítéséhez
az ugyanahhoz az objektumhoz tartozó hasonló jósolt befoglaló téglalapokat összevonhatjuk
a *nem-maximum szuppresszió* (NMS) segítségével.

Így működik a nem-maximum szuppresszió.
Egy $B$ jósolt befoglaló téglalap esetén
az objektumfelismerési modell kiszámítja a jósolt valószínűséget minden osztályhoz.
Jelöljük $p$-vel a legnagyobb jósolt valószínűséget;
az ehhez a valószínűséghez tartozó osztály a $B$ jósolt osztálya.
Konkrétan, $p$-t a $B$ jósolt befoglaló téglalap *megbízhatóságának* (pontszámának) nevezzük.
Ugyanazon a képen
az összes jósolt nem-háttér befoglaló téglalapot megbízhatóság szerint csökkenő sorrendbe rendezik,
hogy előállítsák az $L$ listát.
Majd a rendezett $L$ listán a következő lépéseket hajtják végre:

1. Válasszuk ki az $L$-ből a legmagasabb megbízhatóságú $B_1$ jósolt befoglaló téglalapot alapként, és távolítsuk el az $L$-ből az összes nem-alapként megjelölt jósolt befoglaló téglalapot, amelynek IoU-ja $B_1$-gyel meghaladja az előre meghatározott $\epsilon$ küszöböt. Ezen a ponton $L$ megőrzi a legmagasabb megbízhatóságú jósolt befoglaló téglalapot, de elveti azokat, amelyek túl hasonlók hozzá. Röviden: azokat, amelyek *nem-maximum* megbízhatósági pontszámmal rendelkeznek, *szuppresszálják*.
1. Válasszuk ki az $L$-ből a második legmagasabb megbízhatóságú $B_2$ jósolt befoglaló téglalapot másik alapként, és távolítsuk el az $L$-ből az összes nem-alapként megjelölt jósolt befoglaló téglalapot, amelynek IoU-ja $B_2$-vel meghaladja $\epsilon$-t.
1. Ismételjük a fenti folyamatot, amíg az $L$ összes jósolt befoglaló téglalapját alapként nem használtuk. Ekkorra az $L$-beli bármely jósolt befoglaló téglalap pár IoU-ja kisebb az $\epsilon$ küszöbnél; tehát egyetlen pár sem hasonlít túlságosan a másikra.
1. Adjuk ki az $L$ lista összes jósolt befoglaló téglalapját.

**A következő `nms` függvény csökkenő sorrendbe rendezi a megbízhatósági pontszámokat, és visszaadja indexeiket.**

```{.python .input}
#@tab mxnet
#@save
def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    B = scores.argsort()[::-1]
    keep = []  # A megtartandó jósolt befoglaló téglalapok indexei
    while B.size > 0:
        i = B[0]
        keep.append(i)
        if B.size == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = np.nonzero(iou <= iou_threshold)[0]
        B = B[inds + 1]
    return np.array(keep, dtype=np.int32, ctx=boxes.ctx)
```

```{.python .input}
#@tab pytorch
#@save
def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # A megtartandó jósolt befoglaló téglalapok indexei
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return d2l.tensor(keep, device=boxes.device)
```

A következő `multibox_detection` függvényt definiáljuk, hogy **nem-maximum szuppressziót alkalmazzunk a jósolt befoglaló téglalapokra**.
Ne aggódjon, ha az implementáció kicsit bonyolultnak tűnik: rögtön az implementáció után egy konkrét példán megmutatjuk, hogyan működik.

```{.python .input}
#@tab mxnet
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression."""
    device, batch_size = cls_probs.ctx, cls_probs.shape[0]
    anchors = np.squeeze(anchors, axis=0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = np.max(cls_prob[1:], 0), np.argmax(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Az összes nem 'keep' index megkeresése és az osztály háttérre állítása
        all_idx = np.arange(num_anchors, dtype=np.int32, ctx=device)
        combined = d2l.concat((keep, all_idx))
        unique, counts = np.unique(combined, return_counts=True)
        non_keep = unique[counts == 1]
        all_id_sorted = d2l.concat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted].astype('float32')
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Itt a 'pos_threshold' a pozitív (nem háttér)
        # jóslatok küszöbértéke
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = d2l.concat((np.expand_dims(class_id, axis=1),
                                np.expand_dims(conf, axis=1),
                                predicted_bb), axis=1)
        out.append(pred_info)
    return d2l.stack(out)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression."""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Az összes nem 'keep' index megkeresése és az osztály háttérre állítása
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Itt a 'pos_threshold' a pozitív (nem háttér)
        # jóslatok küszöbértéke
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return d2l.stack(out)
```

Most **alkalmazzuk a fenti implementációkat egy négy horgonydobozból álló konkrét példán**.
Az egyszerűség kedvéért feltételezzük, hogy a jósolt eltolások mind nullák.
Ez azt jelenti, hogy a jósolt befoglaló téglalapok horgonydobozok.
Minden osztályhoz – háttér, kutya és macska – megadjuk a jósolt valószínűséget is.

```{.python .input}
#@tab all
anchors = d2l.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = d2l.tensor([0] * d2l.size(anchors))
cls_probs = d2l.tensor([[0] * 4,  # Jósolt háttér-valószínűség 
                      [0.9, 0.8, 0.7, 0.1],  # Jósolt kutya-valószínűség 
                      [0.1, 0.2, 0.3, 0.9]])  # Jósolt macska-valószínűség
```

**Megrajzolhatjuk ezeket a jósolt befoglaló téglalapokat megbízhatóságukkal együtt a képen.**

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
```

Most meghívhatjuk a `multibox_detection` függvényt a nem-maximum szuppresszió végrehajtásához, ahol a küszöb 0.5-re van állítva.
Megjegyezzük, hogy a tenzor bemenetben hozzáadunk egy dimenziót a példányokhoz.

Láthatjuk, hogy **a visszaadott eredmény alakja** (batch méret, horgonydobozok száma, 6).
A legbelső dimenzió hat eleme ugyanazon jósolt befoglaló téglalap kimeneti információját adja meg.
Az első elem a jósolt osztályindex, amely 0-tól kezdődik (0 kutya, 1 macska). A -1 értéke háttérre vagy nem-maximum szuppresszióban való eltávolításra utal.
A második elem a jósolt befoglaló téglalap megbízhatósága.
A maradék négy elem a jósolt befoglaló téglalap bal felső sarkának és jobb alsó sarkának $(x, y)$-tengelykoordinátái (értékkészlet 0 és 1 között).

```{.python .input}
#@tab mxnet
output = multibox_detection(np.expand_dims(cls_probs, axis=0),
                            np.expand_dims(offset_preds, axis=0),
                            np.expand_dims(anchors, axis=0),
                            nms_threshold=0.5)
output
```

```{.python .input}
#@tab pytorch
output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
output
```

A -1 osztályú jósolt befoglaló téglalapok eltávolítása után
**megjeleníthetjük a nem-maximum szuppresszió által megtartott végső jósolt befoglaló téglalapot**.

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
for i in d2l.numpy(output[0]):
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [d2l.tensor(i[2:]) * bbox_scale], label)
```

A gyakorlatban eltávolíthatjuk az alacsonyabb megbízhatóságú jósolt befoglaló téglalapokat még a nem-maximum szuppresszió végrehajtása előtt is, csökkentve ezzel az algoritmus számítási igényét.
A nem-maximum szuppresszió kimenetét utófeldolgozhatjuk is, például csak a magasabb megbízhatóságú eredmények megtartásával a végső kimenetben.


## Összefoglalás

* A kép minden pixelét középpontként véve különböző alakú horgonydobozokat generálunk.
* A metszet az unióval (IoU), más néven Jaccard-index, a két befoglaló téglalap hasonlóságát méri. A metszeti terület és az uniós terület aránya.
* A tanítóhalmazban minden horgonydobozhoz kétfajta címkére van szükség. Az egyik a horgonydobozhoz kapcsolódó objektum osztálya, a másik pedig a valódi befoglaló téglalapnak a horgonydobozhoz viszonyított eltolása.
* A jóslás során nem-maximum szuppressziót (NMS) alkalmazhatunk a hasonló jósolt befoglaló téglalapok eltávolítására, ezáltal egyszerűsítve a kimenetet.


## Feladatok

1. Módosítsd a `multibox_prior` függvényben a `sizes` és `ratios` értékeit. Miként változnak a generált horgonydobozok?
1. Építs fel és jeleníts meg két befoglaló téglalapot, amelyek IoU-ja 0.5. Hogyan fedik át egymást?
1. Módosítsd az `anchors` változót a :numref:`subsec_labeling-anchor-boxes` és :numref:`subsec_predicting-bounding-boxes-nms` fejezetekben. Hogyan változnak az eredmények?
1. A nem-maximum szuppresszió egy mohó algoritmus, amely a jósolt befoglaló téglalapokat *eltávolítással* szuppresszálja. Lehetséges, hogy az eltávolított befoglaló téglalapok némelyike valójában hasznos? Hogyan módosítható ez az algoritmus *puha* szuppresszióra? Tekintsd meg a Soft-NMS :cite:`Bodla.Singh.Chellappa.ea.2017` cikket.
1. Ahelyett, hogy kézzel terveznék, megtanulható-e a nem-maximum szuppresszió?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/370)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1603)
:end_tab:
