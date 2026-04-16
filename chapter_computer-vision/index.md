# Számítógépes látás
:label:`chap_cv`

Legyen szó orvosi diagnózisról, önvezető járművekről, kamerafelügyeletről vagy intelligens szűrőkről – a számítógépes látás számos alkalmazása szorosan kapcsolódik jelenlegi és jövőbeli életünkhöz.
Az elmúlt években a deep learning forradalmasította a számítógépes látórendszerek teljesítményét.
Elmondható, hogy a legkorszerűbb számítógépes látási alkalmazások szinte elválaszthatatlanok a deep learningtől.
Erre való tekintettel ez a fejezet a számítógépes látás területére összpontosít, és megvizsgálja azokat a módszereket és alkalmazásokat, amelyek a közelmúltban meghatározó szerepet töltöttek be az akadémiai és az ipari kutatásban.

:numref:`chap_cnn` és :numref:`chap_modern_cnn` fejezetekben különféle konvolúciós neurális hálózatokat tanulmányoztunk, amelyeket a számítógépes látásban széles körben alkalmaznak, és egyszerű képosztályozási feladatokon is kipróbáltuk őket.
E fejezet elején két olyan módszert ismertetünk, amelyek javíthatják a modell általánosítóképességét: a *képaugmentációt* és a *finomhangolást*, és ezeket képosztályozásra is alkalmazzuk.
Mivel a mély neurális hálózatok képesek a képeket többszintű reprezentációkban hatékonyan leírni, ezeket a rétegenkénti reprezentációkat sikerrel alkalmazták különféle számítógépes látási feladatokban, mint az *objektumdetektálás*, a *szemantikai szegmentáció* és a *stílusátvitel*.
A számítógépes látásban alkalmazott rétegenkénti reprezentációk kulcsgondolatát követve először az objektumdetektálás fő összetevőivel és technikáival foglalkozunk. Ezután megmutatjuk, hogyan használhatók a *teljesen konvolúciós hálózatok* a képek szemantikai szegmentációjához. Majd ismertetjük, hogyan alkalmazhatók a stílusátviteli technikák képek előállítására – például a könyv borítójához hasonló képek generálásához.
A fejezet végén az itt és a korábbi fejezetekben tanultakat két népszerű számítógépes látási benchmark adathalmazon alkalmazzuk.

```toc
:maxdepth: 2

image-augmentation
fine-tuning
bounding-box
anchor
multiscale-object-detection
object-detection-dataset
ssd
rcnn
semantic-segmentation-and-dataset
transposed-conv
fcn
neural-style
kaggle-cifar10
kaggle-dog
```

