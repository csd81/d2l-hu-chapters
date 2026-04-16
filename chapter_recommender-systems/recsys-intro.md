# Az ajánlórendszerek áttekintése



Az elmúlt évtizedben az internet nagyszabású online szolgáltatások platformjává fejlődött, ami alapjaiban változtatta meg a kommunikáció, a hírolvasás, a termékek vásárlása és a filmek megtekintésének módját. Eközben az online elérhető elemek (*elemet* (item) használjuk filmek, hírek, könyvek és termékek jelölésére) páratlan száma szükségessé tesz egy olyan rendszert, amely segít felfedezni a számunkra preferált elemeket. Az ajánlórendszerek tehát hatékony információszűrő eszközök, amelyek személyre szabott szolgáltatásokat nyújthatnak és egyénre szabott élményt biztosíthatnak az egyes felhasználóknak. Röviden, az ajánlórendszerek kulcsszerepet játszanak a rendelkezésre álló hatalmas adatmennyiség felhasználásában, hogy a választásokat kezelhetővé tegyék. Napjainkban az ajánlórendszerek számos online szolgáltató – például az Amazon, a Netflix és a YouTube – alapját képezik. Gondoljunk csak az Amazon által ajánlott deep learning könyvek példájára a :numref:`subsec_recommender_systems` szakaszban. Az ajánlórendszerek alkalmazásának előnyei kétirányúak: egyrészt nagymértékben csökkenthetik a felhasználók erőfeszítését az elemek megtalálásában, és enyhíthetik az információtúlterhelés problémáját. Másrészt üzleti értéket adhatnak az online szolgáltatóknak, és fontos bevételi forrást jelentenek. Ez a fejezet bemutatja az ajánlórendszerek alapvető fogalmait, klasszikus modelljeit és a deep learning terén elért legújabb eredményeket, megvalósított példákkal együtt.

![Az ajánlási folyamat szemléltetése](../img/rec-intro.svg)


## Kollaboratív szűrés

Az ajánlórendszerek egyik legfontosabb fogalmával, a kollaboratív szűréssel (CF) kezdjük az utazást, amelyet először a Tapestry rendszer :cite:`Goldberg.Nichols.Oki.ea.1992` alkotott meg, arra utalva, hogy „az emberek együttműködnek, hogy segítsék egymást a szűrési folyamatban a levelezőlistákra és hírolvasó csoportokba érkező nagy mennyiségű e-mail és üzenet kezelése érdekében". Ez a kifejezés azóta gazdagabb tartalommal telítődött. Tágabb értelemben véve olyan folyamat, amely több felhasználó, ügynök és adatforrás közötti együttműködést igénylő technikákkal szűri az információkat vagy mintákat. A CF-nek számos formája és megjelenése óta rengeteg CF-módszer született.

Általánosságban a CF-technikák három kategóriába sorolhatók: memória-alapú CF, modell-alapú CF, és ezek kombinációja :cite:`Su.Khoshgoftaar.2009`. A memória-alapú CF reprezentatív technikái a legközelebbi szomszéd alapú CF-ek, mint például a felhasználóalapú CF és az elemalapú CF :cite:`Sarwar.Karypis.Konstan.ea.2001`. A látens faktoros modellek, például a mátrixfaktorizáció, modell-alapú CF-ek példái. A memória-alapú CF-nek korlátai vannak a ritka és nagy méretarányú adatok kezelésében, mivel a hasonlósági értékeket a közös elemek alapján számítja. A modell-alapú módszerek egyre népszerűbbek, mivel jobban képesek kezelni a ritkaságot és a skálázhatóságot. Számos modell-alapú CF-megközelítés kiterjeszthető neurális hálózatokkal, ami rugalmasabb és skálázhatóbb modellekhez vezet a deep learning számítási gyorsításával :cite:`Zhang.Yao.Sun.ea.2019`. Általában a CF csak a felhasználó-elem interakciós adatokat használja az előrejelzések és ajánlások készítéséhez. A CF mellett a tartalomalapú és kontextusalapú ajánlórendszerek is hasznosak az elemek/felhasználók tartalom-leírásainak és kontextuális jelek – például időbélyegek és helyek – beépítésében. Nyilvánvalóan szükség lehet a modell típusának/szerkezetének módosítására, amikor eltérő bemeneti adatok állnak rendelkezésre.



## Explicit és implicit visszajelzés

A felhasználók preferenciáinak megismeréséhez a rendszernek visszajelzést kell gyűjtenie tőlük. A visszajelzés lehet explicit vagy implicit :cite:`Hu.Koren.Volinsky.2008`. Például az [IMDb](https://www.imdb.com/) egy-tíz csillag közötti értékeléseket gyűjt filmekhez. A YouTube „tetszik" és „nem tetszik" gombokat kínál a felhasználóknak preferenciáik kifejezésére. Nyilvánvaló, hogy az explicit visszajelzés gyűjtéséhez a felhasználóknak proaktívan kell jelezniük érdeklődésüket. Mindazonáltal az explicit visszajelzés nem mindig könnyen elérhető, mivel sok felhasználó vonakodhat a termékek értékelésétől. Viszonylag könnyebben elérhető az implicit visszajelzés, mivel az főként az implicit viselkedés – például a felhasználói kattintások – modellezésével foglalkozik. Ezért sok ajánlórendszer az implicit visszajelzésre összpontosít, amely közvetve tükrözi a felhasználó véleményét a felhasználói viselkedés megfigyelésén keresztül. Az implicit visszajelzés sokféle formát ölthet, például vásárlási előzmények, böngészési előzmények, megtekintések, sőt egérmozdulatok is. Például egy felhasználó, aki sok könyvet vásárolt ugyanattól a szerzőtől, valószínűleg kedveli azt a szerzőt. Megjegyzendő, hogy az implicit visszajelzés eredendően zajos. Csak *sejthetjük* a preferenciáikat és valódi indítékaikat. Az, hogy egy felhasználó megnézett egy filmet, nem feltétlenül jelent pozitív véleményt arról a filmről.



## Ajánlási feladatok

Az elmúlt évtizedekben számos ajánlási feladatot vizsgáltak. Az alkalmazási terület alapján léteznek filmajánlások, hírügynöki ajánlások, érdekes helyek ajánlása :cite:`Ye.Yin.Lee.ea.2011` és így tovább. Lehetséges a feladatok megkülönböztetése a visszajelzés típusa és a bemeneti adatok alapján is; például az értékelés-előrejelzési feladat az explicit értékelések előrejelzését célozza meg. A top-$n$ ajánlás (elem rangsorolás) az összes elemet személyesen rangsorolja minden felhasználó számára az implicit visszajelzés alapján. Ha az időbélyeg-információ is rendelkezésre áll, szekvenciatudatos ajánlást is készíthetünk :cite:`Quadrana.Cremonesi.Jannach.2018`. Egy másik népszerű feladat a kattintási arány előrejelzése, amely szintén implicit visszajelzésen alapul, de különböző kategorikus jellemzők is felhasználhatók. Az új felhasználók számára és a meglévő felhasználók számára történő új elemek ajánlása hidegindítású ajánlásnak (cold-start recommendation) nevezzük :cite:`Schein.Popescul.Ungar.ea.2002`.



## Összefoglalás

* Az ajánlórendszerek fontosak az egyéni felhasználók és az ipar számára. A kollaboratív szűrés az ajánlás egyik kulcsfogalma.
* Kétféle visszajelzés létezik: implicit és explicit visszajelzés. Az elmúlt évtizedben számos ajánlási feladatot tártak fel.

## Gyakorlatok

1. Meg tudod magyarázni, hogyan befolyásolják az ajánlórendszerek a mindennapi életedet?
2. Milyen érdekes ajánlási feladatokat gondolsz, hogy érdemes lenne megvizsgálni?

[Megbeszélések](https://discuss.d2l.ai/t/398)
