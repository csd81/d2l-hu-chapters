# Hozzájárulás a könyvhöz
:label:`sec_how_to_contribute`

Az [olvasók](https://github.com/d2l-ai/d2l-en/graphs/contributors) hozzájárulásai segítenek fejleszteni ezt a könyvet. Ha elírást, elavult hivatkozást találsz, vagy úgy véled, hogy kimaradt egy hivatkozás, a kód nem elég elegáns, vagy egy magyarázat nem egyértelmű, járulj hozzá, és segíts nekünk jobban segíteni az olvasóinknak. Míg a hagyományos könyveknél a kiadások közötti késés, és így a hibajavítások közötti idő, akár években is mérhető, ennél a könyvnél egy fejlesztés általában órák vagy napok alatt beépül. Mindez a verziókezelésnek és a folyamatos integrációs (CI) tesztelésnek köszönhető. Ehhez egy [pull requestet](https://github.com/d2l-ai/d2l-en/pulls) kell benyújtani a GitHub-tárolóba. Amikor a szerzők egyesítik a pull requestedet a kódtárral, közreműködővé válsz.

## Kisebb módosítások benyújtása

A leggyakoribb hozzájárulások egyetlen mondat szerkesztése vagy elírások javítása. Javasoljuk, hogy keressük meg a forrásfájlt a [GitHub-tárolóban](https://github.com/d2l-ai/d2l-en), és szerkesszük a fájlt közvetlenül. Például a [Find file](https://github.com/d2l-ai/d2l-en/find/master) gombbal (:numref:`fig_edit_file`) megkereshetjük a forrásfájlt (egy markdown fájlt). Ezután kattintsunk a jobb felső sarokban lévő "Edit this file" gombra, hogy elvégezhessük a módosításokat a markdown fájlban.

![Fájl szerkesztése a Githubon.](../img/edit-file.png)
:width:`300px`
:label:`fig_edit_file`

Ha elkészültünk, írjuk le a módosításainkat az oldal alján található "Propose file change" panelen, majd kattintsunk a "Propose file change" gombra. Ez átirányít egy új oldalra, ahol áttekinthetjük a módosításainkat (:numref:`fig_git_createpr`). Ha minden rendben van, a "Create pull request" gombra kattintva nyújthatunk be pull requestet.

## Nagyobb módosítások javaslata

Ha egy nagyobb szöveg- vagy kódrész frissítését tervezed, akkor érdemes kicsit többet tudnod a könyv által használt formátumról. A forrásfájl a [markdown formátumon](https://daringfireball.net/projects/markdown/syntax) alapul, kiegészítve a [D2L-Book](http://book.d2l.ai/user/markdown.html) csomag bővítményeivel, mint például az egyenletek, képek, fejezetek és hivatkozások kezelése. Bármely markdown szerkesztővel megnyithatod ezeket a fájlokat, és elvégezheted a módosításaidat.

Ha a kódot szeretnéd módosítani, javasoljuk, hogy a Jupyter Notebook segítségével nyisd meg ezeket a markdown fájlokat, ahogyan az a :numref:`sec_jupyter` szakaszban le van írva, hogy futtatni és tesztelni tudd a módosításaidat. Ne felejtsd el törölni az összes kimenetet a módosítások benyújtása előtt, mivel a CI rendszer végrehajtja a frissített szakaszokat a kimenetek előállításához.

Egyes szakaszok több keretrendszer-implementációt is támogathatnak.
Ha új kódblokkot adsz hozzá, jelöld a blokkot a `%%tab` jelöléssel a kezdő sorban. Például:
`%%tab pytorch` egy PyTorch kódblokkhoz, `%%tab tensorflow` egy TensorFlow kódblokkhoz, vagy `%%tab all` egy, az összes implementáció számára közös kódblokkhoz. További információkért nézd meg a `d2lbook` csomagot.

## Nagyobb módosítások benyújtása

Javasoljuk, hogy nagyobb módosítások benyújtásához a szabványos Git folyamatot alkalmazzuk. Röviden összefoglalva, a folyamat a :numref:`fig_contribute` ábrán leírtak szerint működik.

![Hozzájárulás a könyvhöz.](../img/contribute.svg)
:label:`fig_contribute`

Részletesen végigvezetünk a lépéseken. Ha már ismered a Gitet, kihagyhatod ezt a szakaszt. A szemléltetés kedvéért feltételezzük, hogy a közreműködő felhasználóneve "astonzhang".

### A Git telepítése

A Git nyílt forráskódú könyv leírja, [hogyan kell telepíteni a Gitet](https://git-scm.com/book/en/v2). Ez általában Ubuntu Linux rendszeren az `apt install git` paranccsal, macOS-en az Xcode fejlesztői eszközök telepítésével, vagy a GitHub [asztali kliensének](https://desktop.github.com) használatával lehetséges. Ha nincs GitHub-fiókunk, regisztrálnunk kell egyet.

### Bejelentkezés a GitHubra

Írjuk be a böngészőbe a könyv kódtárának [címét](https://github.com/d2l-ai/d2l-en/). Kattintsunk a `Fork` gombra a :numref:`fig_git_fork` ábra jobb felső sarkában lévő piros keretben, hogy készítsünk egy másolatot a könyv tárolójából. Ez most már *a mi másolatunk*, és tetszés szerint módosíthatjuk.

![A kódtár oldala.](../img/git-fork.png)
:width:`700px`
:label:`fig_git_fork`


A könyv kódtára most már el lesz ágaztatva (azaz másolva) a mi felhasználónevünkre, például `astonzhang/d2l-en` formában, ahogyan az a :numref:`fig_git_forked` ábra bal felső sarkában látható.

![Az elágaztatott kódtár.](../img/git-forked.png)
:width:`700px`
:label:`fig_git_forked`

### A tároló klónozása

A tároló klónozásához (azaz helyi másolat készítéséhez) szükségünk van a tároló címére. A :numref:`fig_git_clone` ábrán látható zöld gomb jeleníti meg ezt. Ha azt tervezzük, hogy hosszabb ideig megőrizzük ezt az elágaztatást, győződjünk meg arról, hogy a helyi másolatunk naprakész a fő tárolóval. Egyelőre egyszerűen kövessük a :ref:`chap_installation` utasításait a kezdéshez. A fő különbség az, hogy most *a saját elágaztatásunkat* töltjük le.

![A tároló klónozása.](../img/git-clone.png)
:width:`700px`
:label:`fig_git_clone`

```
# Cseréld le a your_github_username részt a GitHub-felhasználónevedre
git clone https://github.com/your_github_username/d2l-en.git
```


### Szerkesztés és feltöltés

Most itt az ideje szerkeszteni a könyvet. A legjobb, ha a :numref:`sec_jupyter` útmutatása szerint a Jupyter Notebookban szerkesztjük. Végezzük el a módosításokat és ellenőrizzük, hogy rendben vannak-e. Tegyük fel, hogy kijavítottunk egy elírást a `~/d2l-en/chapter_appendix-tools-for-deep-learning/contributing.md` fájlban.
Ezután ellenőrizhetjük, hogy mely fájlokat módosítottuk.

Ezen a ponton a Git jelzi, hogy a `chapter_appendix-tools-for-deep-learning/contributing.md` fájl módosult.

```
mylaptop:d2l-en me$ git status
Az ág neve: master
Az ág naprakész az 'origin/master' ággal.

Még nem előkészített módosítások commitoláshoz:
  (a "git add <file>..." paranccsal adhatod hozzá őket a commit tartalmához)
  (a "git checkout -- <file>..." paranccsal eldobhatod a munkakönyvtárban lévő módosításokat)

	modified:   chapter_appendix-tools-for-deep-learning/contributing.md
```


Miután meggyőződtünk arról, hogy ez az, amit szeretnénk, hajtsuk végre a következő parancsot:

```
git add chapter_appendix-tools-for-deep-learning/contributing.md
git commit -m 'Fix a typo in git documentation'
git push
```


A módosított kód ezután a tároló személyes elágaztatásában lesz. A módosítás hozzáadásának kéréséhez pull requestet kell létrehoznunk a könyv hivatalos tárolójához.

### Pull requestek benyújtása

Ahogy a :numref:`fig_git_newpr` ábra mutatja, menjünk a tároló saját elágaztatásához a GitHubon, és válasszuk a "New pull request" lehetőséget. Ez megnyit egy képernyőt, amely megmutatja a módosításaink és a könyv fő tárolójában jelenleg lévő tartalom közötti különbségeket.

![Új pull request.](../img/git-newpr.png)
:width:`700px`
:label:`fig_git_newpr`


Végül nyújtsuk be a pull requestet a :numref:`fig_git_createpr` ábrán látható gombra kattintva. Győződjünk meg arról, hogy leírjuk a pull requestben elvégzett módosításokat.
Ez megkönnyíti a szerzők számára az áttekintést és a könyvbe való beolvasztást. A módosításoktól függően ez azonnali elfogadáshoz, elutasításhoz, vagy valószínűbben a módosításokra vonatkozó visszajelzéshez vezet. Ha beépítettük a visszajelzéseket, minden rendben van.

![Pull request létrehozása.](../img/git-createpr.png)
:width:`700px`
:label:`fig_git_createpr`


## Összefoglalás

* A GitHubot használhatjuk a könyvhöz való hozzájáruláshoz.
* Kisebb módosítások esetén közvetlenül szerkeszthetjük a fájlt a GitHubon.
* Nagyobb módosítás esetén ágaztassuk el a tárolót, szerkesszük helyben, és csak akkor járuljunk hozzá, ha készen vagyunk.
* A pull requestek segítségével kerülnek összegyűjtésre a hozzájárulások. Próbáljunk nem túl nagy pull requesteket benyújtani, mivel azokat nehéz megérteni és beépíteni. Jobb több kisebb pull requestet küldeni.


## Gyakorlatok

1. Csillagozzuk meg és ágaztassuk el a `d2l-ai/d2l-en` tárolót.
1. Ha észreveszel valami fejlesztésre szorulót, például egy hiányzó hivatkozást, nyújts be pull requestet.
1. Általában jobb gyakorlat pull requestet egy új ágon létrehozni. Tanuljuk meg, hogyan kell ezt megtenni a [Git elágaztatással](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell).

[Discussions](https://discuss.d2l.ai/t/426)
