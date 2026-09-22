[English](README.md) | **繁體中文**

# SOMTrack 3.0

追蹤資料的運動指標、多變量分析與出版級圖表——並附上決定這些圖表能宣稱什麼的統計檢定。

這是 `SOM tracking_v1.2b.ijm`（Wang、Ho & Liao，國立臺灣海洋大學，2020）的 Python 重構版本。原 ImageJ 巨集從 TrackMate 的輸出計算約 24 個運動指標，以 Kohonen 映射圖分群並繪圖。SOMTrack 保留這套流程，修正了 [`AUDIT.md`](AUDIT.md) 中記載的計算錯誤，並新增 50 個指標、五種 SOM 變體、一個收錄十一種投影方法的註冊表，以及一個能回答「圖形無法回答之問題」的統計分析層。

**3.0 版的存在，是因為一張圖不是證據。** 2.0 版能讓你看到各組「看起來」不同，卻無法告訴你它們是否「真的」不同。這個版本加入了回答這個問題的檢定——PERMANOVA 與 PERMDISP、能量檢定、對照置換虛無分布的交叉驗證分類——並且不允許監督式投影被誤當成證據。它也會把答案寫下來：每次執行都會產生一份結論報告，附上方法段落，以及本次實際使用之每個演算法的參考文獻。

### 安裝

**Windows，不需要任何 Python 經驗。** 雙擊 **`install.bat`**，再雙擊 **`start.bat`**。安裝程式會檢查是否有 Python 3.10 以上版本、在 `.venv\` 建立獨立環境、把所有套件安裝進去，並回報最後有哪些投影方法可用。所有東西都不會安裝到系統層級，因此不會干擾其他 Python 專案；刪除 `.venv\` 即可完全復原。重複執行安裝程式既安全又快速。

`start.bat` 也可以當作命令列使用——`start.bat methods`、`start.bat run data.csv --features-table -o results`。

**其他作業系統：**

```bash
pip install -r requirements.txt
python -m somtrack            # 桌面程式
```

### 3.0 版新功能

| | |
|---|---|
| **我的分組站得住腳嗎？** | PERMANOVA、PERMDISP 與能量 *k* 樣本檢定，加上對照置換虛無分布的交叉驗證分類。 |
| **重複（replicate）終於被納入考量** | 置換與交叉驗證的分折都依重複欄位分區，培養皿效應不再能偽裝成處理效應。 |
| **十一種投影，一個註冊表** | PCA、MDS/PCoA、t-SNE、UMAP、densMAP、PaCMAP、PHATE、Isomap、核 PCA，以及加上防護機制的 LDA、PLS-DA、SVM 與監督式 UMAP。 |
| **可稽核的投影** | R<sub>NX</sub> 曲線、對照打亂虛無分布的逐點可靠度、亂數種子穩定性，以及方法之間的一致性矩陣。 |
| **呈現高原區的參數掃描** | 適用於任何方法、可依任何準則評分，並保留、繪出實際的配置結果。 |
| **「比較」步驟** | 所有方法並列並附品質分數與一致性矩陣；也可以在程式內執行參數掃描，一鍵採用結果。 |
| **結論報告** | `RESULTS_REPORT.html`、`RESULTS_REPORT.md` 與 `methods.txt`——依本次執行實際內容產生的白話結論、方法段落與參考文獻清單。 |
| **配色是一套原則，而非一組色盤** | 組別超過八個時，圖表改用分面圖（small multiples），不再以顏色區分；彩虹色譜已移除。 |
| **一個問題取代二十一個設定** | 分析頁面一開始只有「分析方案」選單，其他設定全部收合。 |
| **繁體中文** | 桌面程式提供台灣繁體中文介面；報告先以英文撰寫，其後附上完整的中文翻譯。詳見[介面與報告語言](#介面與報告語言)。 |

2.0 版的所有功能仍可使用，2.0 版的 `analysis_config.json` 也仍可載入。

---

## 兩種開始方式

**A. 座標表格。** 讓 SOMTrack 讀取一個或多個偵測點層級的表格（TrackMate 的「Spots in tracks statistics」，或任何含有 track／x／y／time 欄位的表格），並選擇要計算的指標。欄位名稱會自動偵測。

**B. 多維度表格。** 如果你已經有一張每隻動物一列、包含自己測量值的表格，直接載入即可進行分群。

無論哪種方式，最後都會得到相同的特徵矩陣，之後的所有步驟完全一樣。

---

## 計算哪些指標

共 77 個指標，分為九大類。每個指標都附有單位、一行說明與所需的輸入欄位；介面會隱藏表格中缺少必要欄位的指標。

| 類別 | 範例 |
|---|---|
| 基本運動學 | 速度平均與標準差、加速度、路徑長度 |
| 角度／方向 | 轉向速率、蜿蜒度、**圓形**平均行進方向、合成向量長度 R、Rayleigh p、轉向熵、轉向峰度、左右偏好 |
| 路徑幾何 | 直線度、蜿蜒指數、迴轉半徑、凸包面積、探索比、Katz 碎形維度 |
| 擴散 | MSD 異常擴散指數 α、廣義擴散係數、擬合 R² |
| 持續性與記憶 | 方向相關時間、持續長度、速度 Hurst 指數（DFA） |
| 間歇性（CTRW） | 移動時間比例、停止頻率、移動／停頓持續時間、爆發性 B、記憶係數 |
| 速度分布 | 峰值速度、變異係數、偏態、均方根加速度、均方根急動度 |
| 振盪與生物物理 | 擺動頻率與頻譜純度、側向振幅、**Strouhal 數**、**Reynolds 數**、以體長計的速度 |
| 形態／強度 | 面積、橢圓軸長、長寬比及其變異係數、**軸向**身體方向、偏航漂移、強度、對焦指數 |

v1.2 已有的指標在介面中標示為 `(v1.2)`，而 **v1.2 舊版指標組** 預設組合可完整重現該清單，方便直接比較。

`python -m somtrack metrics` 會列出完整的指標目錄。

---

## 分群與投影

### 自組織映射圖（SOM）

| 演算法 | 適用時機 | 引用 |
|---|---|---|
| **batch**（批次，預設） | 向量化運算、結果固定，比逐筆更新規則快約 100 倍。標準選擇。 | Kohonen 1999 |
| **supervised**（監督式） | XY 融合 SOM。以權重 α 將處理標籤作為第二個資料區塊納入訓練。**在建構上就會把你的組別分開**——請報告 α，並以交叉驗證的統計結果（而非映射圖）作為證據。 | Melssen, Wehrens & Buydens 2006 |
| **relevance**（相關性學習） | 批次 SOM 加上 GRLVQ 相關性學習。為每個指標學習一個權重，讓少數有資訊的指標不再被二十個無資訊的指標淹沒。 | Hammer & Villmann 2002 |
| **online**（線上逐筆） | 原巨集使用的逐筆 Kohonen 更新規則，為了可重現性而保留。 | Kohonen 1982 |
| **growing**（成長型） | 六角形網格上的 GSOM。在量化誤差高的地方增生節點，映射圖大小因此不再是需要憑空猜測的超參數。 | Alahakoon, Halgamuge & Srinivasan 2000 |

另外還提供：六角形或矩形網格、可選的環面邊界、PCA 初始化，以及以量化誤差、拓撲誤差（Kiviluoto 1996）、命中加權的組別純度與標準化相互資訊表示的品質指標。

### 投影方法註冊表

`python -m somtrack methods` 會列出每個方法的參數、白話說明與引用文獻。新增一個方法只需要一個檔案加上一次 `register()` 呼叫——設定、桌面程式的表單、參數掃描、比較圖與參考文獻清單都會自動納入。

| 方法 | 用途 | 引用 |
|---|---|---|
| **PCA** | 線性、結果固定、可直接解讀。建議從這裡開始。 | Hotelling 1933 |
| **MDS / PCoA** | 唯一一種所用距離正是 PERMANOVA 所檢定之距離的投影，因此圖與統計描述的是同一件事。 | Torgerson 1952; Gower 1966 |
| **t-SNE** | 強化局部鄰域。群集的大小與間距沒有意義。 | van der Maaten & Hinton 2008 |
| **UMAP** | 介於 PCA 與 t-SNE 之間。 | McInnes, Healy & Melville 2018 |
| **densMAP** | 同時保留局部密度的 UMAP，讓「這組變異較大」仍然看得出來。 | Narayan, Berger & Cho 2021 |
| **PaCMAP** | 在局部細節與整體形狀之間取得比 t-SNE 或 UMAP 更好的平衡，而且幾乎不需要調參。 | Wang, Huang, Rudin & Shaposhnik 2021 |
| **PHATE** | 適用於呈梯度的生物現象——劑量、時間、發育——而非彼此分離的群集。 | Moon et al. 2019 |
| **Isomap**、**核 PCA** | 彎曲的結構。 | Tenenbaum, de Silva & Langford 2000; Schölkopf, Smola & Müller 1998 |
| **LDA（收縮）** | 經典的判別分析，經正則化，使其在指標數接近樣本數時仍保持穩定。 | Fisher 1936; Ledoit & Wolf 2004 |
| **PLS-DA** | 體學（omics）文獻中慣用的判別投影。 | Barker & Rayens 2003 |
| **線性 SVM** | 組別之間最寬的間隔，且權重可解讀。 | Cortes & Vapnik 1995 |
| **監督式 UMAP** | 用來展示你已經證實的結構。 | McInnes et al. 2018 |

預設值依樣本數 *n* 推導，而不是照抄論文：論文中發表的 perplexity 與鄰居數是針對數萬個點選定的，套用到兩百隻動物上，畫出來的是演算法的樣貌，而不是實驗的樣貌。實際使用的數值會記錄在方法段落中。

> #### ⚠ 監督式投影的防護機制
>
> 監督式投影在畫圖之前就已經拿到答案，因此它一定會把組別分開——對真實資料如此，對雜訊也一樣。這一點已被多次獨立記載：組間 PCA（Cardini, O'Higgins & Rohlf 2019）；PLS-DA，其得分圖上的分離已被證明不帶任何資訊，因為隨機資料也會產生相同的圖（Westerhuis et al. 2008）；以及監督式 UMAP，其 issue tracker 中滿是在訓練資料上分得很開、在保留資料上卻崩解的嵌入結果。
>
> 因此 SOMTrack 讓這些方法**預設為關閉**，啟用時必須明確確認，每個面板都會印上警示，並且每種方法都畫兩次：一次以全部資料擬合，一次以**折外**（out-of-fold）方式——每個樣本都由不含該樣本所擬合的座標軸來放置。樣本內分得開、折外卻崩解，就是「沒有差異」的誠實樣貌。交叉驗證的形式請見 Cardini & Polly（2020）。

### 投影可以相信到什麼程度

每個投影都會附上：

* 由共同排序矩陣（co-ranking matrix）計算的 **R<sub>NX</sub>(K) 及其在 log-K 軸上的曲線下面積**（Lee & Verleysen 2009; Lee et al. 2013）——一個可跨方法比較、同時涵蓋可信度與連續性的數字（Venna & Kaski 2006）；
* **逐點可靠度**，對照打亂嵌入後建立的虛無分布，做法類似 scDEED（Xia, Lee & Li 2024），讓由不可靠點組成的群集能被辨識為假象。將不可靠點數最小化也可以作為參數掃描的準則，讓「該用多少 perplexity」變成一個有圖可答的問題；
* **亂數種子穩定性**，透過重複執行並以 Procrustes 對齊來評估（Gower 1975）；
* 方法之間的**一致性矩陣**，因為在多種投影中都存在的結構是資料本身的特性，只在一種投影中看得到的結構則是該演算法的特性。

### 參數掃描

```bash
somtrack scan features.csv --features-table --method tsne --criterion dubious_fraction
```

適用於 SOM 及任何已註冊的投影，評分準則可為 `rnx_auc`、`dubious_fraction`、`trustworthiness` 或 `group_silhouette`。掃描會保留產生的每一個配置，並回報**高原區（plateau）**——所有分數落在最佳值容許範圍內的設定。因為單一最佳格子幾乎從來不會比鄰近的格子好上多少；把這一點說清楚，才能避免使用者不斷調參，直到圖形符合自己的假設為止。

### 第二層分群

在碼簿上進行 k-means／Ward／GMM 分群（Vesanto & Alhoniemi 2000），掃描一段 k 值範圍，並以輪廓係數（Rousseeuw 1987）、Davies-Bouldin（1979）與 Calinski-Harabasz（1974）指數評分。選定的 k 值與其分數會標示在圖上。

---

## 分組站得住腳嗎？

這是 2.0 版所沒有的一層。三個問題，依讀者提問的順序排列。

### 1. 各組之間到底有沒有差異？

| 檢定 | 回答的問題 | 引用 |
|---|---|---|
| **PERMANOVA** | 各組的平均是否不同？報告 pseudo-F、置換 p 值，以及**作為效應量的 R²**——即組別所能解釋的多變量變異比例。 | Anderson 2001 |
| **PERMDISP** | 各組的*離散程度*是否不同？ | Anderson 2006 |
| **能量 k 樣本檢定** | 各組的分布*是否有任何*不同——無論是位置、尺度或形狀？ | Székely & Rizzo 2013; Gretton et al. 2012 |
| **馬氏距離 D** | 將合併共變異數收縮後再求反矩陣，各組質心相距多遠？ | Mahalanobis 1936; Ledoit & Wolf 2004 |

**PERMANOVA 與 PERMDISP 一律同時報告**，因為這兩種原因都會使 PERMANOVA 拒絕虛無假設。PERMANOVA 顯著、同時 PERMDISP 也顯著，可能代表「這些動物的變異較大」，而不是「這些動物游得比較快」——這是不同的生物學結果，而前者曾多次被當成後者發表。SOMTrack 會在結論中用文字說明是哪一種情況。

### 2. 保留樣本能被判入正確的組別嗎？

交叉驗證分類——收縮 LDA、線性 SVM、PLS-DA、邏輯斯迴歸或隨機森林——以**平衡準確率**（Brodersen et al. 2010）與 Cohen's κ（Cohen 1960）評分，並附拔靴法信賴區間與**置換虛無分布**（Ojala & Garriga 2010）。不採用對交叉驗證準確率所做的二項檢定：該檢定過於寬鬆，會報告出並不存在的差異（Noirhomme et al. 2014）。

**混淆矩陣與各組別配對的準確率**會與單一數值並列呈現，因為「A 與其他所有組都分得開，B 與 C 彼此則分不開」才是在生物學上有用的陳述，而單一數值無法表達這件事。

### 3. 哪些指標承載了差異？

* **Haufe 轉換後的活化模式**（Haufe et al. 2014）。分類器權重大的指標，可能*完全不*帶組別資訊，只是用來抵銷另一個指標中的相關雜訊——也就是抑制變數。把這種權重解讀為「這個指標能區分組別」，是發表錯誤結論的一條有據可查的途徑。SOMTrack 會將原始權重與活化模式並列報告，並說明應該看哪一欄。
* **單一指標與相關指標群的置換重要性**（Breiman 2001; Altmann et al. 2010）。轉向速率、蜿蜒度與直線度測量的幾乎是同一件事，逐一打亂會同時低估三者——模型只會改讀其他指標。因此也會以整群方式打亂。
* **以原始單位表示的效應量**：經小樣本校正的 Hedges' *g*（Hedges 1981）與偏誤校正加速（BCa）拔靴法信賴區間（Efron 1987），以估計圖（estimation plot）呈現（Ho et al. 2019）。
* 2.0 版的逐指標 ANOVA／Kruskal-Wallis 分析，搭配 Benjamini-Hochberg FDR 校正（Benjamini & Hochberg 1995），維持不變。

### ⚠ 重複，以及為什麼你的 p 值太小

`PreparedData` 從 2.0 版起就帶有重複欄位，卻沒有在統計上使用它。現在會了。

如果八隻動物來自同一個培養皿，它們是**一個**實驗單位的八次測量，而不是八個重複（Hurlbert 1984; Lazic, Clarke-Williams & Munafò 2018）。SOMTrack 會偵測這種結構並據以調整：

* **巢套（nested）**（每個重複只屬於一個組別）：置換時以整個重複為單位重新分配，交叉驗證也會把一個重複完整保留在同一折之內；
* **交叉（crossed）**（每個重複包含多個組別）：標籤只在每個區集*之內*置換，這樣才能保留區集設計（Anderson & ter Braak 2003）；
* `--unit replicate` 更進一步，在檢定前先於每個重複內取平均——這是最保守的做法。

報告會說明偵測到的是哪一種設計，以及因此做了什麼處理。在只有純粹培養皿效應、沒有處理效應的模擬資料上，忽略這一點會得到 *p* < 0.01 與高於 0.85 的平衡準確率；正確處理則兩者都不會出現。有一項測試專門確保這一點。

---

## 結論報告

每次執行都會寫出 `RESULTS_REPORT.html`、`RESULTS_REPORT.md` 與 `methods.txt`。

**結論**由數值自動產生，因此不會與其下的表格相矛盾。以下是報告中文翻譯部分的樣貌（報告中英文版在前）：

> 有強力證據顯示組別之間有差異。以歐氏距離進行 PERMANOVA：pseudo-F = 4.84，R2 = 0.085……，p = 0.0050，199 次置換。PERMDISP（組內離散程度的同質性）：F = 2.74，p = 0.055。各組的多變量平均不同，而組內變異程度相當，因此這是表型的真實位移，而非變異程度的改變。……可區分的組別配對：control 對 high（平衡準確率 0.96）；……無法區分的組別配對：low 對 mid（平衡準確率 0.59）。

**方法段落**描述本次執行實際做了什麼，由每個步驟在執行時自行記錄而組成——略過 UMAP 的執行就不會引用 McInnes et al.：

> **統計。** 考量重複結構的置換與交叉驗證，重複巢套於組別之內（共 20 個重複，每個重複只屬於一個組別），因此以重複作為實驗單位（Anderson & ter Braak, 2003; Hurlbert, 1984; Lazic et al., 2018）；PERMANOVA，以歐氏距離進行置換多變量變異數分析，以 R 平方作為效應量（Anderson, 2001）[permutations = 999, distance = euclidean]；……
>
> **視覺化。** SuperPlots，依生物重複上色（Lord et al., 2020）；附拔靴法信賴區間的估計圖（Ho et al., 2019）。

之後是去除重複、附 DOI 的**參考文獻清單**，以及所執行方法附帶的注意事項。

英文報告之後，三個檔案都會附上一份**完整的繁體中文翻譯**，因此兩種語言絕不會在同一行混用。哪些內容會翻譯、哪些不會，請見[介面與報告語言](#介面與報告語言)。

---

## 圖表

約有 40 種圖可用；預設每次執行會寫出大約十二張。圖的順序與報告的閱讀順序一致——先是結論，再來是支持結論的證據，最後是描述性的圖——因為讀了三頁就停下來的讀者，應該已經拿到答案，而不是一張還得自己解讀的自組織映射圖。

`report.profile = "core"` 只寫出承載結果的圖；`"full"` 寫出全部。三十頁的 PDF 沒有人會讀。

### 承載主張的圖

| | |
|---|---|
| **結論卡** | 由數值產生的白話結論。放在第一頁。 |
| **分離摘要** | PERMANOVA 的 R² 與 PERMDISP 並列，並畫出組內離散程度，讓離散效應不會被誤讀為位移。 |
| **兩兩分離** | 具體是哪些組別配對之間有差異，經 FDR 校正。 |
| **可分類性** | 觀察到的平衡準確率對照置換虛無分布，並附混淆矩陣。 |
| **活化模式** | Haufe 轉換後的模式與原始權重並列，並註明應該看哪一個。 |
| **重要性** | 單一指標與相關指標群。 |
| **效應量森林圖** | Hedges' *g* 附拔靴法信賴區間（標出 Cohen 1988 的參考值，作為慣例而非門檻）。 |
| **SuperPlots** | 每個個體依重複上色，並疊上每個重複各自的平均（Lord et al. 2020）。 |
| **估計圖** | 上方為原始資料，下方在對齊的座標軸上呈現效應量及其拔靴法信賴區間（Ho et al. 2019）。 |
| **樣本內 vs. 折外** | 能區分真實差異，與監督式方法本身被給予之差異的面板。 |

### 稽核圖形本身的圖

方法比較格（每個投影附品質標章）、R<sub>NX</sub> 曲線、逐點可靠度、亂數種子穩定性、投影一致性、Shepard 圖、標示高原區的掃描曲面，以及每個掃描設定下實際配置的縮圖格。

### 從 v1.2 移植、並修正計算的圖

HSB 組成圖、各組映射圖、指標軸向量、k-means 節點疊圖、朝次佳節點方向的樣本擺放。另外還有 U 矩陣、命中直方圖、以原始單位呈現的成分平面圖、以圓餅符號呈現的節點組成、訓練診斷曲線、附標籤的 kymograph，以及富集圖。

### 顏色

顏色是一項決策，而不是一組色盤。`viz/palette.py` 回傳的是顏色**加上一項指示**，而圖表層會遵守它：

* **≤ 8 組**——Okabe-Ito，色覺辨識障礙友善的標準配色（Okabe & Ito 2008; Wong 2011）。
* **≤ 10 組**——Paul Tol 的 muted 配色。
* **超過 10 組**——答案是*不要用顏色*。當要求的類別數超過任何色盤能區分的數量時，`categorical()` 會回傳 `mode="facet"`，圖表改畫分面小圖（small multiples），而不是發明三十種沒人能對應到圖例的顏色。也可以只替焦點組上色，其餘組以灰色呈現。
* **類別顏色絕不取自連續色譜。** 在 *viridis* 或 *turbo* 上取 *n* 個點，會暗示類別之間存在它們並沒有的順序；彩虹色譜則會製造出資料中不存在的邊界（Crameri, Shephard & Heron 2020; Crameri 2024）。2.0 版在超過十二組時使用的 `turbo` 備援已經移除；指標梯度圖現在改用連續色譜，因為排序*確實*是有序的。
* **連續色譜在感知上是均勻的**；適合列印的預設色譜 *cividis* 在灰階與綠色盲（deuteranopia）下也能辨識。
* **發散色譜只用於零值有意義的地方**，並且一律以零為中心對稱，讓一個數值的顏色不會取決於資料剛好涵蓋的範圍。`symmetric_norm()` 會強制這一點。
* **灰色保留給背景資訊**，絕不代表任何實驗組別。
* 組別身分以**雙重方式**編碼——顏色*加上*標記形狀。
* `figure.cvd_proof = True` 會把每張主要圖表輸出為綠色盲、紅色盲、藍色盲與灰階版本，讓圖表能在投稿前、而非審稿後被檢查。

SOM 上的節點顏色在任何組數下都與圖例色塊取自同一個色盤——2.0 版在啟用備援配色後並非如此。

### 各組透明度（需求清單第 7 項）

在各組的映射圖上，節點不透明度依該節點中*該組*的樣本數分級：

| 節點中的樣本數 | 不透明度 |
|---|---|
| 0 | 20 % |
| 1 | 40 % |
| 2 | 60 % |
| ≥ 3 | 100 % |

因此在大型映射圖上，一個組別所佔的範圍會是唯一飽和的區域。這些門檻可在「匯出」頁面中調整。

### 輸出格式

* **PDF 與 SVG**——向量圖，文字保留為*文字*（`pdf.fonttype = 42`、`svg.fonttype = "none"`），因此不必重新執行，就能在 Illustrator 或 Inkscape 中編輯每個標籤。`SOMTrack_report.pdf` 依序收錄整套圖表。
* **PNG**——預設 600 dpi。
* **MP4**——映射圖在訓練過程中自我組織的動畫，附量化誤差曲線。若 PATH 中找不到 ffmpeg，則改輸出 GIF。
* **CSV + 一個 `.xlsx` 活頁簿**——所有結果表格，包括統計結果。
* **`RESULTS_REPORT.html` / `.md`**——結論、證據、方法段落與參考文獻清單；先英文，後附繁體中文翻譯。
* **`methods.txt`**——單獨列出的方法段落與參考文獻，可直接貼上，其後附中文方法段落。
* **`analysis_config.json`**——重新載入即可完全重現該次執行。

---

## 命令列

```bash
# 產生一組四處理的模擬資料集供試用
python -m somtrack demo ./demo_data

# 完整分析，包含統計分析層
python -m somtrack run "demo_data/*.csv" \
    --pixel-size 1.6 --frame-interval 0.05 -o results

# 使用你已有的表格，每隻動物一列
python -m somtrack run features.csv --features-table -o results

# 指定投影方法，並以重複作為實驗單位
python -m somtrack run features.csv --features-table \
    --projections pca,mds,pacmap --unit replicate --permutations 9999 -o results

# 同時執行監督式投影（附防護機制）
python -m somtrack run features.csv --features-table --allow-supervised -o results

# 只輸出英文報告，不附中文翻譯
python -m somtrack run features.csv --features-table --translation none -o results

# 掃描投影方法的超參數，以不可靠點數評分
python -m somtrack scan features.csv --features-table \
    --method tsne --criterion dubious_fraction -o scan

# 掃描 SOM，與 2.0 版相同
python -m somtrack scan "demo_data/*.csv" --pixel-size 1.6 -o scan

# 列出可用的項目
python -m somtrack methods      # 投影方法、參數與引用文獻
python -m somtrack metrics      # 指標目錄
```

## 作為函式庫使用

```python
from somtrack import AnalysisConfig, pipeline, recipes
from somtrack.io_tables import build_spot_dataset, detect_spot_columns, load_table

cfg = recipes.apply_recipe(AnalysisConfig(), "test_groups")
cfg.track.pixel_size = 1.6
cfg.track.frame_interval = 0.05
cfg.stats.unit_of_analysis = "replicate"
cfg.export.out_dir = "results"

cm = detect_spot_columns(load_table(files[0]))
spots = build_spot_dataset(files, cm, groups, replicates)
feats = pipeline.features_from_spots(spots, cfg)
res = pipeline.run_analysis(feats, cfg)

print(res.verdict.text())                       # 以文字呈現的結論（英文）
print(res.verdict.text().render("zh_TW"))       # 同一段結論的中文版
print(res.separation.permanova_R2, res.separation.permanova_p)
print(res.classification.summary_line())
pipeline.export_all(res)                        # 圖表、表格與報告
```

統計與投影也可以單獨使用：

```python
import numpy as np
from somtrack.analysis import DataContext, run_projections
from somtrack.analysis.scan import scan
from somtrack.stats import analyse_blocks, analyse_separation, classify

structure = analyse_blocks(group_codes, replicates)   # 巢套？交叉？還是都不是？
sep = analyse_separation(X, group_codes, labels, structure, n_permutations=999)
cls = classify(X, group_codes, labels, structure, model="svm_linear")

ctx = DataContext(X=X, feature_names=names, group_codes=group_codes,
                  group_values=labels, replicates=replicates)
projections = run_projections(ctx, ["pca", "pacmap", "mds"])
best = scan(ctx, "tsne", criterion="dubious_fraction")
print(best.plateau_text())
```

---

## 程式架構

```
somtrack/
  config.py       所有參數，組成一棵可序列化的 dataclass 樹
  citations.py    參考文獻目錄，以及每次執行的方法紀錄
  recipes.py      以所要回答的問題命名的完整分析方案
  report.py       結論報告：結論、方法、參考文獻，以及其翻譯
  i18n.py         介面語言，以及自帶翻譯的句子
  locales/        翻譯目錄（zh_TW.py：繁體中文）
  io_tables.py    載入、欄位自動偵測、資料集組裝
  metrics.py      指標註冊表與逐軌跡的運動學引擎
  preprocess.py   縮放、缺失值處理、共線指標剔除
  som.py          網格、五種 SOM 變體、品質指數、平行掃描
  gsom.py         六角形網格上的成長型 SOM
  nodecluster.py  第二層分群與 k 值選擇
  association.py  哪些指標解釋了分組（逐指標分析層）
  analysis/
    registry.py     MethodSpec / ParamSpec：由方法自行宣告
    projection.py   所有 2D 投影共用的結果型別
    quality.py      共同排序矩陣、R_NX、逐點可靠度、亂數種子穩定性
    scan.py         適用於任何方法的參數掃描，附高原區
    methods/        linear.py、manifold.py、supervised.py
  stats/
    blocks.py       對本實驗而言「獨立」代表什麼
    omnibus.py      PERMANOVA、PERMDISP、能量檢定、馬氏距離
    classify.py     搭配依設計而定之置換虛無分布的交叉驗證
    interpret.py    Haufe 模式、分群置換重要性
    effects.py      附拔靴法信賴區間的 Hedges' g
    verdict.py      報告最上方的結論段落
  viz/
    palette.py      配色原則，包括何時該停止使用顏色
    stats.py        結論卡、分離、可分類性、活化模式
    compare.py      方法比較格、掃描曲面、掃描縮圖、一致性
    reliability.py  R_NX 曲線、逐點可靠度、穩定性、Shepard 圖
    estimation.py   SuperPlots、估計圖、效應量森林圖
    (maps, groups, vectors, embed, assoc, anim, style, hexgeom)
  export.py       PNG / PDF / SVG / MP4 / CSV / XLSX 輸出
  pipeline.py     桌面程式與命令列共用的流程調度
  ui/
    main_window.py  七個步驟的精靈式介面
    pages.py        每個步驟一個類別，包括「結果」與「比較」
    forms.py        由 ParamSpec 建立、附「自動」勾選框的參數控制項
    widgets.py      檔案清單、指標樹、圖庫、紀錄窗格
  cli.py          命令列
  demo.py         模擬資料產生器
install.bat       Windows：檢查 Python、建立 .venv、安裝相依套件
start.bat         Windows：開啟桌面程式，或將參數傳給命令列
tests/            以已知答案的資料測試統計方法，並測試註冊表、報告、
                  桌面程式的串接與翻譯
```

桌面程式與命令列都透過 `pipeline.py` 執行，因此在相同設定下，無論從哪一邊啟動，產出都完全相同。

---

## 沒有統計背景也能使用

分析頁面一開始只問一個問題——*你想了解什麼？*——以及一個「執行分析」按鈕。其他所有設定都收合起來，直到需要時才展開。

| 分析方案 | 適用情境 |
|---|---|
| **探索資料** | 安全的預設選擇。繪製映射圖、投影、檢定。 |
| **檢定我的組別是否不同** | 完整的統計分析層，並刻意關閉監督式投影。 |
| **找出重要的測量指標** | 相關性學習 SOM、活化模式、分群置換重要性、效應量。 |
| **確認圖中的結構是真的** | 六種投影、多個亂數種子、可靠度與一致性。 |
| **尋找劑量或時間梯度** | 加入 PHATE，適用於呈連續序列、而非分離類別的生物現象。 |
| **重現 v1.2 巨集** | 原始流程，用於與舊結果比較。 |

分析方案是資料，而不是一條程式路徑——它只是一個修改了部分欄位的普通 `AnalysisConfig`——因此會透過相同的 JSON 機制載入，而且它設定的每個參數在下方都仍然看得到、也仍然可以修改。

執行之後，「**比較**」步驟會把所有投影放在同一頁並附上品質分數，讓分得最開的那一個，無法悄悄變成你拿出來展示的那一個。在同一頁中，你也可以掃描某個方法的超參數、查看高原區，然後按下「下次執行時使用這些設定」——這會把數值固定為明確的選擇，方法段落也會將其記載為你的選擇。

在底層，可由資料推導的參數都有一個預設勾選的「**自動**」勾選框，並且*會顯示它所選的數值*。不適用的控制項會被隱藏，而不是變成灰色：不適用的控制項，仍然是眼睛必須讀過、再予以忽略的東西。

結果頁面一打開就是結論，而不是一整排圖庫。

---

## 介面與報告語言

**桌面程式**提供英文與台灣繁體中文介面。程式會使用從 **Language / 語言** 選單所選的語言；尚未選擇過時，使用環境變數 `SOMTRACK_LANG`（`zh_TW` 或 `en`）指定的語言；兩者皆無時則依系統語言——因此台灣的 Windows 第一次開啟時就是中文。切換語言後需重新啟動才會生效，程式會詢問是否立即重新啟動。每個標籤、提示、工具提示與紀錄訊息都已翻譯，來自註冊表的文字也是：方法說明與注意事項、參數名稱與說明、指標目錄與分析方案。選項清單會顯示翻譯後的標籤，並在括號中附上設定值，例如 `z 分數標準化（zscore）`，讓你看到的內容能與方法段落和 `analysis_config.json` 中記錄的值相互對應。指標樹中的指標名稱維持原樣，因為它們是所有輸出表格的欄位名稱；旁邊的說明會以中文指標名稱開頭。

**報告**（`RESULTS_REPORT.html`、`RESULTS_REPORT.md`、`methods.txt`）先以英文撰寫，接著再以繁體中文完整寫一次，作為第二份完整的文件，而不是逐行對照。兩者絕不混用。

| 翻譯成中文 | 維持英文 |
|---|---|
| 標題與內文段落 | 圖中的所有文字 |
| 結論及其注意事項 | 表格內容與欄位名稱 |
| 表格標題與圖表索引（圖說） | 指標、組別與參數名稱 |
| 方法段落及其注意事項 | 參考文獻清單（只在英文部分列出一次） |

數值只計算一次、再以兩種語言分別呈現，因此翻譯中的每個數值都與英文報告相同。翻譯可以在「匯出」頁面關閉，或在命令列使用 `--translation none`，或在 `analysis_config.json` 的 `report` 區段中設定 `"translation": ""`。

新增一種語言只需要一個檔案：`somtrack/locales/<語言代碼>.py`，其中包含 `MESSAGES`（英文原文 → 譯文）與 `CONTEXTS`。介面文字透過 `i18n.tr()` 翻譯；分析時產生的句子則是 `i18n.Text` 物件——它本身就是英文字串，之後還能再呈現為譯文——這就是單次執行能寫出報告兩種語言版本的方式。只要翻譯漏掉了某個 `{placeholder}` 佔位符、中文執行時遇到沒有翻譯的字串，或中文出現在圖表中，`tests/test_i18n.py` 就會失敗。

---

## 附註

* **選用的投影方法。** `umap-learn` 可啟用 UMAP、監督式 UMAP 與 densMAP；`pacmap` 啟用 PaCMAP；`phate` 啟用 PHATE。缺少的方法會被略過，並附上一段會寫入報告的說明，而不是讓圖表集無聲無息地變短。`python -m somtrack methods` 會顯示已安裝的方法。
* **選用的配色套件。** `glasbey` 可改善十組以上的類別配色；`cmcrameri` 加入 Crameri 的科學用色譜。兩者都有備援方案。
* MP4 輸出需要 PATH 中有 `ffmpeg`。
* `shapely` 為選用套件，只用來讓節點群集合併後的外框更平滑。
* **效能。** 依本程式設計的使用情境調校——少於約 1000 個樣本、約 50 個指標——在此範圍內，O(n²) 的方法是合理的取捨。以 160 個樣本 × 24 個指標、999 次置換而言，完整執行約需一分鐘。置換虛無分布預設使用線性模型，因為在 999 次迴圈中跑隨機森林並不值得等待。
* **可重現性。** `analysis_config.json` 可完全重現一次執行，2.0 版的設定檔也仍可載入——包括其 `run_pca`／`run_tsne`／`run_umap` 旗標，會被轉換為新的方法清單。
* 將新數值與 v1.2 的輸出比較之前，請先閱讀 [`AUDIT.md`](AUDIT.md)——v1.2 有數項統計量是錯誤的，而且錯誤的方式可能改變結論，特別是角度平均與缺失值（NaN）的處理。
* **尚未完成。** PCC（arXiv:2503.07609）是 3.0 計畫中唯一尚未納入的方法：它還沒有公開發布的套件，而僅憑摘要重新實作一篇 2025 年的論文，不適合放在審稿人會看到的圖表背後。一旦有套件發布，只需新增一個檔案即可加入。SLISEMAP 已經註冊，但需要 `pip install slisemap`，而它會連帶安裝 PyTorch，因此不在預設安裝之中。
* [`ROADMAP.md`](ROADMAP.md) 說明了 3.0 版背後的設計考量。

---

## 參考文獻

SOMTrack 能執行的每個演算法都收錄在 `somtrack/citations.py`，並附有 DOI；每次執行的參考文獻清單，則依該次執行實際做了什麼而產生。3.0 版設計所依據的文獻如下（書目資料維持原文）：

**組別分離。** Anderson (2001) *Austral Ecology* 26:32 — PERMANOVA · Anderson (2006) *Biometrics* 62:245 — PERMDISP · Anderson & ter Braak (2003) *J Stat Comput Simul* 73:85 — 受限置換 · Székely & Rizzo (2013) *J Stat Plan Inference* 143:1249 — 能量統計 · Gretton et al. (2012) *JMLR* 13:723 — 核雙樣本檢定

**避免自欺。** Cardini, O'Higgins & Rohlf (2019) *Evol Biol* 46:303 — 虛假的組別分離 · Cardini & Polly (2020) *Evol Biol* 47:85 — 交叉驗證的 bgPCA · Westerhuis et al. (2008) *Metabolomics* 4:81 — PLS-DA 的驗證 · Ojala & Garriga (2010) *JMLR* 11:1833 — 分類器的置換檢定 · Noirhomme et al. (2014) *NeuroImage Clin* 4:687 — 二項檢定是錯的 · Hurlbert (1984) *Ecol Monogr* 54:187 與 Lazic, Clarke-Williams & Munafò (2018) *PLoS Biol* 16:e2005282 — 偽重複 · Haufe et al. (2014) *NeuroImage* 87:96 — 分類器權重的解讀

**投影及其品質。** Wang, Huang, Rudin & Shaposhnik (2021) *JMLR* 22:1 — PaCMAP · McInnes, Healy & Melville (2018) arXiv:1802.03426 — UMAP · Narayan, Berger & Cho (2021) *Nat Biotechnol* 39:765 — densMAP · Moon et al. (2019) *Nat Biotechnol* 37:1482 — PHATE · Lee & Verleysen (2009) *Neurocomputing* 72:1431 — 共同排序矩陣 · Xia, Lee & Li (2024) *Nat Commun* 15:1753 — scDEED · Gower (1975) *Psychometrika* 40:33 — Procrustes 分析

**圖表與顏色。** Lord, Velle, Mullins & Fritz-Laylin (2020) *J Cell Biol* 219:e202001064 — SuperPlots · Ho, Tumkaya, Aryal, Choi & Claridge-Chang (2019) *Nat Methods* 16:565 — 估計圖 · Crameri, Shephard & Heron (2020) *Nat Commun* 11:5444 — 科學傳播中顏色的誤用 · Crameri (2024) *Curr Protoc* 4:e1126 — 色盤的選擇 · Okabe & Ito (2008) 與 Wong (2011) *Nat Methods* 8:441 — 色盲友善配色 · Glasbey, van der Heijden, Toh & Gray (2007) *Color Res Appl* 32:304

**效應量。** Hedges (1981) *J Educ Stat* 6:107 · Efron (1987) *JASA* 82:171 — BCa 拔靴法 · Brodersen et al. (2010) *ICPR* — 平衡準確率 · Benjamini & Hochberg (1995) *JRSS B* 57:289
