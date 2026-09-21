# SOMTrack 3.0 — 多維度分析平台規劃書

> 目標情境：n < 1000 樣本、d < 50 維、**已知實驗分組**、各維度尺度與變異差異大、
> 需要能投稿的視覺呈現，操作者不一定是資訊或統計背景。
>
> 本文分四部分：(1) 現況架構分析、(2) SOTA 調查、(3) 四大主題規劃、(4) 實作藍圖與分期。

---

## 0. 一句話結論

現有架構的**分層是對的**（`config → preprocess → 演算法 → association → viz → export`，
GUI 與 CLI 共用 `pipeline.py`），但有三個硬結構擋住「放入更多分析工具」：

1. **演算法是寫死的列舉，不是註冊表** — `EmbeddingConfig` 用 `run_pca/run_tsne/run_umap`
   三個布林值，`run_embeddings()` 用寫死的 `jobs` 清單，`viz.plot_embedding_row()`
   甚至寫死 `("PCA", "t-SNE", "UMAP")` 的順序。新增一個方法要改 5 個地方。
2. **一次執行 = 一組參數** — `AnalysisResult` 只存**一個** SOM、一個 embedding dict。
   「同一方法的多組超參數」不是一等公民，要比較只能整條 pipeline 重跑。
3. **缺少「群體是否真的分得開」的統計層** — 目前只有 per-feature 的 ANOVA/Kruskal + BH-FDR
   （`association.py`），**沒有任何多變量整體檢定**，也**沒有 permutation test**。
   `multivariate_importance()` 報了 CV accuracy 與 baseline，但沒有 p 值。
   也就是說：你最想回答的那個問題，目前是靠**看圖**回答的 — 這正是文獻反覆警告的失誤模式。

另外有兩個一定要修的科學問題（詳見 §3.2）：

- **`replicates` 欄位被帶著走，但完全沒進入統計** — 只用在 `plot_group_maps(by_replicate=True)`。
  如果同一批（dish / clutch / 拍攝日）的個體被當成獨立樣本，所有 p 值都是 pseudoreplication。
- **監督式投影（supervised SOM、未來的 LDA/PLS-DA/SVM）在雜訊上也會畫出漂亮的分離**。
  現在 `supervised` SOM 已經在選單裡，README 也推薦它處理「差異細微」的情況 —
  這在沒有 out-of-fold 驗證與 permutation p 的情況下是**循環論證**。

---

## 1. 現況架構分析

### 1.1 資料流與契約

```
檔案 ──io_tables──▶ SpotDataset ──metrics──▶ FeatureDataset
                                                   │
                                            preprocess.prepare()
                                                   ▼
                                            PreparedData ◀── 全系統的共同契約
                                                   │
                     ┌─────────────┬───────────────┼───────────────┐
                     ▼             ▼               ▼               ▼
               som.train_som  embedding.      nodecluster.   association.
               (5 variants)   run_embeddings  cluster_nodes  analyse_associations
                     │             │               │               │
                     └─────────────┴───────┬───────┴───────────────┘
                                           ▼
                                    AnalysisResult
                                           │
                            ┌──────────────┴──────────────┐
                            ▼                             ▼
                   pipeline.build_figures()        result.tables()
                     (22 個 viz.plot_*)           (CSV / XLSX / methods.txt)
```

`PreparedData`（`preprocess.py:20`）是整個系統的樞紐，攜帶：
`X`(已縮放)、`feature_names`、`groups` / `group_codes` / `group_values`、
`replicates`、`sample_ids`、`keep_mask`、`inverse()`（回原始單位）、
`group_replicate_codes()`、`onehot_groups()`。

**這個契約設計得很好，新的分析層可以直接掛上去，不需要動它。**

### 1.2 可以直接沿用的資產

| 資產 | 位置 | 為什麼重要 |
|---|---|---|
| 可序列化的 config tree | `config.py` | 加一個 `StatsConfig` 就能把新參數納入 `analysis_config.json` 的復現機制 |
| `EmbeddingResult` 協定 | `embedding.py:26` | 已含 `coords / feature_axes / axis_labels / trustworthiness / continuity / params / axes_are_loadings` — 幾乎就是通用投影結果的雛形 |
| `make_figure()` 預留 caption/legend 空間 | `viz/style.py` | 版面紀律已建立，新圖直接複用不會撞版 |
| 向量文字輸出 (`pdf.fonttype=42`, `svg.fonttype="none"`) | `viz/style.py` | 投稿改圖的關鍵，已做對 |
| Okabe-Ito 預設色盤 | `viz/style.py:33` | 分類配色基礎正確 |
| `_bh_fdr()` | `association.py` | 多重比較校正已有，可直接給新檢定用 |
| `scan_parameters()` + joblib | `som.py:587` | 平行掃描骨架已在，只是綁死 SOM |
| `result.tables()` → CSV/XLSX 自動落地 | `pipeline.py:46` | **新統計表只要加進這個 dict，匯出完全免寫程式** |
| `write_methods_text()` | `export.py` | methods 段落自動生成，新統計要接進來 |

### 1.3 具體的結構性阻塞點

**(a) 演算法新增成本 O(5)**

```python
# embedding.py:142 — 寫死的 jobs 清單
if cfg.run_pca:   jobs.append(("PCA", run_pca))
if cfg.run_tsne:  jobs.append(("t-SNE", run_tsne))
if cfg.run_umap and umap_available(): jobs.append(("UMAP", run_umap))
```
```python
# viz/embed.py:167 — 寫死的方法名稱與順序
names = [n for n in ("PCA", "t-SNE", "UMAP") if n in embeddings]
```
新增 PaCMAP 要改：`EmbeddingConfig` 欄位、`run_pacmap()`、`jobs` 清單、
`viz.plot_embedding_row` 的 tuple、`ui/pages.py:_embed_tab()` 的 checkbox 與 commit。
**加 6 個方法 = 改 30 處。這是必須先解決的。**

**(b) 無法比較多組參數**

`AnalysisResult.embeddings: dict[str, EmbeddingResult]` 以**方法名**為 key，
所以「perplexity=5 的 t-SNE」和「perplexity=30 的 t-SNE」無法共存。
`scan_parameters()` 只回傳品質數字的扁平 list，**不保留座標**，
所以掃描結果無法畫成 embedding 的 small multiples。

**(c) 統計層的空缺**

`association.py` 目前提供：per-metric ANOVA/Kruskal + η²/ε² + Cohen's d + BH-q、
RF permutation importance + CV accuracy、SOM 梯度、node cluster profile、
hypergeometric enrichment。

**缺少**：
- 多變量整體檢定（PERMANOVA / PERMDISP / energy / MMD）
- CV 分類準確度的 **permutation null**（目前只有 `baseline_accuracy` 當參考）
- **replicate-aware** 的 CV 切分與受限置換
- 效果量的**信賴區間**（bootstrap）
- 線性分類器權重的 **Haufe 轉換**（否則「SVM 權重大 = 這個維度重要」是錯的）

**(d) UI 認知負擔**

`ClusterPage._som_tab()`（`ui/pages.py:600`）在一個 `QFormLayout` 裡一次攤開
11 個 SOM 參數，`_algo_changed()` 只是把不適用的控制項**變灰而非隱藏**。
再加上 embedding tab 的 6 個、node tab 的 4 個 = **21 個控制項同時在畫面上**。
這已經是你說的「看到整頁設定就不想用」的狀態，而我們準備再加 6 個演算法。

**(e) 配色的具體問題**

| 位置 | 目前 | 問題 |
|---|---|---|
| `viz/style.py:54` | 分組數 > 12 時用 `turbo` 連續色階取樣 | 用**彩虹連續色階編碼類別** — Crameri et al. (2020) 點名的典型誤用 |
| `viz/vectors.py:93, 244` | metric 箭頭 / gradient compass 用 `turbo` | 同上，而且 metric 數量常 > 20，任何色盤都失效 |
| `viz/maps.py:192` | U-matrix 用 `bone_r` | 非感知均勻 |
| `viz/assoc.py:118,160,211` | `RdBu_r` / `PuOr_r`，已用 `vmin=-lim, vmax=lim` | ✅ 發散色階對稱於 0 — 這點做對了，應該寫成強制規則 |
| 全域 | 無 CVD 模擬 / 灰階校驗輸出 | 投稿前無法自我檢查 |

`palette_hues()` 的設計意圖（node 顏色 = 該組 marker 的 hue）是對的，
但一旦 n_groups > 12 掉進 turbo fallback，node 色與 legend 色就不再一致。

**(f) 小瑕疵（擴充 config 時順手修）**

`config._from_plain()` 因為 `from __future__ import annotations`，
`f.type` 實際是**字串**，所以 `is_dataclass(ftype)` 那個分支是**死碼**；
巢狀 dataclass 實際靠下面第二個迴圈還原。目前行為正確，但加 `StatsConfig`
時應改用 `typing.get_type_hints()` 或直接刪掉死分支。

---

## 2. SOTA 調查（對應你的資料情境）

### 2.1 降維／投影方法

小樣本（n < 1000）、中維度（d < 50）、已知分組，這個組合有三個特點：
**(i)** 不需要為速度妥協，O(n²) 方法完全可行；
**(ii)** t-SNE/UMAP 的預設超參數是為 10⁴–10⁶ 筆資料調的，**在小 n 下必須重設**；
**(iii)** 已知分組 ⇒ 最大的風險不是「看不出結構」，而是「看出不存在的結構」。

| 方法 | 為什麼值得加 | 小 n 的注意事項 | 套件 |
|---|---|---|---|
| **PaCMAP** ⭐ | 2025 年的多項 benchmark 顯示 PaCMAP / TriMAP / t-SNE / UMAP 穩定排前五，而 PaCMAP 在 local↔global 權衡上最好、且**幾乎不需調參** | 預設 `n_neighbors` 的自動規則是給大 n 的；n<1000 時固定 `n_neighbors=10` | `pacmap` (PyPI) |
| **MDS / PCoA（classical + metric）** ⭐ | **唯一一個圖上距離 = PERMANOVA 檢定的那個距離**的投影。要讓「圖」與「統計」講同一件事，必須有它 | n<1000 直接算完整距離矩陣，零壓力 | scikit-learn |
| **densMAP** | UMAP 的一個 flag，保留**局部密度** ⇒「這組比較分散」在圖上讀得出來。這正好對應 PERMDISP 的離散度差異 | `densmap=True` | `umap-learn` |
| **PHATE** | 適合**連續梯度**（劑量反應、發育時間、濃度序列），locomotion 表型常是這種 | 小 n 時 `knn` 要調小 | `phate` |
| **LDA (shrinkage)** | 「群體能否分開」的古典答案；d 接近 n 時必須用 Ledoit-Wolf 收縮 | `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")` | scikit-learn |
| **PLS-DA / sparse PLS-DA** | 代謝體/組學領域的事實標準，loadings 與 VIP 可解釋 | **必須**配 double CV + permutation test | scikit-learn `PLSRegression` |
| **Linear SVM** | 決策邊界、margin、support vectors 可視化 | 權重要做 Haufe 轉換才可解釋 | scikit-learn |
| **SLISEMAP** | 監督式降維 + **局部解釋**：圖上鄰近 = 「被同一組維度解釋」。直接回答「哪些維度有解釋力」 | O(n²)，n<1000 剛好 | `slisemap` |
| **Kernel PCA / Isomap / Diffusion map** | 註冊表建好後幾乎零成本 | — | scikit-learn |
| **PCC**（arXiv 2503.07609, 2025） | 以 Pearson/Spearman 距離相關性為目標，global structure 保留 SOTA，且**可與 UMAP 疊加** | 尚無成熟套件，列為觀察 | — |

**❗ 最重要的一條設計規則 — 監督式投影的警戒線**

所有監督式投影（bgPCA、LDA、PLS-DA、SVM、supervised UMAP、XY-fused SOM）
**在純雜訊上也會產生完美的視覺分離**。這在三個獨立文獻線上都有記載：

- bgPCA：Cardini & Polly 等人證明 p 相對 n 大時出現**假性群體分離**，
  解法是 **cross-validated bgPCA**（把留出樣本投影上去看）。
- PLS-DA：Westerhuis et al. (2008) 的結論 —
  *score plot 顯示分離本身沒有意義，隨機資料也畫得出同樣的圖*。
- supervised UMAP：`lmcinnes/umap` issues #148 / #663 / #1116 紀錄了
  訓練集完美分離、留出集完全混在一起的典型過擬合。

⇒ **SOMTrack 的做法**：
1. 監督式投影**一律同時輸出 in-sample 與 out-of-fold 兩張圖**（在訓練 fold 上配適、
   投影留出樣本）。in-sample 分開但 out-of-fold 崩掉 = 誠實的「分不開」。
2. 每張監督式圖**強制**帶一個 badge：「supervised projection — 視覺分離不是證據，
   請看 permutation p = …」。
3. 監督式方法**預設關閉**，且要使用者勾一次「我了解監督式投影會依建構分開群體」。
4. 現有的 `supervised` / `relevance` SOM 也套用同一規則（README 目前推薦它處理
   「差異細微」的情況，必須補上這個守則）。

### 2.2 統計層（讓結論可發表）

由上而下三層，**由整體到個別**，這個順序本身就是審稿人期待的敘事：

**第 1 層：多變量整體檢定 —「群體到底有沒有差」**

| 檢定 | 回答什麼 | 為什麼需要 |
|---|---|---|
| **PERMANOVA** (Anderson 2001) | 群心是否不同；pseudo-F + permutation p + **R² 當效果量** | 距離基礎的 MANOVA，d > n 也能用；無常態假設 |
| **PERMDISP** (Anderson 2006) | 群內**離散度**是否不同 | **關鍵**：顯著的 PERMANOVA 可能純粹來自離散度差異。文獻一致建議兩者並報 |
| **Energy distance / MMD k-sample** | 分布整體（位置+尺度+形狀）是否不同 | 對「平均一樣但變異不同」敏感，補 PERMANOVA 的盲點 |
| **Mahalanobis D²**（收縮共變異） | 兩群中心距離的效果量 | 提供「差多少」而不只是「有沒有差」 |

> **這一層是整份規劃書裡科學價值最高的一塊。**
> 「位置不同」vs「離散度不同」是完全不同的生物學結論：
> 前者是「處理讓魚游得更快」，後者是「處理讓個體差異變大」。
> 目前的系統無法區分這兩者。

**第 2 層：交叉驗證分類 + 置換檢定 —「能不能被分開」的直接操作化**

- 模型：LDA(shrinkage)、linear SVM、PLS-DA、RF（沿用現有）
- 指標：**balanced accuracy** 與 **Cohen's κ**（不是 raw accuracy — 組別可能不平衡）、
  逐對 AUC、**混淆矩陣**（哪兩組分得開比整體數字有用得多）
- **置換檢定**：打亂標籤 ≥999 次、每次重跑完整 CV，
  p = (1 + #{perm ≥ obs}) / (1 + B)
- ⚠️ 不可用二項檢定評估 CV accuracy（Noirhomme et al. 2014,
  *Biased binomial assessment of cross-validated classification accuracies*）

**第 3 層：per-feature（現有的，升級）**

- 保留 ANOVA/Kruskal + η²/ε² + BH-q
- 加 **Hedges' g + bootstrap CI**（estimation statistics，見 §2.3）
- 加 **Haufe 轉換**（Haufe et al., NeuroImage 2014）：把分類器的
  *extraction filter* W 轉成可解釋的 *activation pattern* `A = Σ_x · W · Σ_s⁻¹`。
  這是「哪個維度驅動分離」最站得住腳的答案 —— 直接讀 SVM 權重會把
  **suppressor variable** 誤判為重要維度。
- 相關維度的**群組化置換重要度**：你的 metric 集合高度共線
  （turning rate / meander / straightness 測的是同一件事），
  逐一置換會低估整組的重要性。現有的 dendrogram-ordered 相關矩陣已經給出分群，
  拿來當置換的單位即可。

**❗ replicate / pseudoreplication — 必修**

`PreparedData.replicates` 已經存在但完全沒進統計。處理方式：

```python
StatsConfig.unit_of_analysis: "sample" | "replicate"
StatsConfig.block_by_replicate: bool = True
```

- **置換**：標籤在 **replicate 層級**打亂（restricted permutation / strata），不是 sample 層級
- **CV**：`GroupKFold` / leave-one-replicate-out — 否則同一 dish 的個體同時出現在
  train 與 test，CV accuracy 學到的是「哪個 dish」而不是「哪個處理」
- 保守 fallback：先在 replicate 內取平均再檢定（pseudoreplication 文獻的標準建議之一）

### 2.3 視覺化

| 技術 | 用在哪 | 出處 |
|---|---|---|
| **SuperPlots** | 依 replicate 上色/改形狀，疊上 replicate 平均的大點 | Lord et al., *J Cell Biol* 2020 — 細胞/個體生物學現在是**審稿期待** |
| **Estimation plots**（Gardner-Altman / Cumming） | 原始資料 + 效果量的 bootstrap CI，取代裸 p 值 | Ho et al., *Nat Methods* 2019（DABEST） |
| **Raincloud plots** | 分布 + 箱型 + 原始點三合一 | Allen et al. 2019/2021 |
| **Small multiples** | **比較演算法與超參數的主要載體** | Tufte；也是 parameter scan 圖的形式 |
| **Reliability overlay** | 每個點的可信度著色、不可信區域標示、Shepard diagram | CHI 2025 *Unveiling High-dimensional Backstage* 調查（133 篇） |
| **RNX(K) 曲線 + AUC_lnK** | 跨方法可比的降維品質單一曲線，涵蓋 trustworthiness/continuity | Lee & Verleysen；co-ranking matrix |
| **scDEED 每點可信度** | 以置換資料建虛無分布，標出 dubious 的點；**並用「dubious 點數最少」當超參數選擇準則** | Xia et al., *Nat Commun* 15:1753 (2024) |
| **Procrustes 種子穩定度** | 多 seed 疊合，畫每點的離散橢圓 | consensus DR 文獻 |

**scDEED 這條特別重要**：它把「perplexity / n_neighbors 要選多少」
從品味問題變成**有圖有準則的決定**，正好是你問題 2 的「呈現超參數選擇的依據」。

### 2.4 配色

| 類型 | 建議 | 理由 |
|---|---|---|
| 類別 ≤ 8 | **Okabe-Ito**（現有） | CVD 安全的事實標準（Wong, *Nat Methods* 2011） |
| 類別 ≤ 10 | **Paul Tol muted / bright** | 設計上即 CVD 安全 |
| 類別 > 10 | **glasbey**（含 CVD 約束模式） | 演算法生成最大區辨色；**絕不用連續色階取樣** |
| 類別 > 8（實務） | **不要用顏色編碼** → 改用 facet（small multiples）或 highlight+grey | 這是「上千種顏色」的正解 |
| 連續 | viridis / **cividis** / Crameri `batlow` | 感知均勻；cividis 對 deuteranopia 與灰階列印都安全 |
| 發散 | Crameri `vik` / `berlin`（或保留 RdBu/PuOr） | **強制對稱於 0**，且只在 0 有意義時使用 |
| 全域 | 匯出 **CVD 模擬 + 灰階校驗** 頁 | 投稿前自我檢查 |

依據：Crameri, Shephard & Heron, *The misuse of colour in science communication*,
**Nat Commun** 11:5444 (2020)；Crameri, *Current Protocols* (2024) 色盤選擇指南。

---

## 3. 四大主題規劃

### 3.1 主題一：選擇、比較多種演算法與 parameter scan

#### (A) 方法註冊表 — 一次解決 O(5) 新增成本

```python
# somtrack/analysis/registry.py
@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: Literal["int", "float", "choice", "bool"]
    default: Any
    low: float | None = None
    high: float | None = None
    choices: tuple = ()
    tier: Literal["common", "advanced"] = "advanced"   # ← 驅動漸進揭露
    label: str = ""          # UI 標籤，白話
    help: str = ""           # 一句話 tooltip，不用術語
    suggest: Callable[[DataShape], Any] | None = None  # 由 n/d/k 推導的預設
    scan_default: tuple = () # 「掃描這個參數」時的預設格點

@dataclass(frozen=True)
class MethodSpec:
    key: str                 # "pacmap"
    label: str               # "PaCMAP"
    family: Literal["linear", "manifold", "supervised", "som"]
    supervised: bool
    needs_groups: bool
    params: tuple[ParamSpec, ...]
    run: Callable[[np.ndarray, DataContext, dict], ProjectionResult]
    available: Callable[[], bool]   # 沿用現有 umap_available() 的守門模式
    citation: str
    caveat: str = ""         # supervised ⇒ 必填，會印在每張圖上
```

效果：
- **新增方法 = 一個檔案 + 一行 `register()`，UI 零修改**（表單由 `ParamSpec` 生成）
- `run_embeddings()` 變成對註冊表的一個迴圈
- `plot_embedding_row()` 的順序改由 `family` 決定，不再寫死 tuple

`EmbeddingConfig` 改為：
```python
enabled: list[str] = ["pca", "tsne", "umap"]      # 取代三個布林
overrides: dict[str, dict] = {}                    # method key → 參數覆寫
```
**向後相容**：`from_json` 偵測舊的 `run_pca/run_tsne/run_umap` 並轉譯，
確保既有的 `analysis_config.json` 仍能復現（README 有承諾這點）。

#### (B) 統一結果物件

`ProjectionResult` 是現有 `EmbeddingResult` 的**超集**（舊圖不用改）：

```python
# 沿用
coords, feature_axes, feature_names, axis_labels
explained_variance, trustworthiness, continuity, axes_are_loadings
# 新增
rnx: np.ndarray | None          # RNX(K) 曲線
rnx_auc: float                  # 跨方法可比的單一品質數字
reliability: np.ndarray | None  # 每點可信度（scDEED 式）
n_dubious: int                  # 不可信點數 ← 超參數選擇的目標函數
stability: np.ndarray | None    # 跨 seed 的每點離散度（Procrustes 對齊後）
oof_coords: np.ndarray | None   # 監督式方法的 out-of-fold 座標
cv: ClassifyResult | None       # 監督式方法的 CV + permutation p
method_key, params, supervised, caveat, citation   # provenance
```

#### (C) 通用 parameter scan

把 `som.scan_parameters()` 抽成 `analysis/scan.py`，對**任何**註冊方法通用：

```python
@dataclass
class ScanResult:
    method: str
    grid: dict[str, list]
    rows: pd.DataFrame                 # 每格一列：參數 + 所有品質指標
    coords: dict[tuple, np.ndarray]    # 保留座標（n<1000，記憶體無壓力）
    objective: str                     # 哪一欄決定「最佳」
    best: dict
    plateau: list[dict]                # 與最佳無實質差異的格點 ← 關鍵
```

- 目標函數只是一個欄名 ⇒ SOM 用 QE/TE/purity/NMI，投影用 RNX-AUC /
  n_dubious / CV balanced accuracy，**同一套程式**
- 格點預設來自 `ParamSpec.scan_default` ⇒ UI 上只是參數旁邊一個
  「掃描這個」勾選框，不是獨立頁面
- **`plateau` 是誠實性設計**：明確標出「這個範圍內結果一樣」，
  避免使用者過度調參、也避免挑一個好看的值

兩張圖：
1. **scan surface** — 二參數用 heatmap、單參數用帶 CI 的折線；
   標出選定格點**與整個 plateau**
2. **scan thumbnails** — 每個格點的實際 embedding 縮圖矩陣，
   依品質 badge 著邊框 ⇒ 使用者**看得到**結構在 plateau 上是穩定的

#### (D) 比較的三個層次

| 層次 | 圖 | 回答 |
|---|---|---|
| 方法 × 方法 | small multiples（列=方法），共用 legend，每格帶 RNX-AUC / n_dubious badge | 「哪個方法看到的結構是真的」 |
| 參數 × 參數 | scan surface + thumbnails | 「參數選這個的依據是什麼」 |
| seed × seed | Procrustes 疊合 + 每點離散橢圓 | 「這個結構穩不穩」 |

**共識原則**：三個以上方法都看到的結構才叫 robust；
只有一個方法看到的是那個演算法的性質 —— 這句話 README 已經寫了，
現在要讓它**可量化**（跨方法的鄰域一致性矩陣）而不只是文字。

---

### 3.2 主題二：科研與學術發表的視覺化 + 正確的生物統計

#### (A) 統計層新模組

```
somtrack/stats/
  omnibus.py    PERMANOVA / PERMDISP / energy / MMD / Mahalanobis D²
  classify.py   CV 協定、permutation null、混淆矩陣、逐對 AUC
  blocks.py     replicate-aware 切分器 + 受限置換
  effects.py    Hedges' g + bootstrap CI（estimation statistics）
  interpret.py  Haufe activation pattern、PLS-DA VIP、群組化置換重要度
  verdict.py    白話結論生成器
```

兩個核心結果物件，直接掛進 `AnalysisResult`，
並加進 `result.tables()` ⇒ **CSV / XLSX 匯出零額外程式**：

```python
@dataclass
class SeparationReport:
    permanova_F, permanova_R2, permanova_p
    permdisp_F, permdisp_p
    energy_stat, energy_p
    pairwise: pd.DataFrame        # 逐對：R², p, q(BH), Mahalanobis D + CI
    n_permutations: int
    blocked_by: str | None        # provenance，寫進 methods.txt

@dataclass
class ClassifyResult:
    model: str
    cv_scheme: str                # "leave_one_replicate_out" 等
    balanced_accuracy: float; ba_ci: tuple
    kappa: float
    pairwise_auc: pd.DataFrame
    confusion: np.ndarray
    permutation_p: float; n_permutations: int
    baseline: float
    weights: np.ndarray           # 原始 filter
    activation: np.ndarray        # Haufe 轉換後 ← 可解釋的那個
```

**依賴策略**：PERMANOVA / PERMDISP / energy test 在 numpy 裡各約 30–50 行
（Gower 中心化距離矩陣 + trace 比值），**不需要新套件**。
`hyppo` 可以當 optional 的交叉驗證用。

**效能核算**（n ≤ 1000, d ≤ 50）：
- PERMANOVA 9999 次置換、1000×1000 距離矩陣 → 數秒（純 numpy）
- 置換式分類檢定是瓶頸：999 × 5-fold × LDA/linear-SVM → 可接受（線性模型在 50 維是微秒級）
- **RF 不進置換迴圈**（太慢），RF 只跑一次做 importance；
  置換 null 預設用 LDA/SVM，或降到 199 次並在圖上註明

#### (B) 新圖（接在現有 01–22 之後）

| 編號 | 圖 | 回答的問題 |
|---|---|---|
| 23 | **Verdict card**（PDF 第一頁的文字卡） | 「所以結論是什麼」 |
| 24 | **Separation summary** — R² 的 forest plot + permutation CI；逐對 q 值三角熱圖 | 「哪兩組有差、差多少」 |
| 25 | **Location vs dispersion** — PERMANOVA 與 PERMDISP 並列 | 「是位置不同還是變異度不同」 |
| 26 | **Classifiability** — 觀測 balanced accuracy vs 置換虛無分布直方圖 + 混淆矩陣 | 「能不能被分開，p 是多少」 |
| 27 | **In-sample vs out-of-fold** 監督式投影並列 | 「這個分離是真的還是建構出來的」 ⭐ |
| 28 | **Haufe activation vs raw weights** 並列長條 | 「哪個維度真的驅動分離」 |
| 29 | **Method comparison grid**（small multiples） | 「結構在不同方法間一致嗎」 |
| 30 | **Scan surface + plateau** | 「超參數為什麼選這個」 |
| 31 | **Scan thumbnails** | 同上，視覺佐證 |
| 32 | **Reliability panel** — 每點可信/存疑 + 全方法 RNX 曲線同軸 + Shepard | 「這張圖哪裡不能信」 |
| 33 | **Seed stability** — Procrustes 疊合 + 離散橢圓 | 「換個 seed 還長這樣嗎」 |
| 34 | **SuperPlot / estimation panel**（top-k 維度，replicate 感知） | 「個別維度的差異與效果量 CI」 ⭐ |
| 35 | **Palette proof sheet**（選配匯出） | 投稿前 CVD/灰階自我檢查 |

⭐ 是對你的情境價值最高的三張。

#### (C) Verdict card — 讓非統計背景讀得懂

由統計層自動生成，不是手寫。目標語氣：

> **A 組與 B 組可以分開**（balanced accuracy 0.81 [95% CI 0.72–0.88]，
> permutation p = 0.002，leave-one-replicate-out CV，999 次置換）。
> **C 組與 D 組分不開**（0.54，p = 0.31）。
> 整體 PERMANOVA R² = 0.18, p = 0.001；但 **PERMDISP p = 0.004**，
> 表示部分分離來自**群內變異度差異**而非群心位移 ——
> 請解讀為「C 組個體間差異較大」，而不是「C 組游得比較快」。
> 驅動分離的前三個維度（Haufe activation）：mean speed、turn entropy、burstiness B。

這一段同時寫進 `methods.txt` 與 Results 頁的最上方。
**這是整個工具對非統計使用者最大的價值。**

#### (D) 現有統計的升級

- `multivariate_importance()` 加 `permutation_p` 與 `GroupKFold` 選項
- `univariate_association()` 加 Hedges' g + bootstrap CI 欄位
- `group_enrichment()` 的 hypergeometric 在 replicate 結構下要改用受限置換（或註明限制）
- `methods.txt` 納入所有新檢定的參數與置換次數（審稿必問）

---

### 3.3 主題三：UIUX — 讓非資訊/統計背景的人願意用

#### (A) 三層漸進揭露

```
Level 1  Recipe（預設）    1 個下拉 + 1 句白話說明 + [開始分析]
Level 2  Common            5–8 個控制項（ParamSpec.tier == "common"）
Level 3  Advanced          全部（預設收合）
```

**Recipe 是資料不是程式** — 一份 JSON/TOML preset 直接設定整棵 `AnalysisConfig`，
複用現有的 `AnalysisConfig.from_json`，**零新持久化程式碼**：

| Recipe | 做什麼 |
|---|---|
| **標準探索** | batch SOM + PCA/t-SNE/UMAP/PaCMAP + 整體檢定；不碰監督式方法 |
| **檢定我的分組** ⭐ | PCoA + PERMANOVA/PERMDISP/energy + LOO-replicate CV 分類 + permutation p + Verdict |
| **找出關鍵維度** | RF + LDA/SVM + Haufe + 群組化重要度 + SuperPlot |
| **穩健性檢查** | 多方法 × 多 seed × parameter scan，只出比較圖 |
| **重現 v1.2** | 現有的 legacy preset |

#### (B) Results-first，而不是 wizard-first

現在是「走完 6 頁才看到第一張圖」。改成：
1. 載入資料後**立刻**用安全預設跑一次，先給結果
2. 使用者從**想改的那張圖**點進設定（"edit here" affordance），而不是回頭翻設定頁
3. `TaskRunner` 已經會發 progress signal ⇒ 擴充成**逐張串流 panel**，
   不必等全部跑完（`ui/workers.py` 只要多一個 signal）

#### (C) 具體的 UI 修正

| 現況 | 改成 |
|---|---|
| `_algo_changed()` 把不適用的參數**變灰** | **隱藏**。看不到的東西不造成認知負擔 |
| 21 個控制項同時可見 | Level 1 只有 1 個；Level 2 依所選方法動態生成 |
| 參數自動值（perplexity=0 → auto）藏在 tooltip | 「自動」勾選框 + **顯示算出來的值** + 一句「為什麼」 |
| 監督式 SOM 與非監督式並列在同一個下拉 | 分區，監督式區塊帶警告圖示 + 一次性確認勾選 |
| 無比較介面 | 新增 **Compare 頁**（Results 與 Export 之間）：small multiples + scan surface + 可排序品質表 + 點選「採用這組設定」 |
| 統計散在 Results 的表裡 | 新增 **Statistics 頁**：Verdict card 在最上，其下是分離報告、分類結果、estimation plots |

#### (D) 新的 config 區塊

```python
@dataclass
class StatsConfig:
    unit_of_analysis: Literal["sample", "replicate"] = "sample"
    block_by_replicate: bool = True
    n_permutations: int = 999
    cv_scheme: Literal["stratified", "leave_one_replicate_out",
                       "repeated_stratified"] = "stratified"
    cv_repeats: int = 5
    classifiers: list[str] = field(default_factory=lambda: ["lda_shrinkage", "svm_linear"])
    run_permanova: bool = True
    run_permdisp: bool = True
    run_energy: bool = True
    alpha: float = 0.05
    bootstrap: int = 5000

@dataclass
class ReportConfig:
    profile: Literal["core", "full"] = "core"   # core ≈ 12 張；full = 全部
    recipe: str = "standard"
```

> **報告長度控制**：現在 22 張，加完會到 35 張 —— 一份 35 頁的 PDF 沒人看。
> 用 `ReportConfig.profile` 綁定 recipe，預設只出 core 的 10–12 張。

---

### 3.4 主題四：配色策略

#### (A) 色盤政策引擎 `viz/palette.py`

不是再加一個色盤常數，而是一個**決策函式**：

```python
def categorical(n: int, cfg: FigureConfig, *, focus: int | None = None) -> PaletteDecision:
    """
    n <= 8   → Okabe-Ito
    n <= 10  → Tol muted
    n <= cfg.categorical_max → glasbey(cvd_safe=True)
    n >  cfg.categorical_max → PaletteDecision(mode="facet")      # 建議改用 small multiples
    focus is not None        → PaletteDecision(mode="highlight")  # 焦點組上色、其餘灰
    """
```

`PaletteDecision` 除了顏色，還回傳 **mode**，讓繪圖函式知道
「顏色已經不夠用了，請改用分面或 highlight」。
**這是「上千種顏色」問題的結構性解法** —— 把它變成程式的決定，不是使用者的失誤。

#### (B) 硬性規則（寫進 `viz/palette.py` 並在 CI 中檢查）

1. **類別絕不使用連續色階取樣** → 移除 `style.py:54` 與 `vectors.py:93, 244` 的 `turbo`
2. **發散色階只在 0 有意義時使用，且強制對稱** → 現有 `vmin=-lim, vmax=lim` 的做法升格為
   `symmetric_norm()` 輔助函式，所有發散圖必須經過它
3. **順序色階一律感知均勻**：viridis / cividis / `batlow`（移除 `bone_r`）
4. **≥ 2 重編碼**：顏色 + marker 形狀；關鍵圖用**直接標註**取代 legend
5. **node 色 = legend 色**：`palette_hues()` 必須從實際色盤取 hue，
   在任何 n 下都成立（修掉 turbo fallback 造成的不一致）
6. **灰色保留給 context**（highlight 模式的背景組、不顯著的項目），不分配給任何實驗組

#### (C) 多維度分析特有的配色考量

| 情境 | 策略 |
|---|---|
| 實驗組 ≤ 8 | Okabe-Ito，全系統一致（圖、SOM node、legend、報告表格底色） |
| 實驗組 > 8 | facet：每組一個小圖，其餘組畫成灰點當背景 |
| metric 箭頭 / gradient compass（常 > 20 項） | **不要上色**。改用：粗細 = 強度、透明度 = 擬合 R²、只標註 top-k 文字 |
| node cluster（k 通常 2–8） | Okabe-Ito 的子集，且**與實驗組色系刻意不同**（例如實驗組用飽和色、node cluster 用同色系的淡色填充 + 深色外框），避免兩種類別在同一張圖上撞色 |
| 顯著性 | **不要用紅綠**。用「有無外框 / 實心空心 / 星號」表示，顏色留給分組 |
| 連續 metric 疊在 SOM 上 | cividis（列印與 CVD 雙安全） |
| 富集 log2(obs/exp) | `vik`（Crameri）或保留 `RdBu_r`，強制對稱 |

#### (D) 校驗機制

- `FigureConfig` 加 `cvd_proof: bool` → 匯出時額外產生
  deuteranopia / protanopia / tritanopia / 灰階四種模擬版本
- 色盤政策引擎在選色時即檢查最小感知色差（CIEDE2000），不足時降級到 facet 模式
- Export 頁一個按鈕：「產生投稿前色彩校驗頁」

---

## 4. 實作藍圖

### 4.1 新的模組配置

```
somtrack/
  analysis/
    registry.py       MethodSpec / ParamSpec / register() / all_methods()
    projection.py     ProjectionResult（EmbeddingResult 的超集）
    methods/
      linear.py       PCA, MDS/PCoA, LDA-shrinkage, PLS-DA
      manifold.py     t-SNE, UMAP, densMAP, PaCMAP, PHATE, Isomap, KernelPCA
      supervised.py   linear SVM, supervised UMAP, SLISEMAP
      som_adapter.py  把現有 som.py 包成一個註冊方法
    quality.py        co-ranking / RNX / scDEED reliability / Procrustes stability
    scan.py           通用 ParameterScan（取代 som.scan_parameters）
  stats/
    omnibus.py  classify.py  blocks.py  effects.py  interpret.py  verdict.py
  viz/
    palette.py      色盤政策引擎
    compare.py      small multiples / scan surface / stability
    estimation.py   SuperPlot / Gardner-Altman / raincloud
    reliability.py  每點可信度 / RNX 曲線 / Shepard / distortion overlay
```

`config.py`、`preprocess.py`、`pipeline.py`、`viz/style.py`、`export.py`
維持為主幹，新東西**掛上去**而不是取代。

### 4.2 分期

| 階段 | 內容 | 新依賴 | 使用者可見變化 |
|---|---|---|---|
| **P0 地基** | registry + `ProjectionResult` + config 向後相容。把現有 PCA/t-SNE/UMAP 包成註冊項，`run_embeddings()` 改成迴圈 | 無 | **無**（以輸出逐位元相同來驗證重構正確） |
| **P1 統計** ⭐ | PERMANOVA / PERMDISP / energy / Mahalanobis；replicate-aware CV + permutation；Verdict card；接進 `methods.txt` | 無 | 新增 Statistics 頁與圖 23–26 |
| **P2 方法** | PaCMAP, MDS/PCoA, densMAP, LDA-shrinkage, PLS-DA, linear SVM；監督式守則 + out-of-fold 圖；Haufe | `pacmap` | 方法選單變長但由註冊表生成；圖 27–28 |
| **P3 比較** | 通用 scan + quality（RNX / scDEED / seed stability）+ Compare 頁 | 無 | 圖 29–33 |
| **P4 視覺/UX** | palette 引擎、SuperPlot/estimation、recipes、漸進揭露、results-first、CVD proof | `glasbey`, `cmcrameri` | UI 大改版；圖 34–35 |
| **P5 進階** | PHATE, SLISEMAP, Isomap/KPCA；PCC（待套件成熟） | `phate`, `slisemap` | Advanced 區 |

所有新依賴沿用現有的 `umap_available()` 守門模式
（註冊表的 `available()` callback 統一處理），**沒裝也不會壞**。

### 4.3 風險與對策

| 風險 | 對策 |
|---|---|
| n < 1000 ⇒ 每個 CV 估計都很吵 | 一律報 CI 不報點估計；巢狀 CV；圖上標 n |
| 方法太多 ⇒ 誘發 cherry-picking「挑最漂亮那張」 | Compare 頁**同時**顯示所有跑過的方法；報告記錄全部；**結論由整體檢定決定，不由最好看的圖決定**（寫成明文設計原則） |
| 35 張圖的 PDF 沒人看 | `ReportConfig.profile`，預設 core 10–12 張 |
| 置換檢定太慢 | 線性模型進置換迴圈，RF 不進；joblib 平行（骨架已有）；UI 顯示預估時間 |
| 監督式方法被誤用 | 預設關閉 + 一次性確認 + 強制 badge + 強制 out-of-fold 並列圖 |
| 重構破壞既有復現性 | P0 以「輸出逐位元相同」為驗收；舊 `analysis_config.json` 必須能載入（加回歸測試） |

### 4.4 建議的第一步

**P0 + P1 的前半**，因為：
- P0 不改變任何輸出，風險最低，但解鎖後面所有工作
- **PERMANOVA + PERMDISP + replicate-aware permutation 是純 numpy、約 150 行、
  零新依賴，但直接把「我看圖覺得有分開」升級成「pseudo-F = 4.2, R² = 0.18,
  p = 0.001，且 PERMDISP p = 0.31 表示這是群心位移不是變異度差異」** ——
  這句話就是可以寫進論文的那句話。

---

## 參考文獻

**降維方法**
- Wang, Huang, Rudin & Shaposhnik (2021). *Understanding how dimension reduction tools work: t-SNE, UMAP, TriMAP, and PaCMAP.* JMLR 22(201).
- [Benchmarking of dimensionality reduction methods to capture drug response in transcriptome data](https://www.nature.com/articles/s41598-025-12021-7). *Sci Rep* (2025).
- [Preserving clusters and correlations (PCC)](https://arxiv.org/abs/2503.07609). arXiv:2503.07609 (2025).
- [SLISEMAP: supervised dimensionality reduction through local explanations](https://arxiv.org/abs/2201.04455). *Machine Learning* (2022). [程式碼](https://github.com/edahelsinki/slisemap)
- [PaCMAP](https://github.com/YingfanWang/PaCMAP) · [PHATE](https://pypi.org/project/phate/)

**降維可信度與超參數選擇**
- [scDEED: 偵測可疑的 2D embedding 並最佳化 t-SNE/UMAP 超參數](https://www.nature.com/articles/s41467-024-45891-y). *Nat Commun* 15:1753 (2024).
- [Unveiling High-dimensional Backstage: A Survey for Reliable Visual Analytics with Dimensionality Reduction](https://dl.acm.org/doi/10.1145/3706598.3713551). CHI 2025.
- [pyDRMetrics — 降維品質評估工具箱](https://pmc.ncbi.nlm.nih.gov/articles/PMC7887408/) (co-ranking, RNX, LCMC).
- [HyperNP: Interactive Visual Exploration of Multidimensional Projection Hyperparameters](https://arxiv.org/pdf/2106.13777).
- [UMAP supervised mode 的過擬合討論](https://github.com/lmcinnes/umap/issues/1116)

**統計**
- Anderson (2001) PERMANOVA；Anderson (2006) PERMDISP。
  [PERMDISP 實務指引](https://uw.pressbooks.pub/appliedmultivariatestatistics/chapter/permdisp/)
- [Cross-validated Between Group PCA Scatterplots: A Solution to Spurious Group Separation?](https://link.springer.com/article/10.1007/s11692-020-09494-x) *Evol Biol* (2020).
- [Assessment of PLSDA cross validation](https://link.springer.com/article/10.1007/s11306-007-0099-6). Westerhuis et al., *Metabolomics* (2008).
- [Biased binomial assessment of cross-validated classification accuracies](https://pmc.ncbi.nlm.nih.gov/articles/PMC4053638/). *NeuroImage: Clinical* (2014).
- [On the interpretation of weight vectors of linear models in multivariate neuroimaging](https://www.sciencedirect.com/science/article/pii/S1053811913010914). Haufe et al., *NeuroImage* 87:96-110 (2014).
- [Computationally efficient permutation tests based on energy distance or MMD](https://arxiv.org/html/2406.06488) · [hyppo](https://github.com/neurodata/hyppo)
- [A practical solution to pseudoreplication bias](https://www.nature.com/articles/s41467-021-21038-1). *Nat Commun* (2021).

**視覺化與配色**
- [The misuse of colour in science communication](https://www.nature.com/articles/s41467-020-19160-7). Crameri, Shephard & Heron, *Nat Commun* 11:5444 (2020).
- [Choosing Suitable Color Palettes for Accessible and Accurate Science Figures](https://currentprotocols.onlinelibrary.wiley.com/doi/10.1002/cpz1.1126). Crameri, *Current Protocols* (2024). · [Scientific colour maps](https://www.fabiocrameri.ch/colourmaps/)
- Lord, Velle, Mullins & Fritz-Laylin (2020). *SuperPlots: Communicating reproducibility and variability in cell biology.* J Cell Biol 219(6).
- Ho, Tumkaya, Aryal, Choi & Claridge-Chang (2019). *Moving beyond P values: data analysis with estimation graphics.* Nat Methods 16:565-566. [DABEST](https://acclab.github.io/dabestr/)
- Wong (2011). *Points of view: Color blindness.* Nat Methods 8:441（Okabe-Ito）· [glasbey](https://glasbey.readthedocs.io/)

**UX**
- [Progressive Disclosure (IxDF)](https://ixdf.org/literature/topics/progressive-disclosure)
- [Designing Scaffolded Interfaces for Enhanced Learning and Performance in Professional Software](https://arxiv.org/pdf/2505.12101)
