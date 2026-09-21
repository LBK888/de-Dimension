"""Traditional Chinese (Taiwan) -- 繁體中文（台灣）.

Terminology follows Taiwanese academic usage: 樣本 (sample), 組別 (group),
重複 (replicate), 指標 (metric), 投影 (projection), 置換檢定 (permutation
test), 交叉驗證 (cross-validation), 平衡準確率 (balanced accuracy), 信賴區間
(confidence interval), 效應量 (effect size), 離散程度 (spread / dispersion).
Names of tests and algorithms (PERMANOVA, PCA, t-SNE, SOM) are kept as they are
cited in the literature.  Placeholders in braces must survive translation
unchanged.
"""

MESSAGES: dict[str, str] = {
    # ======================================================================
    # Main window
    # ======================================================================
    "SOMTrack 3.0 - locomotion metrics, multivariate analysis and publication figures":
        "SOMTrack 3.0 － 運動指標、多變量分析與出版級圖表",
    "locomotion SOM": "運動行為 SOM",
    "1  Data source": "1  資料來源",
    "2  Metrics": "2  指標",
    "3  Feature matrix": "3  特徵矩陣",
    "4  Analysis": "4  分析",
    "5  Results": "5  結果",
    "6  Compare": "6  比較",
    "7  Export": "7  匯出",
    "Log": "紀錄",
    "Ready": "就緒",
    "Back": "上一步",
    "Next": "下一步",
    "Finish": "完成",
    "&File": "檔案(&F)",
    "Save settings...": "儲存設定...",
    "Load settings...": "載入設定...",
    "Quit": "結束",
    "&Help": "說明(&H)",
    "About SOMTrack": "關於 SOMTrack",
    "Something went wrong": "發生錯誤",
    "One more thing": "還差一步",
    "Loaded {n} samples, {m} features.": "已載入 {n} 個樣本、{m} 個特徵。",
    "Pick some metrics": "請選擇指標",
    "Select at least two metrics.": "請至少選擇兩個指標。",
    "Computing metrics...": "正在計算指標...",
    "Computing {n} metrics for {t} tracks...": "正在為 {t} 條軌跡計算 {n} 個指標...",
    "{n} tracks x {m} metrics": "{n} 條軌跡 × {m} 個指標",
    " ({n} track(s) dropped as too short)": "（{n} 條軌跡因過短而略過）",
    "Running analysis...": "正在執行分析...",
    "SOM: {algorithm}, {epochs} epochs; {n} features.":
        "SOM：{algorithm}，{epochs} 個訓練週期；{n} 個特徵。",
    "Done. QE={qe:.3f}  TE={te:.3f}  purity={purity:.3f}  occupancy={occupancy:.0%}":
        "完成。量化誤差 QE={qe:.3f}  拓撲誤差 TE={te:.3f}  純度={purity:.3f}  "
        "節點佔用率={occupancy:.0%}",
    "Run the analysis first": "請先執行分析",
    "A scan re-uses the feature matrix the analysis prepared, so there has to be one. "
    "Go back to Analysis and press Run.":
        "參數掃描會沿用分析時準備好的特徵矩陣，因此必須先有一次分析結果。"
        "請回到「分析」步驟並按下執行。",
    "Cannot scan that": "無法掃描此方法",
    "Scanning {n} settings...": "正在掃描 {n} 組設定...",
    "Scanning {method} over {grid}": "正在掃描 {method}：{grid}",
    "Scanning ({i}/{n})": "掃描中（{i}/{n}）",
    "Nothing to export": "沒有可匯出的內容",
    "Run the analysis first.": "請先執行分析。",
    "Exporting to {path}...": "正在匯出至 {path}...",
    "Exporting to {path}": "匯出至 {path}",
    "Exported {n} figures, {t} tables": "已匯出 {n} 張圖、{t} 個表格",
    ", {n} video": "、{n} 段影片",
    " to {path}": "，位置：{path}",
    "Export complete": "匯出完成",
    "Wrote {n} figures and {t} tables to\n{path}\n\n"
    "SOMTrack_report.pdf holds every figure with selectable text; methods.txt is a "
    "ready-to-paste methods paragraph. RESULTS_REPORT.html gives the report in "
    "English followed by its Chinese translation.":
        "已將 {n} 張圖與 {t} 個表格寫入\n{path}\n\n"
        "SOMTrack_report.pdf 收錄所有圖表（文字可選取）；methods.txt 是可直接貼上的"
        "方法段落。RESULTS_REPORT.html 先列出英文報告，其後附上完整的中文翻譯。",
    "Save settings": "儲存設定",
    "Settings saved to {path}": "設定已儲存至 {path}",
    "Load settings": "載入設定",
    "Could not load settings": "無法載入設定",
    "Settings loaded from {path}. Re-visit the steps to apply them.":
        "已從 {path} 載入設定。請重新瀏覽各步驟以套用。",
    "Change language": "變更語言",
    "SOMTrack will use the new language the next time it starts. Restart now? "
    "Anything not saved will be lost.":
        "SOMTrack 會在下次啟動時使用新的語言。要現在重新啟動嗎？尚未儲存的內容將會遺失。",
    "A task is still running. Restart SOMTrack when it has finished.":
        "仍有工作正在執行。請在完成後再重新啟動 SOMTrack。",
    "Please close and reopen SOMTrack.": "請關閉後重新開啟 SOMTrack。",
    "A Python refactor of <i>SOM tracking_v1.2b.ijm</i> (Wang, Ho &amp; Liao, NTOU "
    "2020).<br><br>Computes locomotion and biophysical metrics from tracking "
    "coordinates, clusters them with self-organising maps and a registry of "
    "projections, and tests whether the groups really differ -- PERMANOVA, "
    "PERMDISP, the energy test and cross-validated classification against a "
    "permutation null -- before it lets a figure claim that they do.<br><br>"
    "Every run writes a conclusion report with a methods paragraph and a reference "
    "list. Figures export as editable-text PDF/SVG, 600 dpi PNG and MP4.":
        "<i>SOM tracking_v1.2b.ijm</i>（Wang、Ho &amp; Liao，國立臺灣海洋大學，2020）"
        "的 Python 重構版本。<br><br>"
        "由追蹤座標計算運動與生物物理指標，以自組織映射圖（SOM）及一系列投影方法進行"
        "分群與視覺化，並在讓圖表宣稱組別有差異之前，先以 PERMANOVA、PERMDISP、"
        "能量檢定，以及對照置換虛無分布的交叉驗證分類，檢驗組別是否真的不同。<br><br>"
        "每次執行都會產生一份結論報告，附有方法段落與參考文獻清單。圖表可匯出為"
        "文字可編輯的 PDF/SVG、600 dpi PNG 與 MP4。",
    "A task is running": "工作執行中",
    "Quit anyway and abandon the running task?": "仍要結束並放棄正在執行的工作嗎？",

    # ======================================================================
    # Shared widgets
    # ======================================================================
    "(none)": "（無）",
    "Add files...": "加入檔案...",
    "Remove selected": "移除選取項目",
    "Clear": "清除",
    "File": "檔案",
    "Group (treatment)": "組別（處理）",
    "Replicate": "重複",
    "Group and replicate are guessed from the file name; double-click a cell to "
    "correct it. Files sharing a group are one treatment; the replicate number "
    "separates repeats of that treatment.":
        "組別與重複編號是由檔名推測而來；雙擊儲存格即可修正。組別相同的檔案屬於同一個"
        "處理；重複編號用來區分同一處理的各次重複實驗。",
    "Select coordinate tables": "選擇座標表格",
    "Tables": "表格",
    "All files": "所有檔案",
    "Column mapping": "欄位對應",
    "Track ID": "軌跡 ID",
    "X position": "X 位置",
    "Y position": "Y 位置",
    "Time / frame": "時間／影格",
    "Group column (optional)": "組別欄位（選填）",
    "Replicate column (optional)": "重複欄位（選填）",
    "Area (optional)": "面積（選填）",
    "Ellipse major (optional)": "橢圓長軸（選填）",
    "Ellipse minor (optional)": "橢圓短軸（選填）",
    "Body orientation (optional)": "身體方向（選填）",
    "Mean intensity (optional)": "平均強度（選填）",
    "Intensity SD (optional)": "強度標準差（選填）",
    "Estimated diameter (optional)": "估計直徑（選填）",
    "Metric": "指標",
    "Unit": "單位",
    "What it measures": "測量內容",
    "Figures": "圖表",
    "Advanced settings": "進階設定",
    "Auto": "自動",
    "Let SOMTrack choose this from the size of your data set. The value it picked "
    "is shown beside the box and is recorded in the methods section.":
        "由 SOMTrack 依資料集大小自動決定。所選的數值會顯示在旁邊，並記錄於方法段落中。",
    "auto: {value}": "自動：{value}",

    # ======================================================================
    # 1. Data source
    # ======================================================================
    "1. Data source": "1. 資料來源",
    "Start from spot-level coordinate tables and let SOMTrack compute the locomotion "
    "metrics, or load a multi-dimensional table you already have.":
        "可從偵測點層級的座標表格開始，由 SOMTrack 計算運動指標；"
        "或直接載入你已有的多維度表格。",
    "A - Coordinate tables (compute metrics)": "A － 座標表格（計算指標）",
    "B - Multi-dimensional table (skip to clustering)": "B － 多維度表格（直接進行分群）",
    "Load demo data": "載入示範資料",
    "Generate a synthetic four-treatment swimming assay so you can try the whole "
    "workflow.": "產生一組模擬的四處理游泳實驗資料，讓你試用完整流程。",
    "First rows of the first file": "第一個檔案的前幾列",
    "Calibration and track filtering": "校正與軌跡篩選",
    "per pixel, unit": "每像素，單位",
    "Spatial scale": "空間尺度",
    "per frame, unit": "每影格，單位",
    "Time scale": "時間尺度",
    "X/Y and T columns are already in physical units": "X/Y 與 T 欄位已是物理單位",
    "Minimum spots per track": "每條軌跡最少偵測點數",
    "Central fraction of ranked values kept when averaging (the v1.2 decile "
    "filter). 1.0 uses every value. Trimming resists tracking glitches but biases "
    "the reported SD downwards.":
        "計算平均時保留的排序中段比例（即 v1.2 的十分位過濾）。1.0 表示使用所有數值。"
        "截尾可抵抗追蹤誤差，但會使報告的標準差偏低。",
    "Robust trim fraction": "穩健截尾比例",
    "Detect the pause threshold from the speed histogram": "由速度直方圖自動判定停頓門檻",
    "v1.2 used a fixed 1.5 pixel step, which does not transfer between "
    "magnifications. This splits the bimodal speed distribution instead.":
        "v1.2 使用固定的 1.5 像素步長，無法在不同放大倍率間通用。"
        "此選項改為切分雙峰的速度分布。",
    "Savitzky-Golay window in frames; 0 disables smoothing.":
        "Savitzky-Golay 平滑視窗（影格數）；0 表示不平滑。",
    "Coordinate smoothing": "座標平滑",
    "Path to a table with one row per sample...": "每個樣本一列的表格路徑...",
    "Browse...": "瀏覽...",
    "Column roles": "欄位角色",
    "Sample ID": "樣本 ID",
    "Feature columns to use": "要使用的特徵欄位",
    "Select all": "全選",
    "Select none": "全不選",
    "Select a multi-dimensional table": "選擇多維度表格",
    "Could not read the table": "無法讀取表格",
    "Loaded {rows} rows x {cols} columns from {file}":
        "已從 {file} 載入 {rows} 列 × {cols} 欄",
    "Could not read {file}: {error}": "無法讀取 {file}：{error}",
    "Could not auto-detect: {columns} -- set them in the column mapping.":
        "無法自動偵測：{columns}——請在欄位對應中手動設定。",
    "{n} file(s); columns detected automatically.": "{n} 個檔案；已自動偵測欄位。",
    "Generating demo data...": "正在產生示範資料...",
    "Demo data failed": "示範資料產生失敗",
    "Demo data written to {path}": "示範資料已寫入 {path}",
    "Add at least one coordinate table.": "請至少加入一個座標表格。",
    "Track, X, Y and time columns must all be mapped.":
        "軌跡、X、Y 與時間欄位都必須完成對應。",
    "Choose a multi-dimensional table.": "請選擇一個多維度表格。",
    "Select at least two feature columns.": "請至少選擇兩個特徵欄位。",

    # ======================================================================
    # 2. Metrics
    # ======================================================================
    "2. Locomotion metrics": "2. 運動指標",
    "Choose what to measure on each track. Metrics marked (v1.2) existed in the "
    "ImageJ macro; the rest are new. Metrics whose input columns are missing from "
    "your table are hidden.":
        "選擇要在每條軌跡上測量的項目。標示 (v1.2) 的指標源自原 ImageJ 巨集，其餘為新增。"
        "表格中缺少所需輸入欄位的指標會被隱藏。",
    "Every metric that your columns support.": "你的欄位所能支援的所有指標。",
    "A broad, low-redundancy default set.": "涵蓋面廣、重複性低的預設組合。",
    "Exactly the metrics the ImageJ macro produced, for direct comparison.":
        "與 ImageJ 巨集產出完全相同的指標，方便直接比較。",
    "Speed, acceleration and turning only.": "僅包含速度、加速度與轉向。",
    "Clear the selection.": "清除所有選取。",
    "Compute metrics": "計算指標",
    "Computed feature table": "計算出的特徵表",
    "{n} metric(s) selected": "已選擇 {n} 個指標",
    "Press 'Compute metrics' first.": "請先按下「計算指標」。",

    # ======================================================================
    # 3. Feature matrix
    # ======================================================================
    "3. Feature matrix": "3. 特徵矩陣",
    "Pick the columns that go into the clustering and decide how they are scaled. "
    "Highly correlated metrics let one behavioural axis dominate the distance, "
    "which is a common reason a SOM fails to separate treatments.":
        "選擇要納入分群的欄位，並決定縮放方式。高度相關的指標會讓單一行為面向主導距離，"
        "這是 SOM 無法區分各處理的常見原因。",
    "Preprocessing": "前處理",
    "z-score puts every metric on equal footing (recommended).\n"
    "min-max reproduces the v1.2 normalisation but is dominated by outliers.\n"
    "robust uses median/IQR; rank is fully non-parametric.":
        "z-score：讓每個指標處於同等地位（建議）。\n"
        "min-max：重現 v1.2 的正規化方式，但容易受離群值主導。\n"
        "robust：使用中位數／四分位距；rank：完全無母數。",
    "Scaling": "縮放方式",
    "Missing values": "缺失值",
    "Drop the later of any pair of metrics more correlated than this. 1.00 disables "
    "the pruning.": "兩個指標的相關高於此值時，剔除後者。1.00 表示不剔除。",
    "Prune |r| above": "剔除 |r| 高於",
    "Clip each metric to this quantile at both tails; 0 disables clipping.":
        "將每個指標的兩端截斷至此分位數；0 表示不截斷。",
    "Winsorise tails": "尾端縮尾（winsorize）",
    "Drop constant metrics": "移除常數指標",
    "Descriptive statistics (original units)": "描述性統計（原始單位）",
    "{n} feature(s) selected": "已選擇 {n} 個特徵",
    "Select at least two features.": "請至少選擇兩個特徵。",

    # ======================================================================
    # 4. Analysis
    # ======================================================================
    "4. Analysis": "4. 分析",
    "Pick what you are trying to find out. Everything below has a sensible default; "
    "open a section only if you want to change it.":
        "選擇你想回答的問題。下方所有項目都已有合理的預設值；只有需要修改時才展開該區塊。",
    "What do you want to find out?": "你想了解什麼？",
    "Which projections to run": "要執行哪些投影",
    "These use the group labels to build the axes, so they separate the groups "
    "whatever the data says — they would separate random noise too. They are "
    "drawn out-of-fold as well, and that panel is the one to believe.":
        "這些方法利用組別標籤建立座標軸，因此無論資料如何都會把組別分開——連隨機雜訊也會"
        "被分開。它們也會以折外（out-of-fold）方式繪製，應以該圖為準。",
    "I understand supervised projections separate groups by construction":
        "我了解監督式投影在建構上就會把組別分開",
    "not installed": "未安裝",
    "How to test whether the groups differ": "如何檢定組別是否有差異",
    "Self-organising map": "自組織映射圖（SOM）",
    "Behavioural clusters on the map": "映射圖上的行為群集",
    "Run analysis": "執行分析",
    "Test whether the groups differ, and by how much": "檢定組別是否不同，以及差異多大",
    "PERMANOVA and PERMDISP on the sample distances, the energy test, and "
    "cross-validated group assignment with a permutation null.":
        "以樣本間距離進行 PERMANOVA 與 PERMDISP、能量檢定，"
        "以及對照置換虛無分布的交叉驗證組別判別。",
    "each sample is independent": "每個樣本彼此獨立",
    "average within each replicate first": "先在每個重複內取平均",
    "If several samples came from one dish, clutch or imaging session, they are not "
    "independent measurements. SOMTrack detects the replicate structure and blocks "
    "the permutations and the cross-validation folds by it either way; this setting "
    "decides whether to go further and average within each replicate, which is the "
    "most conservative option.":
        "若多個樣本來自同一培養皿、同一窩或同一次影像擷取，它們就不是獨立的測量。"
        "SOMTrack 會偵測重複結構，並據此限制置換與交叉驗證的分折；"
        "此設定決定是否更進一步先在每個重複內取平均——這是最保守的做法。",
    "Experimental unit": "實驗單位",
    "Used by PERMANOVA, PERMDISP and the MDS projection, so the picture and the "
    "test describe the same thing.":
        "PERMANOVA、PERMDISP 與 MDS 投影都使用此距離，讓圖與檢定描述的是同一件事。",
    "Distance between samples": "樣本間距離",
    "linear discriminant (regularised)": "線性判別（正則化）",
    "linear support vector machine": "線性支持向量機",
    "logistic regression": "邏輯斯迴歸",
    "random forest": "隨機森林",
    "Assignment model": "判別模型",
    "How many times the group labels are shuffled to work out what this analysis "
    "achieves when there is nothing to find. 999 supports a smallest p value of "
    "0.001; fewer is faster and less precise.":
        "打亂組別標籤的次數，用以估計在沒有真實差異時此分析會得到的結果。"
        "999 次可支援最小 0.001 的 p 值；次數越少越快，但越不精確。",
    "Label shuffles": "標籤置換次數",
    "Significance is judged by shuffling the labels, not by a binomial test on the "
    "cross-validated accuracy — that test is anti-conservative and reports "
    "differences that are not there.":
        "顯著性是以打亂標籤的置換檢定判斷，而不是對交叉驗證準確率做二項檢定——"
        "後者過於寬鬆，會報告出並不存在的差異。",
    "Algorithm": "演算法",
    "batch      -- fast, deterministic, the standard choice.\n"
    "supervised -- XY-fused SOM; pulls the map towards separating your groups, and "
    "so cannot be used as evidence that they differ.\n"
    "relevance  -- learns a weight per metric (GRLVQ) so uninformative metrics stop "
    "diluting the distance.\n"
    "online     -- the sequential rule the ImageJ macro used.\n"
    "growing    -- grows the map where quantisation error is high.":
        "batch（批次）——快速、結果固定，是標準選擇。\n"
        "supervised（監督式）——XY 融合 SOM；會把映射圖拉向區分你的組別，"
        "因此不能作為組別有差異的證據。\n"
        "relevance（相關性學習）——為每個指標學習權重（GRLVQ），"
        "避免無資訊的指標稀釋距離。\n"
        "online（線上逐筆）——ImageJ 巨集所用的逐筆更新規則。\n"
        "growing（成長型）——在量化誤差高的區域擴增節點。",
    "Choose the map size automatically (5*sqrt(N))": "自動決定映射圖大小（5*sqrt(N)）",
    "Map size": "映射圖大小",
    "Lattice": "網格",
    "Toroidal boundary (removes edge effects)": "環面邊界（消除邊緣效應）",
    "Training epochs": "訓練週期（epoch）",
    "PCA initialisation is deterministic and converges faster than the random start "
    "used in v1.2.": "PCA 初始化結果固定，且比 v1.2 使用的隨機起始收斂更快。",
    "Initialisation": "初始化",
    "Learning rate": "學習率",
    "Neighbourhood time constant": "鄰域時間常數",
    "Supervised SOM only. 0 ignores the labels entirely (= batch SOM); high values "
    "force separation and overstate it. Report whatever you used; 0.2-0.4 is a "
    "defensible range.":
        "僅用於監督式 SOM。0 表示完全忽略標籤（等同批次 SOM）；數值高會強迫分離並誇大差異。"
        "請報告你所使用的值；0.2–0.4 是站得住腳的範圍。",
    "Label weight (supervised)": "標籤權重（監督式）",
    "Spread factor (growing)": "擴展係數（成長型）",
    "Snapshot interval": "快照間隔",
    "Random seed": "亂數種子",
    "Divide the map into behavioural clusters": "將映射圖劃分為行為群集",
    "Method": "方法",
    "to": "至",
    "Try cluster counts": "嘗試的群集數",
    "Weight nodes by their sample count": "依樣本數為節點加權",
    "Stops empty codebook vectors, which the SOM only dragged along behind their "
    "neighbours, from defining a cluster.":
        "避免那些只是被鄰近節點拖著走的空碼簿向量自成一個群集。",
    "The best count is chosen by the average rank of the silhouette, Davies-Bouldin "
    "and Calinski-Harabasz indices; every count in the range is still saved.":
        "最佳群集數依輪廓係數、Davies-Bouldin 與 Calinski-Harabasz 指數的平均排名選出；"
        "範圍內每個群集數的結果仍會全部保存。",
    "{n} projection": "{n} 個投影",
    "{n} projections": "{n} 個投影",
    "{n} label shuffles": "{n} 次標籤置換",
    "will run: {plan}": "將執行：{plan}",
    "This map is trained with the group labels, so it separates the groups by "
    "construction. Report the label weight, and take the evidence from the "
    "cross-validated statistics rather than from the map.":
        "此映射圖在訓練時使用了組別標籤，因此在建構上就會把組別分開。"
        "請報告標籤權重，並以交叉驗證的統計結果（而非映射圖）作為證據。",

    # ======================================================================
    # 5. Results
    # ======================================================================
    "5. Results": "5. 結果",
    "The conclusion is generated from the numbers, so it cannot disagree with the "
    "tables behind it. Every figure carries its own caption; use the toolbar above a "
    "figure to pan, zoom or save it.":
        "結論由數值自動產生，因此不會與背後的表格相矛盾。每張圖都附有圖說；"
        "可用圖上方的工具列平移、縮放或儲存。",
    "What this analysis found, and what it does not say.": "本分析的發現，以及它沒有說明的事。",
    "Conclusion": "結論",
    "Multivariate tests. PERMANOVA asks whether the group averages differ; PERMDISP "
    "asks the separate question of whether the groups differ in spread. A "
    "significant PERMANOVA with a significant PERMDISP may mean 'more variable', not "
    "'different'.":
        "多變量檢定。PERMANOVA 檢驗各組平均是否不同；PERMDISP 則另外檢驗各組的離散程度"
        "是否不同。PERMANOVA 與 PERMDISP 同時顯著時，可能代表「變異較大」，而非「不同」。",
    "Group difference": "組別差異",
    "Which metrics carry the difference. Read the activation column, not the "
    "weight: a large weight can belong to a metric that carries no group "
    "information and only cancels noise in another.":
        "哪些指標承載了差異。請看活化模式（activation）欄，而不是權重：權重大的指標"
        "可能本身不含任何組別資訊，只是用來抵銷另一個指標中的雜訊。",
    "Key metrics": "關鍵指標",
    "How much of each projection can be believed. Higher R_NX area preserves more "
    "neighbourhoods; a high unreliable fraction means that picture should be read "
    "for broad structure only.":
        "每個投影可信的程度。R_NX 曲線下面積越高，保留的鄰域越多；"
        "不可靠點比例高，表示該圖只適合用來看大致結構。",
    "Projection quality": "投影品質",
    "Map quality and clustering diagnostics": "映射圖品質與分群診斷",
    "Map quality": "映射圖品質",
    "Per-metric association with the experimental group, sorted by effect size":
        "各指標與實驗組別的關聯，依效應量排序",
    "Metric statistics": "指標統計",
    "Which node each sample landed on": "每個樣本落在哪個節點",
    "Assignments": "樣本歸屬",
    "The statistics layer was switched off, so there is no conclusion to report. "
    "Turn it on under <i>How to test whether the groups differ</i> and run again.":
        "統計分析已關閉，因此沒有結論可以報告。請在<i>如何檢定組別是否有差異</i>中開啟，"
        "然後重新執行。",
    "Separates:": "可區分：",
    "Does not separate:": "無法區分：",
    "Main contributing metrics:": "主要貢獻指標：",
    "Limits of this analysis": "本分析的限制",
    "The full report, including the methods paragraph and the reference list for "
    "every algorithm this run used, is written to RESULTS_REPORT.html and "
    "methods.txt when you export -- in English, followed by a complete Chinese "
    "translation.":
        "完整報告（包括方法段落，以及本次執行所用每個演算法的參考文獻）會在匯出時寫入 "
        "RESULTS_REPORT.html 與 methods.txt——先是英文版，其後附上完整的中文翻譯。",

    # ======================================================================
    # 6. Compare
    # ======================================================================
    "6. Compare": "6. 比較",
    "Structure that survives several projections is a property of your data. "
    "Structure that appears in only one is a property of that algorithm.":
        "在多種投影中都存在的結構，是你資料本身的特性；只出現在單一投影中的結構，"
        "則是該演算法的特性。",
    "Each panel is the same samples under a different projection, with the area "
    "under its R_NX curve and the share of points it placed unreliably.":
        "每個面板都是同一批樣本在不同投影下的呈現，並標出其 R_NX 曲線下面積，"
        "以及被放置得不可靠的點所佔比例。",
    "Methods side by side": "方法並列比較",
    "Higher R_NX area preserves more neighbourhoods. A supervised method will look "
    "good here for the wrong reason — it was given the labels.":
        "R_NX 曲線下面積越高，保留的鄰域越多。監督式方法在這裡會因錯誤的理由而看起來"
        "表現很好——因為它事先拿到了標籤。",
    "Quality table": "品質表",
    "Parameter scan": "參數掃描",
    "Run one method across a range of settings and see whether the answer changes. "
    "A wide plateau is the useful result: it means the setting does not matter for "
    "your data, which is worth reporting.":
        "在一系列設定下執行同一種方法，觀察結果是否改變。寬廣的高原區（plateau）"
        "才是有用的結果：它表示此設定對你的資料影響不大，值得在報告中說明。",
    "Scan": "掃描",
    "scored by": "評分依據",
    "'Unreliably placed points' is the criterion of Xia, Lee & Li (2024): it turns "
    "'what perplexity should I use' into a question with an answer. Scoring by how "
    "well the known groups separate is a supervised choice, and the report says so.":
        "「放置不可靠的點」是 Xia、Lee & Li（2024）提出的準則：它讓「該用多少 "
        "perplexity」成為一個有答案的問題。若以已知組別分開的程度評分，則屬於監督式的"
        "選擇，報告中也會如此註明。",
    "Run scan": "執行掃描",
    "Use these settings for the next run": "下次執行時使用這些設定",
    "Pins the chosen values as explicit overrides. They will appear in the methods "
    "section as a choice you made.":
        "將選定的數值固定為明確的覆寫設定，並會在方法段落中記載為你所做的選擇。",
    "  (not in the last run)": "  （上次執行未包含）",
    "Comparison figure '{name}' skipped: {error}": "已略過比較圖「{name}」：{error}",
    "Scan figure '{name}' skipped: {error}": "已略過掃描圖「{name}」：{error}",
    "{method} will use {settings} on the next run, and the methods section will "
    "record it as your choice.":
        "{method} 將在下次執行時使用 {settings}，方法段落也會將其記載為你的選擇。",

    # ======================================================================
    # 7. Export
    # ======================================================================
    "7. Export": "7. 匯出",
    "PDF and SVG keep every label as editable text, so figures can be restyled in "
    "Illustrator or Inkscape without re-running the analysis.":
        "PDF 與 SVG 會將所有標籤保留為可編輯文字，因此不必重新分析，"
        "就能在 Illustrator 或 Inkscape 中調整圖表樣式。",
    "Output folder": "輸出資料夾",
    "Formats": "格式",
    "PNG (raster, 600 dpi)": "PNG（點陣圖，600 dpi）",
    "PDF (vector, editable text)": "PDF（向量圖，可編輯文字）",
    "SVG (vector, editable text)": "SVG（向量圖，可編輯文字）",
    "MP4 training animation": "MP4 訓練過程動畫",
    "CSV tables + one .xlsx workbook": "CSV 表格 + 一個 .xlsx 活頁簿",
    "ffmpeg was not found on PATH; the animation will fall back to an animated GIF.":
        "在 PATH 中找不到 ffmpeg；動畫將改以 GIF 格式輸出。",
    "Report language": "報告語言",
    "Add a Traditional Chinese translation after the English report":
        "在英文報告之後附上繁體中文翻譯",
    "RESULTS_REPORT.html, RESULTS_REPORT.md and methods.txt are written in English "
    "first; the translation follows as a second, complete version. Text inside the "
    "figures and the contents of the tables stay in English, so every value can be "
    "matched between the two.":
        "RESULTS_REPORT.html、RESULTS_REPORT.md 與 methods.txt 會先以英文撰寫，"
        "翻譯則作為第二份完整版本附於其後。圖中的文字與表格內容維持英文，"
        "方便兩個版本之間逐一對照每個數值。",
    "Figure style": "圖表樣式",
    "180 mm is a typical double-column width; 90 mm single.":
        "180 mm 為常見的雙欄寬度；單欄為 90 mm。",
    "Figure width (mm)": "圖寬（mm）",
    "Base font size (pt)": "基本字級（pt）",
    "Font family": "字型",
    "PNG resolution (dpi)": "PNG 解析度（dpi）",
    "'somtrack' is the Okabe-Ito colour-blind-safe set; 'legacy_hsb' reproduces the "
    "v1.2 hue wheel.":
        "「somtrack」是對色盲友善的 Okabe-Ito 配色；「legacy_hsb」重現 v1.2 的色相環。",
    "Group palette": "組別配色",
    "Draw a figure legend under every panel": "在每個面板下方加上圖說",
    "Per-group map transparency": "各組映射圖透明度",
    " % opaque": " % 不透明",
    "0 samples": "0 個樣本",
    "1 sample": "1 個樣本",
    "2 samples": "2 個樣本",
    ">= 3 samples": "≥ 3 個樣本",
    "100 % opaque (fixed)": "100 % 不透明（固定）",
    "Export everything": "全部匯出",
    "Open output folder": "開啟輸出資料夾",

    # ======================================================================
    # Recipes
    # ======================================================================
    "Explore the data": "探索資料",
    "Map the data, see how it is arranged, and check whether the groups differ.":
        "將資料繪成映射圖、觀察其分布方式，並檢查組別之間是否有差異。",
    "Batch SOM, the installed unsupervised projections, and the full statistics "
    "layer. The safe default if you are not sure.":
        "批次 SOM、已安裝的非監督式投影，以及完整的統計分析。不確定時最安全的預設選擇。",
    "Test whether my groups differ": "檢定我的組別是否不同",
    "Answer 'can these treatments be told apart' with a number and a p value, not an "
    "impression.": "用數字與 p 值、而非印象，回答「這些處理能否被區分」。",
    "PERMANOVA and PERMDISP on the distances the pictures are drawn from, the energy "
    "test, and cross-validated classification with a permutation null. Supervised "
    "projections stay off, because they separate groups by construction and would "
    "beg the question.":
        "以繪圖所用的同一種距離進行 PERMANOVA 與 PERMDISP、能量檢定，以及對照置換虛無"
        "分布的交叉驗證分類。監督式投影保持關閉，因為它們在建構上就會分開組別，"
        "等於先假設了結論。",
    "Find which measurements matter": "找出重要的測量指標",
    "Rank the metrics that carry the difference, and say how much each one is worth.":
        "為承載差異的指標排序，並說明每個指標的貢獻有多大。",
    "Relevance-learning SOM, a linear SVM with Haufe-transformed activation "
    "patterns, permutation importance per metric and per correlated cluster, and "
    "effect sizes in the original units.":
        "相關性學習 SOM、經 Haufe 轉換為活化模式的線性 SVM、以單一指標與相關指標群計算的"
        "置換重要性，以及原始單位的效應量。",
    "Check that the picture is real": "確認圖中的結構是真的",
    "Run several projections and several random seeds, and report only what "
    "survives all of them.": "執行多種投影與多個亂數種子，只報告在所有情況下都存在的結構。",
    "Six projections, per-point reliability against a scrambled null, seed "
    "stability by Procrustes alignment, and the agreement matrix between methods.":
        "六種投影、對照打亂虛無分布的逐點可靠度、以 Procrustes 對齊評估的亂數種子穩定性，"
        "以及方法之間的一致性矩陣。",
    "Look for a dose or time gradient": "尋找劑量或時間梯度",
    "For treatments that form a series rather than separate classes.":
        "適用於構成連續序列、而非彼此分離類別的處理。",
    "Adds PHATE, which is built for continuous trajectories, next to PCA and MDS so "
    "a gradient is not forced into clusters.":
        "在 PCA 與 MDS 之外加入專為連續軌跡設計的 PHATE，避免把梯度硬是切成群集。",
    "Reproduce the v1.2 macro": "重現 v1.2 巨集",
    "The original ImageJ workflow, for comparison with old results.":
        "原始的 ImageJ 流程，用於與舊結果比較。",
    "Sequential SOM with random initialisation, the legacy hue wheel, PCA only, and "
    "the statistics layer switched off. Read AUDIT.md before comparing numbers: "
    "several v1.2 statistics were wrong.":
        "隨機初始化的逐筆 SOM、舊版色相環、僅 PCA，並關閉統計分析。比較數值前請先閱讀 "
        "AUDIT.md：v1.2 有數項統計量是錯誤的。",

    # ======================================================================
    # Projection registry
    # ======================================================================
    "Linear projections": "線性投影",
    "Non-linear projections": "非線性投影",
    "Supervised projections": "監督式投影",
    "PCA": "主成分分析（PCA）",
    "MDS / PCoA": "多元尺度分析（MDS／PCoA）",
    "t-SNE": "t-SNE",
    "UMAP": "UMAP",
    "densMAP": "densMAP",
    "PaCMAP": "PaCMAP",
    "PHATE": "PHATE",
    "Isomap": "Isomap",
    "Kernel PCA": "核主成分分析（Kernel PCA）",
    "LDA": "線性判別分析（LDA）",
    "PLS-DA": "PLS-DA",
    "Linear SVM": "線性 SVM",
    "Supervised UMAP": "監督式 UMAP",
    "SLISEMAP": "SLISEMAP",
    # summaries (the method picker)
    "Linear, deterministic, directly interpretable. Start here.":
        "線性、結果固定、可直接解讀。建議從這裡開始。",
    "The only projection whose distances are the ones the statistics test.":
        "唯一一種所用距離與統計檢定完全相同的投影。",
    "Sharpens local neighbourhoods. Cluster sizes and gaps mean nothing.":
        "強化局部鄰域。群集的大小與間距沒有意義。",
    "Between PCA and t-SNE: keeps some global shape, sharpens local detail.":
        "介於 PCA 與 t-SNE 之間：保留部分整體形狀，同時強化局部細節。",
    "UMAP that also preserves how tightly packed each region is, so 'this group is "
    "more variable' stays visible.":
        "同時保留各區域密集程度的 UMAP，讓「這組變異較大」仍然看得出來。",
    "Balances local detail and overall shape better than t-SNE or UMAP, and barely "
    "needs tuning.": "在局部細節與整體形狀之間取得比 t-SNE 或 UMAP 更好的平衡，幾乎不需調參。",
    "Best when the biology is a gradient (dose, time, development) rather than "
    "separate clusters.": "最適合生物現象呈梯度（劑量、時間、發育）而非分離群集的情況。",
    "Preserves distances measured along the data's own surface.":
        "保留沿著資料本身曲面量測的距離。",
    "PCA after a non-linear transform; picks up curved structure.":
        "經非線性轉換後再做 PCA；能捕捉彎曲的結構。",
    "The classical 'can these groups be told apart' projection, regularised for "
    "small samples.": "經典的「這些組別能否區分」投影，並針對小樣本做了正則化。",
    "The standard discriminant projection in metabolomics. Its score plot means "
    "nothing without the cross-validated number beside it.":
        "代謝體學中標準的判別投影。若旁邊沒有交叉驗證的數值，其得分圖毫無意義。",
    "Finds the widest gap between groups. Its weights need the Haufe transform "
    "before they can be read as 'this metric matters'.":
        "找出組別之間最寬的間隔。其權重須經 Haufe 轉換後，才能解讀為「這個指標很重要」。",
    "UMAP pulled towards the labels. Useful for showing a structure you have already "
    "demonstrated; never for demonstrating one.":
        "被拉向標籤的 UMAP。可用於展示已證實的結構，但絕不能用來證明結構存在。",
    "Places samples by which metrics explain them, so you can see whether a "
    "treatment acts the same way on every individual.":
        "依照「由哪些指標解釋」來擺放樣本，讓你看出某個處理是否對每個個體都有相同作用。",
    # details (the methods paragraph)
    "principal component analysis on the scaled feature matrix, with component "
    "loadings reported in correlation units":
        "對縮放後的特徵矩陣進行主成分分析，成分負荷量以相關係數單位報告",
    "multidimensional scaling of the sample distance matrix, using the same distance "
    "as the multivariate tests":
        "對樣本距離矩陣進行多元尺度分析，使用與多變量檢定相同的距離",
    "t-distributed stochastic neighbour embedding": "t 分布隨機鄰域嵌入",
    "uniform manifold approximation and projection": "均勻流形近似與投影",
    "density-preserving UMAP, which keeps relative local density so that differences "
    "in within-group spread survive the projection":
        "保留密度的 UMAP，會維持相對的局部密度，使組內離散程度的差異在投影後仍然保留",
    "pairwise-controlled manifold approximation, which balances near-neighbour, "
    "mid-near and far pairs explicitly":
        "成對控制的流形近似，明確平衡近鄰、中近與遠距樣本對",
    "PHATE diffusion-based embedding": "基於擴散的 PHATE 嵌入",
    "Isomap geodesic multidimensional scaling": "Isomap 測地多元尺度分析",
    "kernel principal component analysis": "核主成分分析",
    "linear discriminant analysis with Ledoit-Wolf shrinkage, drawn both in-sample "
    "and out-of-fold":
        "採 Ledoit-Wolf 收縮的線性判別分析，同時繪製樣本內與折外結果",
    "partial least squares discriminant analysis": "偏最小平方判別分析",
    "a linear support vector machine, with weights converted to activation patterns "
    "before interpretation": "線性支持向量機，其權重在解讀前先轉換為活化模式",
    "supervised UMAP with the group label as the target variable":
        "以組別標籤為目標變數的監督式 UMAP",
    "SLISEMAP supervised manifold visualisation, which places each sample by the "
    "local linear model that explains it":
        "SLISEMAP 監督式流形視覺化，依解釋每個樣本的局部線性模型來擺放樣本",
    # caveats (printed on the figures' report entries and in the methods caveats)
    "Distances between well-separated clusters are not meaningful; only local "
    "neighbourhoods are.": "明顯分離的群集之間的距離沒有意義；只有局部鄰域才有意義。",
    "Designed for continuous gradients; clusters may be drawn as branches.":
        "為連續梯度而設計；群集可能被畫成分支。",
    "Assumes the data lie on one connected surface; disconnected groups are joined "
    "by the shortest available path.":
        "假設資料位於單一連通曲面上；彼此不相連的組別會以最短可用路徑連接。",
    "Axes are combinations in a transformed space and are not directly interpretable "
    "as metric loadings.": "座標軸是轉換空間中的組合，無法直接解讀為指標負荷量。",
    "Supervised projection: the group labels were used to build these axes, so the "
    "groups separate here by construction and would separate on random data too. "
    "Judge separation from the cross-validated score, not from the picture.":
        "監督式投影：這些座標軸是用組別標籤建立的，因此組別在此處的分離是建構使然，"
        "即使是隨機資料也會分開。請以交叉驗證分數、而非圖形來判斷分離程度。",
    "Supervised projection: the group labels were used to build these axes, so the "
    "groups separate here by construction and would separate on random data too. "
    "Judge separation from the cross-validated score, not from the picture. This "
    "method in particular is known to separate training data that it cannot "
    "separate on held-out data.":
        "監督式投影：這些座標軸是用組別標籤建立的，因此組別在此處的分離是建構使然，"
        "即使是隨機資料也會分開。請以交叉驗證分數、而非圖形來判斷分離程度。"
        "此方法尤其已知會把訓練資料分開，卻無法分開保留（held-out）資料。",
    "Supervised projection: positions come from local models fitted to the labels, "
    "so proximity here means 'explained the same way', not 'similar measurements', "
    "and the groups will look organised whatever the data says.":
        "監督式投影：位置來自以標籤擬合的局部模型，因此此處的鄰近代表「以相同方式被解釋」，"
        "而非「測量值相似」；無論資料如何，各組看起來都會井然有序。",
    "UMAP needs:  pip install umap-learn": "UMAP 需要安裝：  pip install umap-learn",
    "densMAP needs:  pip install umap-learn": "densMAP 需要安裝：  pip install umap-learn",
    "PaCMAP needs:  pip install pacmap": "PaCMAP 需要安裝：  pip install pacmap",
    "PHATE needs:  pip install phate": "PHATE 需要安裝：  pip install phate",
    "Supervised UMAP needs:  pip install umap-learn":
        "監督式 UMAP 需要安裝：  pip install umap-learn",
    "SLISEMAP needs:  pip install slisemap   (it pulls in PyTorch)":
        "SLISEMAP 需要安裝：  pip install slisemap   （會一併安裝 PyTorch）",
    # parameters
    "Distance": "距離",
    "Perplexity": "困惑度（perplexity）",
    "Iterations": "迭代次數",
    "Starting layout": "起始配置",
    "Neighbours": "鄰居數",
    "Minimum separation": "最小間距",
    "Mid-near weight": "中近樣本對權重",
    "Far-pair weight": "遠距樣本對權重",
    "Kernel decay": "核衰減",
    "Kernel": "核函數",
    "Kernel width": "核寬度",
    "Label permutations": "標籤置換次數",
    "Components": "成分數",
    "Regularisation": "正則化",
    "Label influence": "標籤影響力",
    "Embedding radius": "嵌入半徑",
    "Sparsity": "稀疏度",
    "How distance between two samples is measured. Keep this the same as the "
    "statistics setting so the picture and the test describe the same thing.":
        "兩個樣本之間距離的計算方式。請與統計設定保持一致，讓圖與檢定描述同一件事。",
    "Classical scaling is exact and deterministic. SMACOF fits the distances "
    "iteratively and reports a stress value.":
        "古典尺度法精確且結果固定；SMACOF 以迭代方式擬合距離，並報告應力（stress）值。",
    "Roughly how many neighbours each point is pulled towards. Small values show "
    "fine detail, large values show broad structure. It must stay below (n-1)/3.":
        "大約是每個點被拉向多少個鄰居。數值小呈現細節，數值大呈現整體結構。"
        "必須小於 (n-1)/3。",
    "More iterations settle the layout; 1000 is almost always enough.":
        "迭代次數越多，配置越穩定；1000 次幾乎都已足夠。",
    "Starting from PCA makes the result reproducible and keeps more of the global "
    "arrangement.": "從 PCA 開始可使結果可重現，並保留更多整體排列。",
    "How much of the data each point looks at. Small values show local detail, "
    "large values show the overall shape.":
        "每個點參考多少資料。數值小呈現局部細節，數值大呈現整體形狀。",
    "How tightly points may be packed. Larger values spread clusters out and make "
    "them easier to read.": "點可以擠得多緊密。數值越大，群集越分散、越容易判讀。",
    "How distance between two samples is measured.": "兩個樣本之間距離的計算方式。",
    "10 works for almost any data set below ten thousand samples; this is the "
    "parameter PaCMAP is least sensitive to.":
        "對樣本數少於一萬的資料集，10 幾乎都適用；這是 PaCMAP 最不敏感的參數。",
    "Higher values pull the overall arrangement together.": "數值越高，整體排列越聚攏。",
    "Higher values push unrelated points further apart.": "數值越高，無關的點被推得越遠。",
    "How far the diffusion step reaches.": "擴散步驟所能觸及的範圍。",
    "How quickly influence falls off with distance.": "影響力隨距離衰減的速度。",
    "How many neighbours define the surface. Too few breaks it into pieces; too many "
    "short-circuits it.": "用多少鄰居定義曲面。太少會使曲面破碎；太多則會造成捷徑。",
    "Which kind of curvature to allow.": "允許何種曲率。",
    "0 lets scikit-learn choose (1 / number of metrics).":
        "0 表示由 scikit-learn 自動選擇（1／指標數）。",
    "How many times the group labels are shuffled to work out what this projection "
    "would achieve by chance.": "打亂組別標籤的次數，用以估計此投影在純屬機遇時會得到的結果。",
    "How many latent directions to fit. More components fit the training data better "
    "and generalise worse.": "要擬合多少個潛在方向。成分越多，越貼合訓練資料，但泛化能力越差。",
    "How many times the labels are shuffled to build the null.":
        "為建立虛無分布而打亂標籤的次數。",
    "Small values keep the boundary simple and are the safer choice when there are "
    "few samples.": "數值小會使邊界保持簡單，在樣本少時是較安全的選擇。",
    "0 ignores the labels entirely (ordinary UMAP); 1 lets them dominate. Report "
    "whatever you used.": "0 表示完全忽略標籤（即一般的 UMAP）；1 表示讓標籤主導。"
                          "請報告你所使用的值。",
    "How spread out the layout is. Larger values separate the local models more "
    "sharply.": "配置的展開程度。數值越大，各局部模型分得越清楚。",
    "Higher values force each local model to use fewer metrics, which makes the "
    "explanations easier to read.": "數值越高，每個局部模型使用的指標越少，解釋也越容易閱讀。",
    "{method} is not installed.": "{method} 尚未安裝。",
    "{method} needs at least two labelled groups.": "{method} 至少需要兩個有標籤的組別。",
    "{method} needs at least {n} samples (this data set has {have}).":
        "{method} 至少需要 {n} 個樣本（此資料集有 {have} 個）。",
    "{method} failed and was skipped: {error}": "{method} 執行失敗，已略過：{error}",

    # ======================================================================
    # Parameter scan
    # ======================================================================
    "neighbourhood preservation across all scales": "各尺度下的鄰域保留程度",
    "fraction of points the projection places unreliably": "投影中被放置得不可靠的點所佔比例",
    "absence of invented neighbours": "不產生虛假鄰居的程度",
    "how cleanly the known groups sit apart in the projection":
        "已知組別在投影中分開的清晰程度",
    "The best setting was {best}; no other setting came within {tol:.0%} of it.":
        "最佳設定為 {best}；其他設定都未達到其 {tol:.0%} 範圍內。",
    "{param} from {lo} to {hi}": "{param} 從 {lo} 到 {hi}",
    "{n} of {total} settings scored within {tol:.0%} of the best ({ranges}), so the "
    "result does not depend on the exact choice. {best} was used.":
        "{total} 組設定中有 {n} 組的分數落在最佳值的 {tol:.0%} 範圍內（{ranges}），"
        "因此結果不取決於確切的選擇。本次採用 {best}。",
    "Settings were scored by {criterion}.": "各設定依{criterion}評分。",
    "Settings were scored by quantisation error.": "各設定依量化誤差評分。",
    "{method} parameter scan": "{method} 參數掃描",
    "{n} settings scored by {criterion}": "共 {n} 組設定，依{criterion}評分",
    "SOM parameter scan": "SOM 參數掃描",
    "{n} settings compared by map quality": "共 {n} 組設定，依映射圖品質比較",

    # ======================================================================
    # Progress, warnings and notes written during a run
    # ======================================================================
    "Computing metrics ({i}/{n} tracks)": "計算指標中（{i}/{n} 條軌跡）",
    "Preparing feature matrix": "準備特徵矩陣",
    "'{algorithm}' SOM needs >= 2 groups; falling back to batch SOM.":
        "「{algorithm}」SOM 需要至少 2 個組別；改用批次 SOM。",
    "Training SOM": "訓練 SOM",
    "Training SOM (epoch {i}/{n})": "訓練 SOM（第 {i}/{n} 個訓練週期）",
    "Clustering nodes": "節點分群",
    "Running projections": "執行投影",
    "Projections ({i}/{n})": "投影（{i}/{n}）",
    "Testing metric-group associations": "檢定指標與組別的關聯",
    "Analysis complete": "分析完成",
    "Testing whether the groups differ": "檢定組別是否有差異",
    "Cross-validating group assignment": "交叉驗證組別判別",
    "Measuring which metrics matter": "評估哪些指標重要",
    "Estimating effect sizes": "估計效應量",
    "Rendering {name}": "繪製 {name}",
    "Figures complete": "圖表完成",
    "Writing figures": "寫入圖表",
    "Writing tables": "寫入表格",
    "Rendering animation": "繪製動畫",
    "Writing the report": "撰寫報告",
    "Rendering the colour check": "繪製色覺檢查圖",
    "Supervised projections ({methods}) were requested but are off by default, "
    "because they separate groups by construction. Enable them with "
    "embedding.allow_supervised = True once the cross-validated result is in hand.":
        "已要求執行監督式投影（{methods}），但它們預設為關閉，因為它們在建構上就會分開"
        "組別。取得交叉驗證結果後，可設定 embedding.allow_supervised = True 啟用。",
    "Figure '{name}' skipped: {error}": "已略過圖「{name}」：{error}",
    "Excel workbook not written: {error}": "未寫入 Excel 活頁簿：{error}",
    "Training animation skipped: {error}": "已略過訓練動畫：{error}",
    "Report not written: {error}": "未寫入報告：{error}",
    "Colour check not written: {error}": "未寫入色覺檢查圖：{error}",
    "No features selected.": "未選擇任何特徵。",
    "Dropped {n} all-NaN feature(s).": "已移除 {n} 個全為缺失值（NaN）的特徵。",
    "Dropped {n} sample(s) containing NaN.": "已移除 {n} 個含有缺失值（NaN）的樣本。",
    "Dropped {n} feature(s) containing NaN.": "已移除 {n} 個含有缺失值（NaN）的特徵。",
    "Median-imputed {n} missing value(s).": "已以中位數填補 {n} 個缺失值。",
    "Fewer than 3 samples remain after the NaN policy.": "依缺失值處理原則處理後，剩下的樣本少於 3 個。",
    "Winsorised to the [{lo:.1%}, {hi:.1%}] range.": "已縮尾至 [{lo:.1%}, {hi:.1%}] 範圍。",
    "Dropped {n} constant feature(s).": "已移除 {n} 個常數特徵。",
    "Pruned {n} feature(s) with |r| > {r:.2f}: {names}":
        "已剔除 {n} 個 |r| > {r:.2f} 的特徵：{names}",
    "Applied user feature weights.": "已套用使用者設定的特徵權重。",
    "Fewer than 2 usable features remain.": "可用的特徵少於 2 個。",
    "Supervised SOM needs at least two experimental groups.": "監督式 SOM 至少需要兩個實驗組別。",
    "Relevance SOM needs at least two experimental groups.": "相關性學習 SOM 至少需要兩個實驗組別。",

    # ======================================================================
    # Replicate structure and cross-validation
    # ======================================================================
    "Samples were treated as independent units.": "樣本被視為彼此獨立的單位。",
    "The replicate column held a single value, so samples were treated as "
    "independent units.": "重複欄位只有單一值，因此樣本被視為彼此獨立的單位。",
    "Replicates were nested within groups ({n} replicates, each belonging to one "
    "group), so the replicate was treated as the experimental unit.":
        "重複巢套於組別之內（共 {n} 個重複，每個重複只屬於一個組別），"
        "因此以重複作為實驗單位。",
    "Every replicate contained several groups ({n} blocks), so the design was "
    "treated as randomised blocks and labels were permuted within blocks.":
        "每個重複都包含多個組別（共 {n} 個區集），因此將設計視為隨機區集，"
        "並在各區集內置換標籤。",
    "Replicates were partly nested and partly crossed with the groups ({n} "
    "replicates); labels were permuted within blocks where possible and whole "
    "replicates were kept inside one cross-validation fold.":
        "重複與組別之間部分巢套、部分交叉（共 {n} 個重複）；在可行處於區集內置換標籤，"
        "並將整個重複保留在同一個交叉驗證折內。",
    "leave-one-replicate-out cross-validation": "留一重複（leave-one-replicate-out）交叉驗證",
    "Each fold held out one whole replicate.": "每一折都保留一整個重複作為測試。",
    "{k}-fold cross-validation with whole replicates held out together":
        "整個重複一併保留為測試資料的 {k} 折交叉驗證",
    "No replicate appeared in both the training and the test set, so the score "
    "cannot come from recognising a replicate.":
        "沒有任何重複同時出現在訓練集與測試集中，因此分數不可能來自辨認出某個重複。",
    "{k}-fold grouped cross-validation": "分組 {k} 折交叉驗證",
    "No replicate appeared in both training and test sets.":
        "沒有任何重複同時出現在訓練集與測試集中。",
    "{k}-fold stratified cross-validation": "分層 {k} 折交叉驗證",
    "Folds preserved the group proportions.": "各折保持了組別比例。",
    "{k}-fold stratified cross-validation repeated {repeats} times":
        "重複 {repeats} 次的分層 {k} 折交叉驗證",
    "Folds preserved the group proportions; repeating the split reduces the "
    "influence of any one partition.": "各折保持了組別比例；重複切分可降低任何單一切分方式的影響。",

    # ======================================================================
    # Statistics: tests, classification, verdict
    # ======================================================================
    "euclidean": "歐氏",
    "correlation": "相關",
    "cityblock": "城市街區（曼哈頓）",
    "cosine": "餘弦",
    "hexagonal": "六角形",
    "rectangular": "矩形",
    "At least two groups are needed for a separation test.": "組別分離檢定至少需要兩個組別。",
    "The multivariate test could not be run.": "無法執行多變量檢定。",
    "The groups did not differ detectably in their overall multivariate phenotype.":
        "各組在整體多變量表型上沒有可偵測的差異。",
    "The groups differ, but they also differ in how variable they are, so at least "
    "part of the separation is a difference in spread rather than a shift in the "
    "group average. Read a result like this as 'these animals are more variable', "
    "not 'these animals are faster', unless the per-metric effects say otherwise.":
        "各組之間有差異，但它們的變異程度也不同，因此至少有一部分的分離來自離散程度的差異，"
        "而非組平均的位移。除非各指標的效應另有說明，否則這類結果應解讀為「這些動物變異"
        "較大」，而不是「這些動物游得較快」。",
    "The groups differ in their multivariate average, and their within-group "
    "variability is comparable, so this is a genuine shift in phenotype rather than "
    "a change in variability.":
        "各組的多變量平均不同，而組內變異程度相當，因此這是表型的真實位移，"
        "而非變異程度的改變。",
    "regularised linear discriminant analysis": "正則化線性判別分析",
    "linear discriminant analysis with Ledoit-Wolf shrinkage of the covariance "
    "estimate, which is what keeps it stable when the number of metrics approaches "
    "the number of samples":
        "對共變異數估計採 Ledoit-Wolf 收縮的線性判別分析，"
        "使其在指標數接近樣本數時仍保持穩定",
    "a linear support vector machine with balanced class weights":
        "採平衡類別權重的線性支持向量機",
    "regularised logistic regression": "正則化邏輯斯迴歸",
    "multinomial logistic regression with an L2 penalty": "採 L2 懲罰項的多項邏輯斯迴歸",
    "a random forest with balanced class weights": "採平衡類別權重的隨機森林",
    "Classification could not be run.": "無法執行分類。",
    " [95% CI {lo:.2f}-{hi:.2f}]": " [95% 信賴區間 {lo:.2f}–{hi:.2f}]",
    "permutation p = {p:.3f}": "置換 p = {p:.3f}",
    "no permutation test": "未進行置換檢定",
    "balanced accuracy {ba:.2f}{ci} (chance {chance:.2f}; {p}, {n} permutations)":
        "平衡準確率 {ba:.2f}{ci}（機率水準 {chance:.2f}；{p}，{n} 次置換）",
    "Too few samples per group to cross-validate.": "每組樣本太少，無法進行交叉驗證。",
    "Cross-validation produced no complete out-of-fold pass.":
        "交叉驗證未能產生一次完整的折外預測。",
    "The design allows only {eff} distinct label arrangements, so no p value below "
    "{p:.3g} is attainable.": "此設計只允許 {eff} 種不同的標籤排列，因此 p 值不可能低於 {p:.3g}。",
    "Too few samples per group for permutation importance.": "每組樣本太少，無法計算置換重要性。",
    "Strong evidence that the groups differ.": "有強力證據顯示組別之間有差異。",
    "Moderate evidence that the groups differ.": "有中等程度的證據顯示組別之間有差異。",
    "Weak or borderline evidence that the groups differ.": "僅有微弱或臨界的證據顯示組別之間有差異。",
    "No detectable difference between the groups.": "組別之間沒有可偵測的差異。",
    "The evidence could not be assessed.": "無法評估證據。",
    "Group pairs that separate: {pairs}.": "可區分的組別配對：{pairs}。",
    "Group pairs that do not separate: {pairs}.": "無法區分的組別配對：{pairs}。",
    "Metrics contributing most to the separation: {metrics}.":
        "對分離貢獻最大的指標：{metrics}。",
    "Read with these limits in mind: {caveats}": "閱讀時請留意以下限制：{caveats}",
    "Separates: {pairs}": "可區分：{pairs}",
    "Does not separate: {pairs}": "無法區分：{pairs}",
    "Main drivers: {metrics}": "主要驅動指標：{metrics}",
    "p not available": "無法取得 p 值",
    "p = {p:.4f}, which is the floor set by the number of permutations rather than a "
    "measured value": "p = {p:.4f}（此為置換次數所決定的下限，並非實際測得的數值）",
    "{n} samples in {k} groups were described by {m} metrics.":
        "{k} 個組別中的 {n} 個樣本，以 {m} 個指標描述。",
    "PERMANOVA on {metric} distances: pseudo-F = {F:.2f}, R2 = {r2:.3f} ({pct:.1f}% "
    "of the multivariate variation is accounted for by the grouping), {p}, {n} "
    "permutations.":
        "以{metric}距離進行 PERMANOVA：pseudo-F = {F:.2f}，R2 = {r2:.3f}"
        "（組別解釋了 {pct:.1f}% 的多變量變異），{p}，{n} 次置換。",
    "PERMDISP (equality of within-group spread): F = {F:.2f}, {p}.":
        "PERMDISP（組內離散程度的同質性）：F = {F:.2f}，{p}。",
    "Because the groups also differ in spread, the PERMANOVA result alone cannot "
    "distinguish a shift in the average from a change in variability.":
        "由於各組的離散程度也不同，單憑 PERMANOVA 的結果無法區分平均的位移與變異程度的改變。",
    "The energy test found a difference in the distributions that PERMANOVA did not: "
    "the groups may differ in shape rather than in average or spread.":
        "能量檢定發現了 PERMANOVA 沒有發現的分布差異：各組的差異可能在於分布形狀，"
        "而非平均或離散程度。",
    " [95% CI {lo:.2f} to {hi:.2f}]": " [95% 信賴區間 {lo:.2f} 至 {hi:.2f}]",
    "Held-out samples were assigned to their group with a balanced accuracy of "
    "{ba:.2f}{ci} against a chance level of {chance:.2f}, using {model} and {cv} "
    "({p}, {n} label permutations).":
        "使用{model}與{cv}，保留樣本被判入正確組別的平衡準確率為 {ba:.2f}{ci}，"
        "機率水準為 {chance:.2f}（{p}，{n} 次標籤置換）。",
    "No replicate structure was supplied, so every sample was treated as an "
    "independent experimental unit. If several samples came from one dish, clutch "
    "or imaging session, supply that column and re-run: the p values here would "
    "otherwise be too small.":
        "未提供重複結構，因此每個樣本都被視為獨立的實驗單位。若多個樣本來自同一培養皿、"
        "同一窩或同一次影像擷取，請提供該欄位並重新執行：否則此處的 p 值會偏小。",
    "A supervised projection was run. Supervised projections separate the groups by "
    "construction and will do so even on random data, so their figures are "
    "illustrations, not evidence; the cross-validated numbers above are the evidence.":
        "本次執行了監督式投影。監督式投影在建構上就會分開組別，即使是隨機資料也一樣，"
        "因此其圖形只是示意而非證據；上方交叉驗證的數值才是證據。",
    "More than 15% of points are poorly placed in {methods}; read those panels for "
    "broad structure only.": "{methods} 中有超過 15% 的點放置不佳；這些面板只適合用來看大致結構。",
    "Group(s) {groups} have fewer than ten samples; estimates for them are unstable "
    "however small the p value.": "組別 {groups} 的樣本少於十個；無論 p 值多小，對它們的估計都不穩定。",
    "{a} vs {b} (balanced accuracy {ba:.2f})": "{a} 對 {b}（平衡準確率 {ba:.2f}）",

    # ======================================================================
    # Methods log
    # ======================================================================
    "{label}, {detail}": "{label}，{detail}",
    "{text} ({refs})": "{text}（{refs}）",
    "Data preparation": "資料前處理",
    "Feature extraction": "特徵擷取",
    "Clustering": "分群",
    "Projection": "投影",
    "Statistics": "統計",
    "Visualization": "視覺化",
    "Software": "軟體",
    "feature scaling and pruning": "特徵縮放與剔除",
    "metrics were scaled with the {scaler} transform and collinear metrics were "
    "pruned at |r| > {r:g}; missing values were handled by '{nan_policy}'":
        "指標以 {scaler} 轉換縮放，並剔除 |r| > {r:g} 的共線指標；缺失值以「{nan_policy}」處理",
    "metrics were scaled with the {scaler} transform; missing values were handled by "
    "'{nan_policy}'": "指標以 {scaler} 轉換縮放；缺失值以「{nan_policy}」處理",
    "batch self-organising map": "批次自組織映射圖",
    "sequential (online) self-organising map": "逐筆（線上）自組織映射圖",
    "XY-fused supervised self-organising map": "XY 融合監督式自組織映射圖",
    "self-organising map with GRLVQ relevance learning": "採 GRLVQ 相關性學習的自組織映射圖",
    "growing self-organising map": "成長型自組織映射圖",
    "self-organising map": "自組織映射圖",
    "trained on the scaled feature matrix, with map quality reported as quantisation "
    "and topographic error": "以縮放後的特徵矩陣訓練，映射圖品質以量化誤差與拓撲誤差報告",
    "The group labels were used during training, so the map separates the groups "
    "partly by construction; report the label weight and judge separation from the "
    "cross-validated statistics.":
        "訓練時使用了組別標籤，因此映射圖對組別的分離有一部分是建構使然；"
        "請報告標籤權重，並以交叉驗證的統計結果判斷分離程度。",
    "second-level clustering of the SOM codebook": "SOM 碼簿的第二層分群",
    "{method} over k = {k_min}-{k_max}, with k chosen by the average rank of three "
    "internal validity indices": "以 {method} 嘗試 k = {k_min}–{k_max}，並依三個內部效度指數的平均排名選出 k",
    "per-metric association tests": "逐指標關聯檢定",
    "one-way ANOVA and Kruskal-Wallis per metric with eta-squared and "
    "epsilon-squared effect sizes, corrected across metrics by the "
    "Benjamini-Hochberg procedure":
        "對每個指標進行單因子變異數分析與 Kruskal-Wallis 檢定，以 eta 平方與 epsilon 平方"
        "作為效應量，並以 Benjamini-Hochberg 程序進行跨指標校正",
    "aggregation to experimental units": "彙整至實驗單位",
    "metrics were averaged within each replicate before testing, so the replicate "
    "rather than the individual is the unit of analysis":
        "檢定前先在每個重複內將指標取平均，使分析單位為重複而非個體",
    "replicate-aware permutation and cross-validation": "考量重複結構的置換與交叉驗證",
    "PERMANOVA": "PERMANOVA",
    "permutational multivariate analysis of variance on {metric} distances, with "
    "R-squared as the effect size": "以{metric}距離進行置換多變量變異數分析，以 R 平方作為效應量",
    "PERMDISP": "PERMDISP",
    "a separate permutation test of homogeneity of multivariate dispersions, because "
    "PERMANOVA rejects for a difference in spread as readily as for a difference in "
    "location":
        "另行以置換檢定檢驗多變量離散程度的同質性，因為 PERMANOVA 對離散程度的差異"
        "與對位置的差異同樣容易拒絕虛無假設",
    "energy k-sample test": "能量 k 樣本檢定",
    "a distribution-free test consistent against any difference in distribution, not "
    "only a shift in the mean": "一種不需分布假設的檢定，對任何分布差異都具一致性，而不僅限於平均的位移",
    "Mahalanobis distance between group centroids": "組別質心之間的馬氏距離",
    "with the pooled within-group covariance shrunk towards a well-conditioned target "
    "before inversion": "反矩陣運算前，先將合併組內共變異數收縮至條件良好的目標矩陣",
    "cross-validated classification": "交叉驗證分類",
    "{model}, evaluated by {cv} and scored by balanced accuracy; significance was "
    "assessed by shuffling the group labels rather than by a binomial test, which is "
    "anti-conservative for cross-validated accuracies":
        "{model}，以{cv}進行評估，並以平衡準確率計分；顯著性以打亂組別標籤的置換檢定評估，"
        "而非二項檢定——後者對交叉驗證準確率過於寬鬆",
    "Haufe transform of the classifier weights": "分類器權重的 Haufe 轉換",
    "classifier weights were converted to activation patterns before interpretation, "
    "because a large weight can belong to a metric that carries no group information "
    "and only cancels noise in another":
        "解讀前先將分類器權重轉換為活化模式，因為權重大的指標可能本身不含組別資訊，"
        "只是用來抵銷另一個指標中的雜訊",
    "permutation importance": "置換重要性",
    "cross-validated permutation importance scored by the loss in balanced accuracy, "
    "computed per metric and per cluster of metrics correlated above |r| = {r:g}":
        "以平衡準確率的損失計分的交叉驗證置換重要性，分別針對單一指標，"
        "以及相關高於 |r| = {r:g} 的指標群計算",
    "effect sizes with bootstrap intervals": "附拔靴法信賴區間的效應量",
    "Hedges' g and the raw difference in measurement units, each with a "
    "bias-corrected and accelerated bootstrap 95% interval":
        "Hedges' g 與以測量單位表示的原始差異，各附偏誤校正加速（BCa）拔靴法 95% 信賴區間",
    "Python analysis stack": "Python 分析套件",
    "NumPy, SciPy, scikit-learn, pandas and matplotlib":
        "NumPy、SciPy、scikit-learn、pandas 與 matplotlib",

    # ======================================================================
    # Figure captions (the report's figure index)
    # ======================================================================
    "The conclusion, generated from the numbers in the panels below.":
        "結論，由下方各面板中的數值自動產生。",
    "Whether the groups differ, and whether the difference is a shift in the average "
    "or a change in within-group spread.":
        "組別是否不同，以及差異來自平均的位移還是組內離散程度的改變。",
    "Which specific pairs of groups differ.": "具體是哪些組別配對之間有差異。",
    "Cross-validated group assignment against a permutation null.":
        "交叉驗證的組別判別，並與置換虛無分布對照。",
    "Which metrics carry the difference, after the Haufe transform that separates "
    "them from suppressor variables.": "經 Haufe 轉換排除抑制變數後，承載差異的指標。",
    "How much held-out accuracy each metric, and each correlated cluster of metrics, "
    "is worth.": "每個指標及每個相關指標群對保留樣本準確率的貢獻。",
    "Standardised effect size per metric with bootstrap intervals.":
        "各指標的標準化效應量，附拔靴法信賴區間。",
    "{metric} by group, showing every individual and each replicate's own mean.":
        "依組別呈現 {metric}，顯示每個個體以及每個重複各自的平均。",
    "{metric}: the size of the difference, with its uncertainty.":
        "{metric}：差異的大小及其不確定性。",
    "Every projection that was run, side by side with its quality score.":
        "所有已執行的投影並列比較，附其品質分數。",
    "How much of each neighbourhood each projection preserved, across all scales.":
        "每個投影在各尺度下保留了多少鄰域。",
    "The leading projections on one row for direct comparison.": "主要投影並排於一列，方便直接比較。",
    "{method} projection with metric-direction arrows.": "{method} 投影，附指標方向箭頭。",
    "Which points {method} placed unreliably.": "{method} 將哪些點放置得不可靠。",
    "{method} in-sample against out-of-fold -- the panel that separates a real "
    "difference from one the method was handed.":
        "{method} 的樣本內與折外結果對照——這個面板能區分真實的差異與方法本身被給予的差異。",
    "How much the projections agree about who is next to whom.":
        "各投影對「誰與誰相鄰」的一致程度。",
    "How much of each layout is the random seed.": "每個配置中有多少是亂數種子造成的。",
    "Trustworthiness and continuity of each projection.": "各投影的可信度與連續性。",
    "Where each group's samples land on the map, and how mixed each node is.":
        "各組樣本落在映射圖的哪些位置，以及每個節點的混雜程度。",
    "Node composition as pie glyphs.": "以圓餅符號呈現的節點組成。",
    "One map per group, with node opacity graded by occupancy.":
        "每組一張映射圖，節點不透明度依佔用程度分級。",
    "The same maps split by replicate, to show whether the pattern holds in each.":
        "同樣的映射圖依重複拆分，以顯示模式是否在每個重複中都成立。",
    "Distances between neighbouring nodes: the map's own cluster boundaries.":
        "相鄰節點間的距離：映射圖本身的群集邊界。",
    "How many samples each node captured.": "每個節點捕捉到多少樣本。",
    "Each metric's value across the map, in its original units.":
        "各指標在映射圖上的數值分布，以原始單位表示。",
    "The map divided into behavioural clusters.": "劃分為行為群集的映射圖。",
    "Why that number of clusters was chosen.": "為何選擇這個群集數。",
    "What distinguishes each behavioural cluster, in measurement units.":
        "以測量單位呈現各行為群集的區別特徵。",
    "The direction each metric increases across the map.": "各指標在映射圖上遞增的方向。",
    "All metric directions on one polar plot: metrics at the same bearing are "
    "redundant.": "所有指標方向畫在同一張極座標圖上：方位相同的指標彼此重複。",
    "{metric} across the map.": "{metric} 在映射圖上的分布。",
    "Where {group} sits more often than chance predicts.":
        "{group} 出現頻率高於機率預期的位置。",
    "Training diagnostics for the map.": "映射圖的訓練診斷。",
    "Per-metric effect size against multivariate importance.":
        "各指標的效應量與多變量重要性之對照。",
    "Group means per metric, standardised.": "各指標的組平均（標準化）。",
    "Distribution of the strongest metrics, by group.": "最強指標依組別的分布。",
    "Which metrics are measuring the same thing.": "哪些指標在測量同一件事。",

    # ======================================================================
    # The report
    # ======================================================================
    "Multivariate analysis of locomotion metrics": "運動指標的多變量分析",
    "{title} (translation)": "{title}（中文翻譯）",
    "Methods -- {title} (translation)": "方法——{title}（中文翻譯）",
    "Generated {date} by SOMTrack {version}": "由 SOMTrack {version} 於 {date} 產生",
    "This is a translation of the English report above. Figures, table contents, "
    "metric and group names and the reference list are kept in English, so every "
    "value can be matched between the two versions.":
        "本文為上方英文報告的中文翻譯。圖表、表格內容、指標與組別名稱以及參考文獻清單"
        "均維持英文，方便兩個版本之間逐一對照每個數值。",
    "English": "English",
    "Translation": "中文翻譯",
    "Data": "資料",
    "{n} samples described by {m} metrics, in {k} groups: {counts}.":
        "共 {n} 個樣本，以 {m} 個指標描述，分屬 {k} 個組別：{counts}。",
    "{n} metrics were dropped before analysis ({names}).": "分析前移除了 {n} 個指標（{names}）。",
    "Do the groups differ?": "組別之間有差異嗎？",
    "PERMANOVA on {metric} distances gave pseudo-F = {F:.2f} with R2 = {r2:.3f} and "
    "p = {p:.4f} over {n} permutations. The grouping therefore accounts for "
    "{pct:.1f}% of the multivariate variation.":
        "以{metric}距離進行 PERMANOVA，經 {n} 次置換得到 pseudo-F = {F:.2f}、"
        "R2 = {r2:.3f}、p = {p:.4f}。因此組別解釋了 {pct:.1f}% 的多變量變異。",
    "PERMDISP, which asks the separate question of whether the groups differ in how "
    "variable they are, gave F = {F:.2f}, p = {p:.4f}.":
        "PERMDISP 另外檢驗各組的變異程度是否不同，結果為 F = {F:.2f}、p = {p:.4f}。",
    "The energy k-sample test, which responds to any difference in distribution "
    "rather than only to a shift in the average, gave p = {p:.4f}.":
        "能量 k 樣本檢定會對任何分布差異產生反應，而不僅是平均的位移；其結果為 p = {p:.4f}。",
    "Omnibus tests": "整體檢定",
    "Every pair of groups, FDR-corrected": "所有組別配對（經 FDR 校正）",
    "Can new samples be assigned to a group?": "新樣本能被判入正確的組別嗎？",
    " (95% CI {lo:.2f} to {hi:.2f})": "（95% 信賴區間 {lo:.2f} 至 {hi:.2f}）",
    "Using {model} and {cv}, held-out samples were assigned to their group with a "
    "balanced accuracy of {ba:.3f}{ci}, against a chance level of {chance:.3f}. "
    "Shuffling the labels {n} times gave p = {p:.4f}. Cohen's kappa was {kappa:.3f}.":
        "使用{model}與{cv}，保留樣本被判入正確組別的平衡準確率為 {ba:.3f}{ci}，"
        "機率水準為 {chance:.3f}。將標籤打亂 {n} 次得到 p = {p:.4f}。"
        "Cohen's kappa 為 {kappa:.3f}。",
    "Confusion matrix (out-of-fold predictions)": "混淆矩陣（折外預測）",
    "Separability of each pair of groups": "各組別配對的可區分程度",
    "Which metrics drive the difference?": "哪些指標驅動了差異？",
    "Classifier weights are reported next to Haufe-transformed activation patterns. "
    "The weights say how the model extracts the signal and can be large for a metric "
    "that carries none; the activation pattern says which metrics actually covary "
    "with the group difference, and is the column to read.":
        "分類器權重與經 Haufe 轉換的活化模式並列呈現。權重說明模型如何擷取訊號，"
        "即使指標本身不含訊號，權重也可能很大；活化模式則說明哪些指標真正與組別差異共變，"
        "應以此欄為準。",
    "Classifier weights and activation patterns": "分類器權重與活化模式",
    "Permutation importance measures how much balanced accuracy is lost when one "
    "metric is shuffled, averaged over {n} repeats within each cross-validation fold.":
        "置換重要性衡量打亂單一指標時平衡準確率損失多少，並在每個交叉驗證折內重複 {n} 次取平均。",
    "Permutation importance, per metric": "置換重要性（單一指標）",
    "Metrics correlated above |r| = {r:g} were also permuted as whole clusters, "
    "because shuffling one member of a correlated group understates all of them: the "
    "model simply reads the others.":
        "相關高於 |r| = {r:g} 的指標也以整群方式置換，因為只打亂相關指標群中的一員會低估"
        "所有成員的重要性：模型只會改讀其他成員。",
    "Permutation importance, per correlated cluster": "置換重要性（相關指標群）",
    "Effect sizes are given in the original measurement units with bias-corrected "
    "bootstrap confidence intervals, so the size of each difference can be judged "
    "independently of its p value.":
        "效應量以原始測量單位表示，並附偏誤校正的拔靴法信賴區間，因此每個差異的大小可以"
        "獨立於其 p 值來判斷。",
    "Effect size per metric, against the reference group": "各指標相對於參考組的效應量",
    "Projections": "投影",
    "Every projection is reported with its neighbourhood preservation, so a picture "
    "can be judged before it is believed. Trustworthiness penalises neighbours the "
    "projection invented; continuity penalises neighbours it lost; the area under the "
    "R_NX curve summarises both across all neighbourhood sizes and is comparable "
    "between methods.":
        "每個投影都附有其鄰域保留程度，讓圖形在被採信之前先接受評估。可信度會懲罰投影憑空"
        "製造的鄰居；連續性會懲罰投影遺失的鄰居；R_NX 曲線下面積則綜合所有鄰域大小下的"
        "兩者，可在方法之間比較。",
    "Across the projections that were run, neighbouring samples agreed on average "
    "{pct:.0f}% of the time. Structure that survives several projections is a "
    "property of the data; structure visible in only one is a property of that "
    "algorithm.":
        "在已執行的各投影之間，相鄰樣本平均有 {pct:.0f}% 的時間一致。在多種投影中都存在的"
        "結構是資料本身的特性；只在單一投影中可見的結構，則是該演算法的特性。",
    "Shared nearest neighbours between projections": "各投影之間共有的最近鄰",
    "Supervised projections were run and are shown both in-sample and out-of-fold. A "
    "supervised projection separates the groups by construction and does so on "
    "random data too, so its in-sample panel is an illustration; the out-of-fold "
    "panel and the cross-validated score are the evidence.":
        "本次執行了監督式投影，並同時呈現樣本內與折外結果。監督式投影在建構上就會分開組別，"
        "連隨機資料也不例外，因此其樣本內面板只是示意；折外面板與交叉驗證分數才是證據。",
    "A {w} x {h} {lattice} map was trained for {epochs} epochs with the {algorithm} "
    "algorithm.": "以 {algorithm} 演算法訓練一個 {w} × {h} 的{lattice}網格映射圖，"
                  "共 {epochs} 個訓練週期。",
    "Map quality: {values}": "映射圖品質：{values}",
    "The codebook was divided into {k} behavioural clusters by {method}, chosen by "
    "the average rank of the silhouette, Davies-Bouldin and Calinski-Harabasz "
    "indices.":
        "以 {method} 將碼簿劃分為 {k} 個行為群集，群集數依輪廓係數、Davies-Bouldin 與 "
        "Calinski-Harabasz 指數的平均排名選出。",
    "Methods": "方法",
    "The text below describes the steps this run actually performed, with the "
    "parameters it used. Citations are given in the form used by the reference list "
    "that follows.":
        "以下文字描述本次執行實際進行的步驟及所用參數。引用格式與後附的參考文獻清單一致。",
    "The text below describes the steps this run actually performed, with the "
    "parameters it used.": "以下文字描述本次執行實際進行的步驟及所用參數。",
    "Caveats attached to the methods above.": "上述方法附帶的注意事項。",
    "Caveats": "注意事項",
    "References": "參考文獻",
    "The reference list is the one at the end of the English report above; "
    "bibliographic entries are not translated.":
        "參考文獻清單即上方英文報告末尾所列；書目資料不予翻譯。",
    "The reference list is the one at the end of the English section above; "
    "bibliographic entries are not translated.":
        "參考文獻清單即上方英文部分末尾所列；書目資料不予翻譯。",
    "Reproducibility": "可重現性",
    "SOMTrack version": "SOMTrack 版本",
    "Python": "Python",
    "Platform": "平台",
    "Config file": "設定檔",
    "Generated": "產生時間",
    "Re-running `somtrack run` with the saved `analysis_config.json` reproduces every "
    "number above.": "以儲存的 `analysis_config.json` 重新執行 `somtrack run`，即可重現上述所有數值。",
    "Generated by SOMTrack {version} on {date}.": "由 SOMTrack {version} 於 {date} 產生。",
    "Figures were produced with matplotlib; PDF and SVG output embeds text as "
    "editable text (TrueType, fonttype 42).":
        "圖表以 matplotlib 製作；PDF 與 SVG 輸出會將文字嵌入為可編輯文字（TrueType，fonttype 42）。",

    # ======================================================================
    # Metric catalogue: categories, names, descriptions
    # ======================================================================
    "Basic kinematics": "基本運動學",
    "Angular / directional": "角度／方向",
    "Path geometry": "路徑幾何",
    "Diffusion (MSD)": "擴散（MSD）",
    "Persistence & memory": "持續性與記憶",
    "Intermittency (CTRW)": "間歇性（CTRW）",
    "Speed distribution": "速度分布",
    "Oscillation & biophysics": "振盪與生物物理",
    "Position": "位置",
    "Morphology / intensity": "形態／強度",

    "Track length": "軌跡長度",
    "Number of detections in the track.": "軌跡中的偵測點數。",
    "Track duration": "軌跡持續時間",
    "Elapsed time from first to last detection.": "從第一個到最後一個偵測點所經過的時間。",
    "Mean speed": "平均速度",
    "Trimmed mean of instantaneous speed.": "瞬時速度的截尾平均。",
    "Speed SD": "速度標準差",
    "Trimmed SD of instantaneous speed.": "瞬時速度的截尾標準差。",
    "Mean acceleration": "平均加速度",
    "Trimmed mean of the signed tangential acceleration.": "帶正負號切向加速度的截尾平均。",
    "Acceleration SD": "加速度標準差",
    "Trimmed SD of tangential acceleration.": "切向加速度的截尾標準差。",
    "Path length": "路徑長度",
    "Total distance travelled along the track.": "沿軌跡移動的總距離。",
    "Mean angular velocity": "平均角速度",
    "Trimmed mean of the signed turning rate.": "帶正負號轉向速率的截尾平均。",
    "Angular velocity SD": "角速度標準差",
    "Trimmed SD of the turning rate.": "轉向速率的截尾標準差。",
    "Mean angular acceleration": "平均角加速度",
    "Trimmed mean of the change in turning rate.": "轉向速率變化量的截尾平均。",
    "Angular acceleration SD": "角加速度標準差",
    "Trimmed SD of angular acceleration.": "角加速度的截尾標準差。",
    "Mean absolute meander": "平均絕對蜿蜒度",
    "Turning per unit path length (curvature proxy).": "每單位路徑長度的轉向量（曲率的替代指標）。",
    "Meander SD": "蜿蜒度標準差",
    "SD of turning per unit path length.": "每單位路徑長度轉向量的標準差。",
    "Mean heading": "平均行進方向",
    "Circular mean of step headings (replaces the linear angle average).":
        "各步行進方向的圓形平均（取代線性角度平均）。",
    "Directional concentration R": "方向集中度 R",
    "Mean resultant length of headings; 1 = perfectly straight bearing.":
        "行進方向的平均合成向量長度；1 = 方位完全筆直。",
    "Heading circular SD": "行進方向圓形標準差",
    "Circular standard deviation of step headings.": "各步行進方向的圓形標準差。",
    "Rayleigh p (directionality)": "Rayleigh p（方向性）",
    "Rayleigh test against a uniform heading distribution.": "以 Rayleigh 檢定對照均勻的行進方向分布。",
    "Mean |turn angle|": "平均 |轉向角|",
    "Mean absolute turn between consecutive steps.": "相鄰兩步之間轉向角絕對值的平均。",
    "Turn bias (L/R)": "轉向偏好（左／右）",
    "Mean signed turn; non-zero indicates a circling bias.": "帶正負號轉向角的平均；非零表示有繞圈傾向。",
    "Turn-angle kurtosis": "轉向角峰度",
    "Peakedness of the turn distribution; high = long straight runs with rare sharp "
    "turns.": "轉向分布的尖峰程度；高 = 長距離直行，偶爾急轉。",
    "Turn-angle entropy": "轉向角熵",
    "Shannon entropy of the binned turn distribution; low = stereotyped turning.":
        "分箱後轉向分布的 Shannon 熵；低 = 轉向模式刻板。",
    "Net displacement": "淨位移",
    "Straight-line distance from first to last position.": "從第一個到最後一個位置的直線距離。",
    "Straightness index": "直線度指數",
    "Net displacement / path length (confinement ratio).": "淨位移／路徑長度（侷限比）。",
    "Tortuosity": "曲折度",
    "Path length / net displacement.": "路徑長度／淨位移。",
    "Sinuosity": "蜿蜒指數",
    "Bovet & Benhamou sinuosity index of the correlated random walk.":
        "相關隨機漫步的 Bovet & Benhamou 蜿蜒指數。",
    "Radius of gyration": "迴轉半徑",
    "RMS distance of the track from its own centroid.": "軌跡相對於其自身質心的均方根距離。",
    "Convex hull area": "凸包面積",
    "Area explored, as the convex hull of all positions.": "探索的面積，以所有位置的凸包計算。",
    "Convex hull perimeter": "凸包周長",
    "Perimeter of the explored area.": "探索區域的周長。",
    "Exploration ratio": "探索比",
    "Hull area / (pi * Rg^2); 1 = isotropic filling, <1 = elongated or reused path.":
        "凸包面積／(pi * Rg^2)；1 = 各向均勻填滿，<1 = 路徑細長或重複使用。",
    "Bounding-box aspect": "外接框長寬比",
    "Long/short axis ratio of the PCA-aligned bounding box.": "以 PCA 對齊之外接框的長軸／短軸比。",
    "Katz fractal dimension": "Katz 碎形維度",
    "Path complexity; 1 = straight line, higher = more convoluted.":
        "路徑複雜度；1 = 直線，越高越曲折。",
    "MSD anomalous exponent": "MSD 異常擴散指數",
    "Slope of log MSD vs log lag: 1 = Brownian, >1 super-diffusive, <1 confined.":
        "log MSD 對 log 時間延遲的斜率：1 = 布朗運動，>1 超擴散，<1 受侷限。",
    "Generalised diffusion coefficient": "廣義擴散係數",
    "Prefactor of the MSD power-law fit.": "MSD 冪律擬合的前置係數。",
    "MSD fit R^2": "MSD 擬合 R^2",
    "Goodness of the MSD power-law fit; low values invalidate alpha and D.":
        "MSD 冪律擬合的適配度；數值低時 alpha 與 D 不可採信。",
    "Directional correlation time": "方向相關時間",
    "Lag at which <cos(delta heading)> falls to 1/e.": "<cos(行進方向變化量)> 降至 1/e 時的時間延遲。",
    "Persistence length": "持續長度",
    "Directional correlation time x mean speed.": "方向相關時間 × 平均速度。",
    "Speed correlation time": "速度相關時間",
    "Lag at which the speed autocorrelation falls to 1/e.": "速度自相關降至 1/e 時的時間延遲。",
    "Speed Hurst exponent (DFA)": "速度 Hurst 指數（DFA）",
    "Detrended fluctuation exponent of the speed series; >0.5 = persistent bursts.":
        "速度序列的去趨勢波動指數；>0.5 = 持續性的爆發。",
    "Mean halt duration": "平均停頓時間",
    "Mean accumulated pause time (legacy definition).": "累積停頓時間的平均（舊版定義）。",
    "Halt duration SD": "停頓時間標準差",
    "SD of accumulated pause time.": "累積停頓時間的標準差。",
    "Fraction of time moving": "移動時間比例",
    "Proportion of frames above the halt speed threshold.": "速度高於停頓門檻的影格比例。",
    "Stop frequency": "停止頻率",
    "Number of move-to-pause transitions per unit time.": "每單位時間由移動轉為停頓的次數。",
    "Mean run duration": "平均移動持續時間",
    "Mean length of an uninterrupted moving bout.": "一段不間斷移動的平均長度。",
    "Mean pause duration": "平均停頓持續時間",
    "Mean length of an uninterrupted pause bout.": "一段不間斷停頓的平均長度。",
    "Burstiness B": "爆發性 B",
    "Goh-Barabasi burstiness of run durations; >0 = bursty, <0 = regular.":
        "移動持續時間的 Goh-Barabasi 爆發性；>0 = 爆發式，<0 = 規律。",
    "CTRW memory M": "CTRW 記憶 M",
    "Lag-1 correlation of consecutive run durations.": "相鄰移動持續時間的 lag-1 相關。",
    "Peak speed": "峰值速度",
    "95th-percentile speed, robust to tracking spikes.": "第 95 百分位數速度，不受追蹤突波影響。",
    "Speed CV": "速度變異係數",
    "Coefficient of variation of speed.": "速度的變異係數。",
    "Speed skewness": "速度偏態",
    "Asymmetry of the speed distribution.": "速度分布的不對稱程度。",
    "RMS acceleration": "均方根加速度",
    "Root-mean-square tangential acceleration (effort proxy).": "切向加速度的均方根（出力的替代指標）。",
    "RMS jerk": "均方根急動度",
    "Root-mean-square rate of change of acceleration (smoothness proxy).":
        "加速度變化率的均方根（平順度的替代指標）。",
    "Turning beat frequency": "轉向擺動頻率",
    "Dominant frequency of the angular-velocity series (tail/cilia beat proxy).":
        "角速度序列的主頻率（尾部／纖毛擺動的替代指標）。",
    "Beat spectral purity": "擺動頻譜純度",
    "Power at the dominant frequency / total power; high = regular rhythmic beating.":
        "主頻率功率／總功率；高 = 規律的節律性擺動。",
    "Speed oscillation frequency": "速度振盪頻率",
    "Dominant frequency of the speed series (stroke cycle proxy).":
        "速度序列的主頻率（划動週期的替代指標）。",
    "Lateral oscillation amplitude": "側向振盪振幅",
    "RMS deviation of the path from its smoothed centre line.": "路徑偏離其平滑中心線的均方根。",
    "Strouhal number": "Strouhal 數",
    "f x 2A / U; efficient undulatory swimming sits near 0.2-0.4.":
        "f × 2A / U；有效率的波動式游泳約落在 0.2–0.4。",
    "Reynolds number": "Reynolds 數",
    "rho x U x L / mu in water at 25 C; needs a body-length column or estimated "
    "diameter.": "25 °C 水中的 rho × U × L / mu；需要體長欄位或估計直徑。",
    "Speed in body lengths": "以體長計的速度",
    "Mean speed normalised by body length.": "以體長標準化的平均速度。",
    "Mean X (relative)": "平均 X（相對）",
    "Mean X position relative to the track's own minimum.": "相對於該軌跡自身最小值的平均 X 位置。",
    "Mean Y (relative)": "平均 Y（相對）",
    "Mean Y position relative to the track's own minimum.": "相對於該軌跡自身最小值的平均 Y 位置。",
    "XY dispersion": "XY 離散度",
    "Geometric mean of the X and Y positional SDs.": "X 與 Y 位置標準差的幾何平均。",
    "Mean area": "平均面積",
    "Mean segmented body area.": "分割出之身體面積的平均。",
    "Area SD": "面積標準差",
    "SD of body area (shape change / rotation proxy).": "身體面積的標準差（形狀變化／旋轉的替代指標）。",
    "Mean major axis": "平均長軸",
    "Mean fitted-ellipse major axis.": "擬合橢圓長軸的平均。",
    "Major axis SD": "長軸標準差",
    "SD of the major axis.": "長軸的標準差。",
    "Mean minor axis": "平均短軸",
    "Mean fitted-ellipse minor axis.": "擬合橢圓短軸的平均。",
    "Minor axis SD": "短軸標準差",
    "SD of the minor axis.": "短軸的標準差。",
    "Body aspect ratio": "身體長寬比",
    "Major / minor axis; elongation of the body.": "長軸／短軸；身體的細長程度。",
    "Aspect ratio CV": "長寬比變異係數",
    "Variability of elongation, i.e. body deformation during swimming.":
        "細長程度的變異，即游泳時的身體形變。",
    "Mean body orientation": "平均身體方向",
    "Circular mean of the fitted-ellipse orientation.": "擬合橢圓方向的圓形平均。",
    "Body orientation circular SD": "身體方向圓形標準差",
    "Circular SD of body orientation.": "身體方向的圓形標準差。",
    "Mean yaw drift": "平均偏航漂移",
    "Mean angle between body axis and direction of travel (slip / crabbing).":
        "身體軸與行進方向之間夾角的平均（側滑／斜行）。",
    "Yaw drift SD": "偏航漂移標準差",
    "SD of the body-axis / travel-direction angle.": "身體軸與行進方向夾角的標準差。",
    "Mean intensity": "平均強度",
    "Mean particle intensity.": "粒子強度的平均。",
    "Intensity SD": "強度標準差",
    "SD of particle intensity.": "粒子強度的標準差。",
    "Mean focus index": "平均對焦指數",
    "Per-track normalised intensity SD; a relative depth-of-focus proxy, not a "
    "calibrated Z.": "各軌跡標準化後的強度標準差；為相對景深的替代指標，並非校正過的 Z 值。",
    "Focus index SD": "對焦指數標準差",
    "Variability of the focus index, i.e. vertical excursion proxy.":
        "對焦指數的變異，即垂直位移的替代指標。",
}

# Short words that translate differently depending on where they appear.  In
# every one of these the English word is also a configuration value, so the
# translation carries it in parentheses: what the user reads can be matched to
# what the methods section and analysis_config.json record.
CONTEXTS: dict[str, dict[str, str]] = {
    # parameter choices from the projection registry, and the distance setting
    "choice": {
        "euclidean": "歐氏距離（euclidean）",
        "correlation": "相關距離（correlation）",
        "cityblock": "曼哈頓距離（cityblock）",
        "manhattan": "曼哈頓距離（manhattan）",
        "cosine": "餘弦距離（cosine）",
        "classical": "古典尺度法（classical）",
        "smacof": "SMACOF 迭代法（smacof）",
        "pca": "由 PCA 起始（pca）",
        "random": "隨機（random）",
        "rbf": "徑向基底函數（rbf）",
        "poly": "多項式（poly）",
        "sigmoid": "S 型函數（sigmoid）",
    },
    "scaler": {
        "zscore": "z 分數標準化（zscore）",
        "minmax": "最小-最大縮放（minmax）",
        "robust": "穩健縮放（robust）",
        "rank": "秩轉換（rank）",
        "none": "不縮放（none）",
    },
    "nan_policy": {
        "impute_median": "以中位數填補（impute_median）",
        "drop_sample": "移除含缺失值的樣本（drop_sample）",
        "drop_feature": "移除含缺失值的指標（drop_feature）",
    },
    "som_algorithm": {
        "batch": "批次（batch）",
        "supervised": "監督式（supervised）",
        "relevance": "相關性學習（relevance）",
        "online": "線上逐筆（online）",
        "growing": "成長型（growing）",
    },
    "lattice": {
        "hex": "六角形（hex）",
        "rect": "矩形（rect）",
    },
    "som_init": {
        "pca": "PCA（pca）",
        "sample": "從樣本抽取（sample）",
        "random": "隨機（random）",
    },
    "node_cluster": {
        "kmeans": "k-means 分群（kmeans）",
        "ward": "Ward 階層分群（ward）",
        "gmm": "高斯混合模型（gmm）",
    },
    "palette": {
        "somtrack": "somtrack（Okabe-Ito 色盲友善）",
        "tab10": "tab10",
        "legacy_hsb": "legacy_hsb（v1.2 色相環）",
    },
    "metric_preset": {
        "All": "全部",
        "Recommended": "建議組合",
        "v1.2 legacy set": "v1.2 舊版指標組",
        "Kinematics only": "僅運動學",
        "None": "全不選",
    },
}
