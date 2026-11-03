# 🕶️ ESPNER Colab T4 Luxury Commercial Video Studio (`v5.2`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/btcson66-rgb/espner-colab-t4-video-studio/blob/main/ESPNER_Colab_T4_%E7%9C%81%E8%A8%98%E6%86%B6%E9%AB%94%E7%89%88_v5.1.ipynb)

專為 **標準 Google Colab T4 GPU (`15 GB VRAM` / `12.7 GB System RAM`)** 打造的 **ESPNER 奢華太陽眼鏡與滑雪護目鏡 9:16 直式廣告自動生成系統**（免開付費 High-RAM）。

只要上傳一張**產品正面照片**（或選用內建測試圖），即可自動完成：
1. **階段 3A（純 OpenCV 零記憶體去背 + 真人模特兒配戴首幀合成）**：
   - 使用 OpenCV FloodFill + 亮度柔和邊緣去背演算法（`0 MB` 額外 RAM、免 `rembg`）。
   - 支援中文自訂場景描述（自動翻譯為好萊塢級英文導演 Prompt）。
   - 透過 `RealVisXL V4.0 Lightning` 生成無眼鏡真人時尚肖像，再將 **100% 原始上傳產品正面圖** 依據人臉與雙眼座標精準配戴合成（附加真實鼻樑光學陰影）。
2. **階段 3B（拆離式文字編碼器 `transformer=None, vae=None`）**：
   - 單獨將文字編碼器載入 GPU（自動偵測 T4 `sm_75` 使用原生 `torch.float16`），算完 `prompt_embeds.pt` 後立即退出子程序釋放 ~9.5 GB 記憶體。
3. **階段 3C（無文字編碼器 `text_encoder=None` 圖生影片推論）**：
   - 支援 **方案 1A (`LTX-Video-2B`)**、**方案 1B (`Wan2.2-TI2V-5B`)**、**方案 1C (`SVD-XT`)** 與 **方案 2（純產品 3D/2.5D 刀光立體運鏡）**。
   - 針對 Tesla T4 (`Compute Capability 7.5`) 強制啟用原生 `torch.float16` 與 `enable_model_cpu_offload()`，徹底解決 `bfloat16` 在 T4 上觸發 `cutlassF: no kernel found to launch!` 的問題。
4. **步驟 4（ESPNER v5 正式版後製管線）**：
   - 內建 6 大視覺風格（含 `0_CLASSIC_MADRID` 2.0s 金框石牆過場、`1_CYBERPUNK_NEON`、`2_VINTAGE_1985`、`3_HAUTE_COUTURE_NOIR`、`4_ALPINE_EXTREME`、`5_RIVIERA_SUNSET`）。
   - 片尾 4.5s~5.0s 純白 `#FFFFFF` 攝影棚正面原圖壓軸定格（`1.000 -> 1.035` 微推、底部柔和接觸陰影、`E S P N E R` 品牌字標、`把目光，還給世界。`）。

---

## 📁 專案結構

- `ESPNER_Colab_T4_省記憶體版_v5.1.ipynb`：主 Colab 筆記本（含 T4 `float16` 修正與即時子程序日誌串流）。
- `build_colab_notebook.py`：生成並同步 `.ipynb` 至本機與 Google Drive (`G:\我的雲端硬碟\Colab Notebooks\`) 的建置腳本。
- `espner_colab_t4_pipeline.py`：可於 CLI 獨立執行的完整管線腳本。
- `sample_products/`：內建 3 款官方正面測試圖 (`AVIATOR_front.jpg`, `MADRID_front.jpg`, `ZERMATT_front.jpg`)。
- `ESPNER_Blender廣告製作_20261031/`：本機 v5 正式版母帶後製腳本參考實作。
- `CLAUDE.md`：供 Claude Code 與 Antigravity 協同開發的架構規範與注意事項。
