# RBL Report — Programming Knowledge Tracing (IEEE Conference / LaTeX)

Khung báo cáo LaTeX theo template **IEEE conference (IEEEtran)** cho đề tài:
*Programming Knowledge Tracing with Knowledge-Component and Code-Structure
Signals: Does Predictive Accuracy Translate into Better Exercise Sequencing?*

## 1. Cấu trúc thư mục

```
report/
├── main.tex                 # File chính — compile file này
├── references.bib           # Danh mục tài liệu tham khảo (BibTeX)
├── README.md                # File này
├── figures/                 # Chứa hình (accuracy_vs_utility.pdf, ...)
└── sections/
    ├── 00_abstract.tex
    ├── 01_introduction.tex        # Vấn đề, RQ (a)+(b), đóng góp
    ├── 02_related_work.tex        # BKT/DKT/SAKT/AKT/simpleKT; Code-DKT/KCGen-KT/TIKTOC
    ├── 03_datasets.tex            # CodeWorkout, FalconCode, ASSISTments
    ├── 04_methodology.tex         # RO1 pipeline + RO2 mô hình đề xuất
    ├── 05_experiments.tex         # Setup, hyperparameters, protocol
    ├── 06_results_discussion.tex  # RO3 accuracy, RO4 sequencing+correlation, RO5 transfer
    ├── 07_threats.tex             # Threats to validity (KT-as-simulator ...)
    └── 08_conclusion.tex          # Kết luận + AI usage declaration
```

## 2. Compile trên Overleaf (khuyến nghị)

1. Nén cả thư mục `report/` thành `.zip`.
2. Overleaf → **New Project → Upload Project** → chọn file `.zip`.
3. Menu (góc trái) → **Settings**:
   - **Compiler**: `pdfLaTeX`
   - **Main document**: `main.tex`
   - **TeX Live version**: mặc định (mới nhất) là được.
4. Bấm **Recompile**. Overleaf tự chạy BibTeX nên trích dẫn `\cite{}` sẽ hiện đúng sau 1–2 lần recompile.

> IEEEtran là class có sẵn trên Overleaf, không cần cài thêm.

## 3. Compile ở máy (nếu không dùng Overleaf)

Cần cài TeX Live / MiKTeX, rồi chạy trong thư mục `report/`:

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

(Chạy `pdflatex` 2 lần cuối để cập nhật số trích dẫn và tham chiếu chéo.)

## 4. Quy trình viết — map vào Research Objectives

| Section | Research Objective | Việc cần làm |
|---------|--------------------|--------------|
| 04 Methodology (Pipeline) | RO1 | Mô tả pipeline pyKT + baselines DKT/SAKT/AKT/simpleKT |
| 04 Methodology (Model)    | RO2 | Chốt kiến trúc: fusion KC + code-structure (AST) |
| 06 Results (Table pred.)  | RO3 | Điền AUC/RMSE/ACC |
| 06 Results (Seq. + corr.) | RO4 | Điền sequencing utility + hệ số tương quan (đóng góp lõi) |
| 06 Results (Transfer)     | RO5 | Điền cross-dataset + cross-domain |

## 5. Việc còn phải làm (tìm `\todo{}` trong file)

Các chỗ cần điền được đánh dấu bằng `\todo{...}` (hiện màu đỏ khi compile).
Tìm nhanh trong Overleaf bằng ô Search hoặc grep `\todo`. Chính:
- Số liệu dataset (số học viên, số interaction, số KC) trong `03_datasets.tex`.
- Kiến trúc fusion và cách trích xuất AST trong `04_methodology.tex`.
- Hyperparameters trong `05_experiments.tex`.
- Toàn bộ bảng kết quả trong `06_results_discussion.tex`.
- Hình `figures/accuracy_vs_utility.pdf` (đang comment trong `06_...`, mở ra khi có hình).
- Khai báo AI usage trong `08_conclusion.tex`.

> **Trước khi nộp**: bỏ 2 lệnh helper `\todo`/`\note` trong `main.tex` (hoặc
> định nghĩa lại cho chúng in ra rỗng) để bản cuối không còn chữ đỏ.

## 6. Kiểm tra `references.bib`

Một số entry (`pktattn2021`, `tiktoc2025`, `kcgenkt2026`, `helpdkt`,
`falconcode2023`) đang để tác giả/venue tạm (`Anonymous` / `verify`).
Cần đối chiếu bài gốc và điền đầy đủ author, venue, năm, trang, DOI trước khi nộp.

## 7. Nguồn code & dataset tham khảo (cho phần thực nghiệm)

- **pyKT toolkit** (baselines DKT/SAKT/AKT/simpleKT): https://github.com/pykt-team/pykt-toolkit
- **Code-DKT**: tìm "Code-DKT YangAzure github"
- **KCGen-KT**: https://github.com/umass-ml4ed/kcgen-kt
- **TIKTOC**: https://github.com/umass-ml4ed/tiktoc
- **CodeWorkout (CSEDM 2021)**: https://sites.google.com/ncsu.edu/csedm-dc-2021/dataset
- **FalconCode**: https://huggingface.co/datasets/koutch/falcon_code
