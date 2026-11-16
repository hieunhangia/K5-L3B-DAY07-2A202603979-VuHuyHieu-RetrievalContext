# CHECKLIST THỰC HIỆN BÀI TẬP DAY 7 — AGENTIC RAG
> **Dự án:** Agentic RAG — Retrieval Context Engineering  
> **Sinh viên:** Vũ Huy Hiệu — **MSSV:** 2A202603979  
> **Lớp:** K5-L3B  
> **Mục tiêu điểm số:** 85/85 (Phần cơ bản) + 15/15 (Thử thách nâng cao bonus) = **100/100 điểm**  
> **Quy định bắt buộc:** Python 3.11+, dùng model thật và vector store thật, KHÔNG đưa gold vào retrieval/agent, KHÔNG commit API key/.env, chốt cấu hình trên dev TRƯỚC KHI chạy test.

---

## 📅 Lộ trình thực hiện tổng quan (Timeline)

```text
[Giai đoạn 0: Setup] ➔ [Giai đoạn 1: Dữ liệu & Benchmark] ➔ [Giai đoạn 2: 3 Phương pháp Retrieval]
         │
         ▼
[Giai đoạn 3: Agentic Loop & Context] ➔ [Giai đoạn 4: Thử nghiệm Dev & Cải tiến]
         │
         ▼
[Giai đoạn 5: Đánh giá độc lập Test] ➔ [Giai đoạn 6: Báo cáo & Nộp bài]
         │
         ▼ (Tùy chọn)
[Giai đoạn 7: Thử thách nâng cao 100+ trang (+15 điểm bonus)]
```

---

## GIAI ĐOẠN 0: THIẾT LẬP MÔI TRƯỜNG & KHỞI TẠO REPO
*Mục tiêu: Đảm bảo môi trường sạch, mã nguồn chạy thông suốt, cấu trúc Git đúng quy định.*

- [x] **0.1. Kiểm tra cấu hình Git và Remote cá nhân**
  - [x] Kiểm tra tên thư mục repo: `K5-L3B-DAY07-2A202603979-VuHuyHieu-RetrievalContext`.
  - [x] Kiểm tra remote repo cá nhân: `origin` trỏ về `https://github.com/hieunhangia/K5-L3B-DAY07-2A202603979-VuHuyHieu-RetrievalContext.git`.
  - [x] Kiểm tra branch chính: `main`.
- [x] **0.2. Cài đặt và kiểm tra môi trường Python**
  - [x] Đảm bảo phiên bản Python >= 3.11 (`python3 --version` -> Python 3.14.10).
  - [x] Khởi tạo virtual environment: `.venv`.
  - [x] Cài đặt các thư viện cần thiết (`requirements.txt`): `openai`, `google-genai`, `google-generativeai`, `rank_bm25`, `numpy`, `scikit-learn`, `python-dotenv`, `tqdm`.
  - [x] Tạo file mẫu `.env.example` và cấu hình module `src/config.py` hỗ trợ OpenAI Chat & Google GenAI Embedding. Đảm bảo `.env` được `.gitignore` bảo vệ.
- [x] **0.3. Chạy kiểm tra bộ unit test mock mẫu**
  - [x] Chạy lệnh: `.venv/bin/python3 -m unittest discover -s tests -v`.
  - [x] Đảm bảo toàn bộ 15 test cases trong 4 test suites đều pass (OK).

---

## GIAI ĐOẠN 1: DỮ LIỆU DAY 6 & XÂY DỰNG BỘ BENCHMARK (DEV / TEST)
*Trọng số Rubric: 13 điểm (Dữ liệu & Benchmark)*  
*Mục tiêu: Chuẩn bị kho dữ liệu chunk sạch và bộ câu hỏi đánh giá đạt chuẩn tối thiểu 24 câu.*

- [x] **1.1. Chuẩn bị Corpus dữ liệu Day 6**
  - [x] Xuất kho dữ liệu từ Day 6 ra file JSON đúng cấu trúc: `id`, `text`, `source_id`, `source`, `page`, `metadata`.
  - [x] Đảm bảo mỗi chunk có `id` duy nhất; tài liệu gốc có độ dài hợp lệ (32 trang nội dung hữu ích từ Quy chế Đào tạo VinUniversity v8.0).
  - [x] Lưu file dữ liệu vào: `data/chunks.json` (105 chunks theo Chiến lược C Paragraph-aware).
  - [x] Ghi lại thông tin: Nguồn tài liệu (VinUniversity), quyền sử dụng, số trang (35 vật lý, 32 nội dung), số chunk (105), embedding model (Google GenAI `gemini-embedding-2` / `text-embedding-004`, 3072 chiều).
- [x] **1.2. Xây dựng Bộ câu hỏi Benchmark (Tối thiểu 24 câu)**
  - [x] Tạo file benchmark câu hỏi: `data/questions.json`, `data/all_questions.json`, `data/unanswerable_questions.json`, và `data/agent_cases.json`.
  - [x] **Phân chia tập Dev (17 câu):**
    - [x] **15 câu có đáp án (answerable = true):** `q01`–`q15`, đầy đủ `gold: [{source_id, page, evidence}]`, `expected_answer`, `filter`. Đã chuẩn hóa và kiểm chứng 100% evidence nằm trọn trong chunk đúng nguồn và trang.
    - [x] **2 câu thiếu bằng chứng (unanswerable / answerable = false):** `q_unans_dev_01`, `q_unans_dev_02` gần chủ đề (phí gửi xe KTX, sinh hoạt phí trao đổi quốc tế), kèm `absence_reason` và `expected_behavior: "search_and_abstain"`.
  - [x] **Phân chia tập Test (7 câu độc lập):**
    - [x] **5 câu có đáp án (answerable = true):** `q16`–`q20`, đầy đủ `gold`, `expected_answer`. Kiểm chứng 100% evidence khớp văn bản chunk.
    - [x] **2 câu thiếu bằng chứng (answerable = false):** `q_unans_test_01`, `q_unans_test_02` (giờ giới nghiêm KTX, trợ cấp học bổng tài năng).
  - [x] **Quy tắc vàng:** TUYỆT ĐỐI KHÔNG đưa nội dung `gold`, `expected_answer` vào context của agent hoặc query retrieval!

---

## GIAI ĐOẠN 2: HIỆN THỰC ÍT NHẤT 3 PHƯƠNG PHÁP RETRIEVAL
*Trọng số Rubric: 17 điểm (Retrieval)*  
*Mục tiêu: Xây dựng 3 kỹ thuật truy xuất dữ liệu độc lập, phân biệt rõ về bản chất thuật toán.*

- [x] **2.1. Phương pháp 1: Dense Retrieval (Bắt buộc)**
  - [x] Kết nối vector store và Google GenAI embedding (`gemini-embedding-2` / `text-embedding-004`, 3072 chiều) tương thích Day 6.
  - [x] Đã embed toàn bộ 105 chunks và lưu cache tại `artifacts/embeddings_cache.json`.
  - [x] Tính toán Cosine Similarity chuẩn hóa, hỗ trợ metadata filter (`section`, `page`, v.v.).
  - [x] Chuẩn hóa output: trả về danh sách dict gồm `id`, `text`, `source_id`, `source`, `page`, `metadata`, `score`.
- [x] **2.2. Phương pháp 2: Lexical Retrieval**
  - [x] Triển khai BM25Okapi (`rank_bm25`) kết hợp bộ tách từ tiếng Việt chuẩn Unicode NFC (`tokenize_vi`).
  - [x] Hỗ trợ metadata filter và chuẩn hóa output đúng schema thống nhất.
- [x] **2.3. Phương pháp 3: Phương pháp Nâng cao tự chọn (Hybrid Retrieval)**
  - [x] Triển khai **Hybrid Retrieval (Dense + BM25 kết hợp RRF - Reciprocal Rank Fusion)**:
    - Thu hồi ứng viên từ cả Dense search và BM25 search.
    - Hợp nhất xếp hạng theo công thức: $RRF(d) = \sum \frac{1}{60 + rank(d)}$.
- [x] **2.4. Kiểm tra chuẩn hóa Output & Đánh giá so sánh**
  - [x] Cả 3 phương pháp trả về đúng top_k chunk duy nhất (không trùng lặp ID).
  - [x] Đã chạy benchmark đánh giá cả 3 phương pháp trên Dev (15 câu) và Test (5 câu) qua `src/evaluate_retrieval.py`.
  - [x] Lưu kết quả và trace tại `results/retrieval_dense.json`, `results/retrieval_lexical.json`, `results/retrieval_hybrid.json`, và `results/retrieval_comparison.json`.
  - [x] Viết unit tests hoàn chỉnh trong `tests/test_retrieval.py` (20/20 test suites pass).

---

## GIAI ĐOẠN 3: THIẾT KẾ AGENTIC LOOP, TOOL CALLING & CONTEXT PACKING
*Trọng số Rubric: 17 điểm (Agent và Context)*  
*Mục tiêu: Agent dùng LLM thật điều phối truy xuất, quản lý context budget, trích dẫn chính xác.*

- [x] **3.1. Xây dựng Agent Controller với LLM thật**
  - [x] Kết nối LLM thật qua OpenAI Client tương thích mô hình người dùng cấu hình (`openai/grok-4.5-free` via `https://api.xah.io/v1/`).
  - [x] Định nghĩa công cụ Function Calling chuẩn: tool `search(query)` và tool `answer(answer, citations)`.
  - [x] Prompt hệ thống chuẩn mực: biết phân tích câu hỏi, điều phối multi-turn retrieval, trích dẫn chuẩn xác `[chunk-id]`, và nhận biết từ chối dứt khoát (*"Không đủ bằng chứng để trả lời."*) khi thiếu bằng chứng.
- [x] **3.2. Quản lý Context Budget và Đóng gói Context (Context Packing)**
  - [x] Quản lý ngân sách context động (`pack_context`), hỗ trợ khử trùng lặp (deduplication) giữa các lần tìm kiếm.
  - [x] Bảo toàn thông tin nguồn gốc (`source_id`, `page`) trong header của từng khối chunk trong context.
- [x] **3.3. Thu thập Trace chi tiết của từng lượt chạy (Logging)**
  - [x] Ghi lại toàn diện: `query_id`, `status`, `question`, `answer`, `citations`, các `events` theo từng turn, thời gian thực thi `elapsed_ms`.
  - [x] Xây dựng script thực thi kiểm thử agent hàng loạt: `src/run_agent_eval.py`.
  - [x] Bổ sung unit test toàn diện tại `tests/test_agent_runner.py` (toàn bộ 23/23 unit tests pass).
  - [x] Kiểm thử thực tế thành công: Trả lời chính xác có trích dẫn với câu có đáp án, và từ chối dứt khoát 100% với câu thiếu evidence.

---

## GIAI ĐOẠN 4: THỬ NGHIỆM TRÊN DEV, PHÂN TÍCH LỖI & TỐI ƯU CẤU HÌNH
*Trọng số Rubric: 13 điểm (Thử nghiệm và Cải thiện)*  
*Mục tiêu: Đánh giá so sánh 3 phương pháp trên Dev, thử 2 cấu hình cải tiến, phân tích lỗi sâu và chốt cấu hình tối ưu.*

- [x] **4.1. Chạy đánh giá Baseline và 3 phương pháp Retrieval trên tập Dev**
  - [x] Chạy Phương pháp 1 (Dense Baseline, budget=800) trên toàn bộ 15 câu Dev có đáp án: Hit@1 = 10/15 (66.7%), Hit@3 = 15/15 (100.0%), Context Evidence Rate = 1/15 (6.7%).
  - [x] Chạy Phương pháp 2 (Lexical BM25, budget=800) trên Dev: Hit@1 = 12/15 (80.0%), Hit@3 = 15/15 (100.0%), Context Evidence Rate = 1/15 (6.7%).
  - [x] Chạy Phương pháp 3 (Hybrid RRF k=60, budget=800) trên Dev: Hit@1 = 12/15 (80.0%), Hit@3 = 15/15 (100.0%), Context Evidence Rate = 1/15 (6.7%).
  - [x] Đo và ghi nhận các chỉ số Retrieval:
    - **Hit@1:** Tỷ lệ câu có evidence hợp lệ nằm trong top 1 kết quả / tổng số câu có đáp án.
    - **Evidence Hit@3:** Tỷ lệ câu có evidence hợp lệ nằm trong top 3 kết quả / tổng số câu có đáp án.
    - **Context Evidence Rate:** Tỷ lệ câu có đủ evidence hợp lệ trong context cuối cùng gửi đến model / tổng số câu có đáp án.
- [x] **4.2. Thử nghiệm ít nhất 2 Cấu hình Cải tiến (Improvement Experiments) trên Dev**
  - [x] **Cấu hình Cải tiến 1: Context Budget Expansion (budget=2000, Hybrid RRF k=60)**
    - [x] **Giả thuyết kỹ thuật (Hypothesis):** Ngân sách mặc định 800 ký tự không đủ chứa nhiều hơn 1 chunk (chunk trung bình ~622 chars + header ~80 chars). Khi chunk Rank 1 không chứa trọn vẹn evidence hoặc evidence nằm ở Rank 2, 3 thì `pack_context` cắt bỏ, làm sụt giảm nghiêm trọng Context Evidence Rate (chỉ 6.7%). Tăng ngân sách lên 2000 ký tự sẽ giữ được trọn vẹn top 2–3 chunks trong prompt.
    - [x] **Kết quả thực nghiệm:** Hit@1 = 12/15 (80.0%), Hit@3 = 15/15 (100.0%), **Context Evidence Rate tăng vọt từ 6.7% lên 15/15 (100.0%)**! Median Latency: 0.56 ms.
  - [x] **Cấu hình Cải tiến 2: Weighted Hybrid RRF with Lexical Boost (k=20, w_bm25=0.70, w_dense=0.30, budget=2000)**
    - [x] **Giả thuyết kỹ thuật:** Mô hình Dense Retrieval bị "Semantic Attraction" đối với các câu hỏi quy chế có từ khóa kỹ thuật chuyên sâu (ví dụ: "bằng kép", "rút học phần", "giờ tự học"), bị hút về trang khái niệm chung thay vì điều khoản chi tiết. Tăng trọng số BM25 lên 0.70 và giảm hằng số RRF $k$ xuống 20 sẽ ưu tiên các chunk có từ khóa định danh chính xác lên Rank 1.
    - [x] **Kết quả thực nghiệm:** **Hit@1 tăng tiếp từ 12/15 (80.0%) lên 13/15 (86.7%)**, Hit@3 đạt **15/15 (100.0%)**, Context Evidence Rate đạt **15/15 (100.0%)**! Median Latency: 0.56 ms.
- [x] **4.3. Phân tích chi tiết ít nhất 2 ca lỗi (Error Analysis) trên Dev**
  - [x] **Ca lỗi 1 (Error Case 1 - `q01` Điều kiện bằng kép):**
    - [x] **Hiện tượng:** Dense Baseline xếp sai chunk Page 21 vào Rank 1 thay vì chunk Page 4 (`quy_che_dao_tao_vinuni_2024_stratC_p4_c003`), khiến Hit@1 bị fail.
    - [x] **Nguyên nhân gốc rễ (Root cause):** "Semantic Attraction" - Cụm từ "học cùng lúc hai chương trình" xuất hiện lặp lại ở cả trang quy định tốt nghiệp (Page 21) và trang điều kiện đăng ký (Page 4). Dense embedding bị thiên lệch về Page 21 do độ tương đồng văn bản vĩ mô.
    - [x] **Khắc phục & Kiểm chứng:** BM25 xếp Page 4 ở Rank 1 nhờ khớp chính xác từ khóa "được cấp bằng kép". Áp dụng Weighted Hybrid RRF ($w_{bm25}=0.70$) đã đẩy Page 4 lên Rank 1 chung cuộc, sửa hoàn toàn lỗi này.
  - [x] **Ca lỗi 2 (Error Case 2 - `q03` Giờ tự học 1 tín chỉ):**
    - [x] **Hiện tượng:** Ở Baseline với budget=800, cả Dense và BM25 đều đưa chunk Page 9 (`vinuni_qd_028`) vào Rank 2. Tuy nhiên, chunk Rank 1 (Page 8) dài 782 ký tự, khiến tổng ký tự có header là 862 chars > 800 chars. Hàm `pack_context()` ngắt ngay sau chunk 1, làm mất chunk Rank 2. Context gửi cho Agent không có đáp án, dẫn đến Context Evidence Rate = False.
    - [x] **Nguyên nhân gốc rễ:** Context Budget Bottleneck làm rơi rớt thông tin ở Rank 2.
    - [x] **Khắc phục & Kiểm chứng:** Nâng ngân sách lên budget=2000 ký tự. Cả chunk Rank 1 và Rank 2 được đóng gói toàn vẹn (tổng ~1520 chars). Agent đọc đủ dữ liệu và trả lời chính xác "15 giờ học trên lớp và 30 giờ tự học, chuẩn bị cá nhân có hướng dẫn", trích dẫn đúng chunk.
- [x] **4.4. Phân tích Đánh đổi (Trade-offs) và Chốt Cấu hình (Lock Selected Config)**
  - [x] So sánh đánh đổi 3 yếu tố:
    - **Chất lượng:** Cải thiện 2 vượt trội hoàn toàn: Hit@1 = 86.7%, Hit@3 = 100%, Context Evidence Rate = 100% (so với Baseline: 66.7% - 100% - 6.7%).
    - **Thời gian (Latency):** Median Latency của Weighted Hybrid chỉ ~0.56 ms (trên CPU với cache), gần như tương đương Dense (~0.1 ms) và BM25 (~0.43 ms). Độ trễ retrieval hoàn toàn không phải là nút thắt cổ chai so với LLM generation (~5-15s).
    - **Chi phí (Cost):** Tăng budget từ 800 lên 2000 chars làm tăng prompt của LLM thêm ~300 tokens ($0.00006/query), đổi lại độ bao phủ bằng chứng đạt 100%, loại bỏ hoàn toàn hiện tượng ảo giác hay từ chối sai do thiếu context.
  - [x] **Chốt Cấu hình Tối ưu (Selected Configuration):**
    - **Retriever:** `WeightedHybridRetriever(rrf_k=20, pool_size=20, dense_weight=0.30, bm25_weight=0.70)`
    - **Context Packing:** `budget = 2000` ký tự, metadata header đầy đủ `[{id} | {source} | page {page}]`.
    - **Agent Controller:** `OpenAIController` + stateless prompt, tool calling `search` & `answer`, abstention check.
  - [x] **KHÓA CẤU HÌNH (CONFIGURATION LOCKED):** Đóng băng toàn bộ siêu tham số trên, tuyệt đối không chỉnh sửa khi bước sang đánh giá trên Test set ở Giai đoạn 5!

---

## GIAI ĐOẠN 5: ĐÁNH GIÁ ĐỘC LẬP TRÊN TEST SET & CHẤM NHÃN THỦ CÔNG
*Trọng số Rubric: 17 điểm (Đánh giá)*  
*Mục tiêu: Đánh giá khách quan trên Test set cho Baseline và Cấu hình đã chọn, đối chiếu bằng chứng thực tế.*

- [x] **5.1. Chạy đánh giá trên tập Test độc lập**
  - [x] Chạy Baseline (Dense, budget=800) trên 7 câu Test: Hit@1 = 3/5 (60.0%), Hit@3 = 5/5 (100.0%), Context Evidence Rate = 1/5 (20.0%).
  - [x] Chạy Cấu hình đã chọn (Weighted Hybrid, budget=2000) trên 7 câu Test: Hit@1 = 5/5 (100.0%), Hit@3 = 5/5 (100.0%), Context Evidence Rate = 5/5 (100.0%).
  - [x] Lưu toàn bộ trace kết quả vào file JSON: `results/test_traces_baseline.json` và `results/test_traces_selected.json`.
  - [x] Lưu báo cáo đối chiếu retrieval vào `results/stage5_test_retrieval_comparison.json`.
- [x] **5.2. Chấm nhãn thủ công chuyên sâu (Qualitative Evaluation)**
  - [x] Đọc và đối chiếu từng câu một cho **toàn bộ 7 câu Test (5 câu có đáp án + 2 câu thiếu bằng chứng)**:
  - [x] **Tiêu chí 1 — Tool Use:**
    - Baseline: 7/7 (100.0%) câu có gọi retrieval tool.
    - Selected Config: 7/7 (100.0%) câu có gọi retrieval tool.
  - [x] **Tiêu chí 2 — Grounded Answer (Thang điểm 0 - 1 - 2):**
    - Baseline: Mức 0: 4 câu, Mức 1: 0 câu, Mức 2: 1 câu (`q20`). Điểm trung bình: **0.40 / 2.0 (20.0%)**.
    - Selected Config: Mức 0: 1 câu (`q19`), Mức 1: 0 câu, Mức 2: 4 câu (`q16`, `q17`, `q18`, `q20`). Điểm trung bình: **1.60 / 2.0 (80.0%)** (Tăng gấp 4 lần so với Baseline).
  - [x] **Tiêu chí 3 — Citation Hợp lệ (Valid Citation):**
    - Kiểm tra 100% citation ID: tồn tại chính xác trong context và đoạn trích dẫn thực sự hỗ trợ câu trả lời.
    - Baseline: Citation validity = 1/1 (100.0%), Answer citation coverage = 1/1 (100.0%).
    - Selected Config: Citation validity = 4/4 (100.0%), Answer citation coverage = 4/4 (100.0%).
  - [x] **Tiêu chí 4 — Khả năng từ chối (Abstention):**
    - Câu thiếu evidence (`q_unans_test_01`, `q_unans_test_02`): Cả Baseline và Selected Config đều từ chối dứt khoát 100% (2/2), trả về *"Không đủ bằng chứng để trả lời."* và `citations: []`. Tuyệt đối không ảo giác.
    - Từ chối nhầm (False Abstention trên câu có đáp án): Baseline bị từ chối nhầm 4/5 câu (80.0%) do nghẽn context. Selected Config giảm mạnh xuống chỉ còn **1/5 câu (20.0%)**.
- [x] **5.3. Tổng hợp kết quả với `agent_metrics.py`**
  - [x] Chuẩn bị đầy đủ các file: `data/agent_cases.json`, `results/test_traces_selected.json`, `results/test_reviews_selected.json`.
  - [x] Chạy script:
    ```bash
    .venv/bin/python starter/agent_metrics.py --cases data/agent_cases.json --traces results/test_traces_selected.json --reviews results/test_reviews_selected.json --split test --output results/summary_test.json
    ```
  - [x] Đã đối chiếu số liệu và xuất cả file tổng hợp cho Baseline tại `results/summary_test_baseline.json`.

---

## GIAI ĐOẠN 6: HOÀN THIỆN BÁO CÁO, KIỂM TRA TÁI LẬP & NỘP BÀI
*Trọng số Rubric: 8 điểm (Tái lập và Báo cáo)*  
*Mục tiêu: Hoàn tất file báo cáo cá nhân chuẩn mẫu, code sạch sẽ, dễ dàng tái lập.*

- [x] **6.1. Khởi tạo file Báo cáo cá nhân**
  - [x] Đã tạo file: `report/individual/2A202603979_VuHuyHieu.md` chuẩn mẫu rubric.
  - [x] Điền đầy đủ thông tin:
    - [x] Dữ liệu: Quy chế VinUni, 32 trang, 105 chunks, vector cache dim 3072, Google GenAI embedding.
    - [x] Bảng tổng hợp Retrieval trên Dev và Test (Hit@1, Hit@3, Context Evidence Rate, Latency, Cost).
    - [x] Báo cáo 2 cấu hình cải tiến (Budget Expansion 2000 chars, Weighted Hybrid RRF) và giả thuyết.
    - [x] Phân tích chi tiết 2 ca lỗi trên Dev (`q01` Semantic Attraction, `q03` Context Budget Bottleneck).
    - [x] Bảng đánh giá Agent trên Test bằng `agent_metrics.py` (Tool use, Grounded Answer 0/1/2, Citation support, Abstention).
    - [x] Bảng chi tiết từng câu Test: Query ID, Tool call, Câu trả lời, Citation, Nhãn chấm và Lý do.
    - [x] Nhận xét đánh đổi (trade-offs), kết luận an toàn và giới hạn của hệ thống.
- [x] **6.2. Kiểm tra tính Tái lập (Reproducibility)**
  - [x] Đặt code hoàn chỉnh trong thư mục `src/` (`retrieval.py`, `agent.py`, `config.py`, `run_stage4_experiments.py`, `run_stage5_test_eval.py`).
  - [x] Cập nhật file `README.md` với mục *Hướng dẫn Tái lập (Reproducibility Guide)* chi tiết từng bước:
    - Lệnh cài đặt thư viện (`pip install -r requirements.txt`).
    - Lệnh chạy unit test (`python -m unittest discover tests -v`).
    - Lệnh chạy retrieval benchmark và tối ưu Dev (`python src/run_stage4_experiments.py`).
    - Lệnh chạy agent trên test (`python src/run_stage5_test_eval.py`).
    - Lệnh sinh file metrics chuẩn (`python starter/agent_metrics.py ...`).
- [x] **6.3. Rà soát an toàn & Quy tắc bảo mật (Audit & Clean)**
  - [x] Chạy lệnh kiểm tra: `git status` và `git diff`.
  - [x] Đảm bảo KHÔNG có file `.env`, API key, token bí mật nào bị commit (đã xác thực `.gitignore`).
  - [x] Đảm bảo KHÔNG commit dữ liệu mật hoặc file rác tạm thời.
  - [x] Sẵn sàng commit toàn bộ code, báo cáo và dữ liệu kiểm thử.

---

## 🌟 GIAI ĐOẠN 7 (TÙY CHỌN): THỬ THÁCH NÂNG CAO (+15 ĐIỂM BONUS)
*Áp dụng nếu muốn đạt điểm tối đa 100/100 (Rubric ADVANCED.md)*  
*Điều kiện tiên quyết: Đạt ít nhất 90% benchmark test (>= 27/30 câu đạt).*

- [ ] **7.1. Lựa chọn và Tiền xử lý Tài liệu thực tế >= 100 trang**
  - [ ] Chọn 1 tài liệu thực tế >= 100 trang chứa hỗn hợp Text, Bảng biểu (Tables), Ảnh (Images) và Biểu đồ (Charts).
  - [ ] Ghi rõ nguồn gốc, phiên bản, giấy phép sử dụng và đường link tải/truy cập công khai.
  - [ ] Xây dựng pipeline index đa phương thức (Multimodal Indexing):
    - [ ] Xử lý Text: OCR / PDF text extraction giữ liên kết số trang gốc.
    - [ ] Xử lý Bảng biểu: Trích xuất cấu trúc bảng (HTML/Markdown table), lưu toạ độ/trang.
    - [ ] Xử lý Ảnh & Biểu đồ: Cắt ảnh (crop), mô tả nội dung bằng Vision LLM hoặc multimodal embedding (đọc được trục, đơn vị, chú giải).
  - [ ] Ghi lại danh sách các trang hoặc phần không xử lý được (nếu có) và lý do.
- [ ] **7.2. Xây dựng Bộ Benchmark Thử thách Nâng cao**
  - [ ] **Tập Dev:** Ít nhất 15 câu có đáp án + 3 câu thiếu bằng chứng.
  - [ ] **Tập Test:** Ít nhất **30 câu** (bao gồm ít nhất 24 câu có đáp án + 6 câu thiếu bằng chứng) phân bổ đúng 5 nhóm chính:
    - [ ] **Bảng (Table):** Tối thiểu 6 câu (đọc đúng hàng, cột, đơn vị, tính toán/so sánh từ các ô).
    - [ ] **Ảnh, sơ đồ (Image/Diagram):** Tối thiểu 5 câu (nhận diện, giải thích nội dung hình vẽ).
    - [ ] **Biểu đồ (Chart):** Tối thiểu 5 câu (đọc trục tọa độ, xu hướng, giá trị kiểm chứng).
    - [ ] **Multi-hop:** Tối thiểu 4 câu (kết hợp evidence từ >= 2 trang, bảng hoặc hình khác nhau).
    - [ ] **Text:** Tối thiểu 4 câu (tổng hợp thông tin văn bản có điều kiện phức tạp).
    - [ ] **Thiếu bằng chứng (Abstention):** Tối thiểu 6 câu (câu hỏi gần chủ đề nhưng không có trong tài liệu).
  - [ ] Mỗi câu có metadata rõ ràng: ID, nhóm chính, câu hỏi, expected_answer, bằng chứng vùng trang/crop image, phép tính/đơn vị.
- [ ] **7.3. Thử nghiệm trên Dev và Tối ưu Pipeline**
  - [ ] Chạy pipeline trên Dev, tinh chỉnh prompt của Vision model / Agent / Retrieval.
  - [ ] Khóa pipeline trước khi chạy Test.
- [ ] **7.4. Đánh giá trên toàn bộ 30 câu Test và Nghiệm thu Ngưỡng >= 90%**
  - [ ] Chạy toàn bộ 30 câu Test, ghi nhận trace đầy đủ từng câu.
  - [ ] Chấm nhị phân **Đạt / Không đạt** cho từng câu:
    - Câu có đáp án: Đạt khi đúng, đủ, đúng đơn vị/phép tính, citation định vị đúng trang/vùng và evidence hỗ trợ toàn bộ câu trả lời.
    - Câu thiếu evidence: Đạt khi từ chối dứt khoát, không suy đoán vô căn cứ.
    - Lỗi hệ thống / timeout: Tính là Không đạt.
  - [ ] **Kiểm tra ngưỡng 90%:** Số câu đạt >= 27/30 (hoặc >= 90% tổng số câu test công bố).
- [ ] **7.5. Điền mục Báo cáo Nâng cao**
  - [ ] Điền mục *"Thử thách nâng cao"* trong `report/individual/2A202603979_VuHuyHieu.md`.
  - [ ] Liệt kê bảng kết quả đạt theo từng nhóm: Bảng, Ảnh, Biểu đồ, Multi-hop, Text, Thiếu bằng chứng.
  - [ ] Cung cấp bằng chứng tái lập, log trace và link truy cập tài liệu.

---

## 📋 MA TRẬN ĐỐI CHIẾU RUBRIC TỰ CHẤM (SELF-AUDIT)

| Hạng mục | Điểm tối đa | Yêu cầu kiểm tra | Trạng thái |
|---|:---:|---|:---:|
| **Dữ liệu & Benchmark** | 13 | Kết nối Day 6 (6đ); Benchmark >= 24 câu (dev 15+2, test 5+2), schema gold chuẩn (7đ) | `[ ] Chưa xong` |
| **Retrieval** | 17 | 3 phương pháp khác biệt gồm Dense Day 6 (10đ); Chuẩn hóa output, nguồn, rank/score, xử lý lỗi (7đ) | `[ ] Chưa xong` |
| **Agent & Context** | 17 | LLM thật, tool call, stopping criteria, trace (9đ); Context budget, format citation, abstention (8đ) | `[ ] Chưa xong` |
| **Đánh giá** | 17 | Metrics Hit@1, Hit@3, Context Rate (9đ); Chấm thủ công Grounded Answer, Citation Support, Abstention (8đ) | `[ ] Chưa xong` |
| **Thử nghiệm & Cải thiện** | 13 | 2 cấu hình cải tiến + giả thuyết + phân tích lỗi dev (5đ); Chốt dev trước khi chạy test (5đ); Trade-off (3đ) | `[ ] Chưa xong` |
| **Tái lập & Báo cáo** | 8 | Code sạch, hướng dẫn chạy chi tiết (5đ); Báo cáo chuẩn mẫu MSSV_HoTen.md, giới hạn (3đ) | `[ ] Chưa xong` |
| **TỔNG CƠ BẢN** | **85** | *Cần hoàn thành đầy đủ để đạt 85/85* | `[ ] Chưa xong` |
| **Thử thách nâng cao** | +15 | Tài liệu >= 100 trang, đa phương thức, 30 câu test, đạt >= 90% (>= 27/30) | `[ ] Tùy chọn` |
| **TỔNG CỘNG** | **100** | **Điểm tối đa toàn bài** | `[ ] Chưa xong` |

---

## 🛠️ DANH SÁCH LỆNH TERMINAL THƯỜNG DÙNG

```bash
# 1. Chạy unit tests kiểm tra khung code
python3 -m unittest discover -s tests -v

# 2. Chuyển đổi dữ liệu Day 6 (nếu dùng format Day 6)
python3 starter/convert_day6.py --chunks <path_chunks.json> --questions <path_questions.json> --output starter/data/converted

# 3. Chạy baseline TF-IDF mẫu
python3 starter/baseline.py --chunks data/chunks.json --questions data/questions.json --split dev --output results/dev-tfidf

# 4. Chạy tổng hợp chỉ số agent từ nhãn tự chấm
python3 starter/agent_metrics.py --cases data/cases.json --traces results/traces.json --reviews data/reviews.json --split test --output results/agent-test-summary.json
```
