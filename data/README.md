# Định dạng dữ liệu

Các file `example_*.json` chỉ minh họa định dạng, không đủ để nộp bài. Dữ liệu thật là JSON array UTF-8; nếu dùng định dạng khác, ghi cách chuyển trong README.

- **Chunk:** `id`, `text`, `source_id`, `source`, `page`, `metadata`. ID phải duy nhất; giữ phiên bản hoặc ngày nếu có.
- **Câu có đáp án:** `id`, `split` (`dev` hoặc `test`), `question`, `expected_answer`, `gold`: danh sách `{source_id, page, evidence}`. Với schema mẫu, mỗi evidence phải nằm trọn trong một chunk đúng nguồn và trang; các phần tử `gold` là evidence thay thế. Với câu multi-hop cần nhiều đoạn, bạn có thể mở rộng schema, ghi rõ nhóm evidence bắt buộc và cách chấm. Có thể thêm `filter` do người dùng cung cấp trước.
- **Câu thiếu evidence:** `id`, `split`, `question`, `answerable: false`, `expected_behavior: "search_and_abstain"`, `absence_reason`. Xem `agent_cases_example.json`.

Không đưa `gold` hoặc nhãn đáp án vào model và retrieval tool. Ghi nguồn dữ liệu và quyền sử dụng. Cách chuyển một số định dạng Day 6: [starter/README.md](../starter/README.md).

---

## Dữ liệu thực tế bài làm (Day 7 Dataset)

Tài liệu sử dụng: **Quy chế Đào tạo Đại học Hệ Chính quy theo Hệ thống Tín chỉ — VinUniversity (v8.0)**, gồm 35 trang vật lý, 32 trang nội dung hữu ích (trang 4 đến 35).

- `data/chunks.json`: Gồm **105 chunks** chuẩn hóa Unicode NFC theo Chiến lược C (Paragraph-aware + Metadata Filter, max 800 ký tự), lưu giữ đầy đủ `id`, `text`, `source_id`, `source`, `page`, và `metadata` (`section`, `char_count`, `strategy`).
- `data/questions.json`: Gồm **20 câu hỏi có đáp án (`answerable = true`)** (15 câu `dev` + 5 câu `test`), có đầy đủ `expected_answer`, `filter`, và `gold` evidence đối chiếu chuẩn xác 100% với văn bản chunk. Phục vụ đo đạc retrieval ranking (`baseline.py`).
- `data/unanswerable_questions.json`: Gồm **4 câu hỏi thiếu bằng chứng (`answerable = false`)** (2 câu `dev` + 2 câu `test`) về các chủ đề học vụ/sinh viên không có trong quy chế, kèm `absence_reason` và `expected_behavior: "search_and_abstain"`.
- `data/all_questions.json`: Tổng hợp toàn bộ **24 câu hỏi benchmark** (Dev: 17 câu; Test: 7 câu) đạt chuẩn Rubric.
- `data/agent_cases.json`: Gồm **24 cases** theo đúng schema chuẩn phục vụ đánh giá agent và script `agent_metrics.py`.
