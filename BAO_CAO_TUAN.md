# Báo cáo tuần: 02/11 – 07/11/2026

Người báo cáo: Ngô Gia Quốc

---

## Change log (sau feedback của mentor)

Sửa:

- Vách đá: per-tensor sụp từ 3 bit, per-channel từ 2 bit. Thêm dòng FP32 vào bảng bài 1.2.
- Bài 1.3: INT8 đóng góp thêm khoảng 1.5x, không phải 1.5–1.8x.
- Notebook 05: 0.06x là trung bình, riêng ≥ 1024 là 0.03–0.04x.
- SmolVLA: chạy đủ 100 episode cho `--num-steps 5` (91/100).
- Sửa lỗi trình bày, bỏ các nhận định chưa kiểm chứng.

Thêm:

- Kernel Triton: thông số kernel, cách đo, GPU, xác nhận bằng PTX nguyên nhân chậm trên T4.
- SmolVLA: suite, số episode, cách đo, nguyên nhân bf16 chậm, thí nghiệm Q8_0 / Q4_0.
- TinyML Chat: số token perplexity, tốc độ theo khoảng 3 lần đo.
- Plan: hai hướng cụ thể cho tuần tới. Mục 4: link và log.

---

## 1. Summary

Highlight

- Hoàn thành toàn bộ phần TODO trong 5 notebook về tối ưu model (quantization, distillation, pruning, NAS, kernel optimization), chạy trên Colab T4.
- Chạy được 2 ứng dụng trên laptop cá nhân, không dùng Colab:

  - TinyML Chat (SmolLM2-360M, CPU): với AWQ, mức tăng perplexity của INT4 giảm từ +22% xuống +5%.
  - SmolVLA qua vla.cpp (RTX 3050 4 GB): robot mô phỏng gắp thành công 93/100 episode. Q8_0 giảm một nửa VRAM (871 MiB), vẫn 90/100.
- Bài lab 1.1 (KD → pruning → INT8): model nhỏ hơn 144x, accuracy chỉ giảm 1.69 điểm.
- Bài lab 1.2 (vách đá quantization + QAT): QAT INT4 đạt 91.38%, không mất accuracy so với FP32.
- Bài lab 1.3 (INT8 thật trên CPU): ONNX Runtime INT8 nhanh hơn PyTorch FP32 3.03x ở batch 1, đổi lại mất 1.28 điểm accuracy; PyTorch INT8 chỉ nhanh hơn 1.88x.

Insight

*Kỹ thuật*

* Model nhỏ hơn không có nghĩa là chạy nhanh hơn, tốc độ phụ thuộc vào kernel.
* KD và INT8 gần như miễn phí, pruning là bước đắt nhất và bắt buộc phải fine-tune.
* Quantization có “vách đá” (mức bit mà từ đó accuracy rơi đột ngột): per-tensor sụp từ 3 bit, per-channel từ 2 bit, QAT giúp lùi vách đá đó.
* Với robot/VLA, giảm số bước khử nhiễu hiệu quả hơn giảm số bit: 10 → 5 bước nhanh hơn 15% và gần như không mất, còn 4 bit mất 27 điểm.

*Phương pháp*

* Kết quả một seed có thể chỉ là nhiễu, phải so sánh công bằng và đo trong điều kiện cô lập.
* Đo đúng thứ cần đo, không đổi cách đo cho đẹp số.
* Cache dữ liệu và viết test sớm để tiết kiệm thời gian.

## 2. Completed Work

### 2.1. Notebook ví dụ (01 – 05)

Hoàn thành mọi TODO, chạy trên Colab T4. Latency đo trên CPU Colab (trung bình 100 lần), dao động giữa các notebook nên chỉ so trước/sau trong cùng một notebook.

| Notebook                  | Trước (size, acc, CPU latency)           | Sau (size, acc, CPU latency)                                                         |
| ------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------ |
| 01 Quantization           | FP32: 4.61 MB, 99.68%, 7.81 ms             | INT8 per-channel: 1.18 MB, 99.68%, 7.80 ms                                           |
| 02 Knowledge Distillation | Teacher: 4.61 MB, 99.67%, 5.58 ms          | Student + KD: 0.07 MB, 99.46%, 0.63 ms                                               |
| 03 Pruning                | Dense: 4.61 MB, 99.64%, 6.07 ms            | Structured 50%: 1.17 MB, 99.69%, 2.62 ms                                             |
| 04 NAS                    | Baseline: 0.38 MB, 99.44%, 1.18 ms         | NAS: 0.23 MB, 99.44%, 0.81 ms                                                        |
| 05 Kernel Optimization    | `torch.matmul`: 22–34 TFLOP/s (≥ 1024) | Triton: ~1 TFLOP/s, 0.03–0.04x torch (≥ 1024; trung bình 512–4096 là 0.06x), T4 |

- Cả 4 kỹ thuật đều giữ được accuracy. Latency giảm nhờ student (KD), pruning và NAS; INT8 chỉ làm model nhỏ hơn.
- Kernel notebook 05 đúng (FP16, `tl.dot`, tile 64×64×32, đo bằng `do_bench`, Tesla T4) nhưng chạy chậm.
- Chạy lại ở bài 1.4: Triton đạt 114% cuBLAS trên L4, 4.5% trên T4. PTX cho thấy L4 dùng tensor core (16 `mma.sync`), T4 rơi về CUDA core (0 `mma.sync`, 1024 `fma.rn.f32`). 

### 2.2. Ứng dụng trên laptop

TinyML Chat: chat với LLM SmolLM2-360M trên CPU laptop. Viết forward pass LLaMA, quantize weight INT8/INT4 và thêm AWQ.

| Precision |  Weight | Perplexity | Tốc độ (khoảng 3 lần đo) |
| --------- | ------: | ---------: | -----------------------------: |
| fp32      | 1447 MB |      33.19 |                4.9–15.4 tok/s |
| bf16      |  724 MB |      33.22 |               14.3–23.1 tok/s |
| int8      |  363 MB |      33.22 |                 5.8–6.2 tok/s |
| int4      |  244 MB |      40.58 |                 2.1–2.3 tok/s |
| awq4      |  244 MB |      34.84 |                 2.5–3.0 tok/s |

- Perplexity đo trên một đoạn văn ngắn (113 token), chênh dưới ~1 là nhiễu. fp32/bf16 dao động 2–3 lần giữa các lần chạy, thứ tự tốc độ giữ nguyên.
- AWQ giúp INT4 gần bằng chất lượng fp32.
- INT8/INT4 nhỏ hơn nhưng chậm hơn bf16, vì kernel số nguyên của PyTorch trên CPU chậm.

SmolVLA + vla.cpp: chạy model robot SmolVLA trên GPU laptop RTX 3050 4 GB. Robot mô phỏng trong LIBERO (`libero_object`) gắp vật bỏ vào giỏ. Mặc định, `--num-steps 5`, Q8_0, Q4_0 chạy 100 episode, các cấu hình còn lại 30 episode. Q8_0/Q4_0 chỉ lượng tử hóa LM. Độ trễ là median 100 lần gọi.

| Cấu hình server        | Thành công | Độ trễ | VRAM weight |
| ------------------------ | -----------: | --------: | ----------: |
| mặc định              |       93/100 |    183 ms |    1763 MiB |
| `--weight-dtype bf16`  |        27/30 |    221 ms |     985 MiB |
| `--num-steps 5`        |       91/100 |    155 ms |    1763 MiB |
| `--flash-attn 1`       |        26/30 |    163 ms |    1763 MiB |
| Q8_0                     |       90/100 |    167 ms |     871 MiB |
| Q8_0 + `--num-steps 5` |       85/100 |    143 ms |     871 MiB |
| Q4_0 + `--num-steps 5` |       64/100 |    142 ms |     715 MiB |

- Giảm số bước khử nhiễu 10 → 5 nhanh nhất (−15%), 91/100 so với 93/100 (trong nhiễu). Q8_0 dùng được (giảm VRAM một nửa, mất 3 episode), Q4_0 sụp. Kết hợp Q8_0 + 5 bước mất 8 episode, cần thêm seed.
- bf16 chậm hơn vì phần suy luận LM + action expert (88 → 125 ms), vision không đổi. Giả thuyết kernel bf16 của ggml-cuda chậm, chưa kiểm chứng (nsys trong WSL2 không ghi được kernel).
- Test thử gắp nhiều vật liên tiếp nhưng chỉ đạt 1–2/4 liên tiếp, dù từng vật gắp riêng đạt 10/10.

### 2.3. Bài lab luyện thêm

Bài 1.1 – Pipeline KD → Pruning → INT8: nén ResNet-18 (CIFAR-10), chỉ tiêu nhỏ hơn ≥ 10x và accuracy giảm ≤ 2 điểm. Đạt: nhỏ hơn 144x, accuracy giảm 1.69 điểm.

| Bước                        |     Size |    Acc |     Δ so với bước trước |  Latency |
| ----------------------------- | -------: | -----: | ----------------------------: | -------: |
| Teacher ResNet-18             | 44.77 MB | 91.20% |                               | 20.45 ms |
| Student + KD                  |  4.62 MB | 91.38% |                         +0.18 |  5.61 ms |
| Pruning 50% (chưa fine-tune) |  1.17 MB | 14.38% |                       −77.00 |  2.57 ms |
| Pruning + fine-tune           |  1.17 MB | 89.49% | −1.89 (so với Student + KD) |  2.45 ms |
| PTQ INT8                      |  0.31 MB | 89.51% |                         +0.02 |  2.55 ms |

- KD và INT8 gần như không mất accuracy. Pruning mất nhiều nhất và bắt buộc phải fine-tune, chưa fine-tune thì accuracy sụp rất nhanh.
- Latency giảm nhờ KD và pruning. INT8 không nhanh hơn vì vẫn tính bằng FP32.

Bài 1.2 – Vách đá quantization + QAT: giảm số bit của weight từ 8 xuống 2 để tìm mức bit accuracy bắt đầu sụp, rồi dùng QAT kéo accuracy trở lại. Chỉ tiêu QAT INT4 mất ≤ 1 điểm: đạt.

|  Bit | PTQ per-tensor | PTQ per-channel | QAT per-channel |
| ---: | -------------: | --------------: | --------------: |
| FP32 |         91.25% |          91.25% |          91.25% |
|    5 |         90.80% |          91.04% |                 |
|    4 |         89.58% |          90.45% |          91.38% |
|    3 |         45.00% |          86.42% |          91.27% |
|    2 |         10.00% |          11.20% |          89.14% |

- Mất hơn 1 điểm: per-tensor từ 4 bit, per-channel từ 3 bit. Sụp hẳn: per-tensor ở 3 bit, per-channel ở 2 bit.
- QAT 8 epoch đưa INT4 và INT3 về ngang FP32. Một phần mức tăng có thể đến từ việc train thêm 8 epoch, và mới chạy một seed.

Bài 1.3 – INT8 thật trên CPU: quantize cả weight lẫn activation để conv chạy bằng kernel số nguyên (PyTorch và ONNX Runtime).

| Model               |    Acc |    Size | Batch 1 | Tăng tốc | Batch 64 | Tăng tốc |
| ------------------- | -----: | ------: | ------: | ---------: | -------: | ---------: |
| PyTorch FP32 (mốc) | 91.25% | 4.62 MB | 5.63 ms |      1.00x |   214 ms |      1.00x |
| PyTorch INT8        | 91.16% | 1.18 MB | 3.00 ms |      1.88x |   141 ms |      1.52x |
| ONNX Runtime FP32   | 91.25% | 4.62 MB | 2.89 ms |      1.95x |   161 ms |      1.33x |
| ONNX Runtime INT8   | 89.97% | 1.19 MB | 1.86 ms |      3.03x |   110 ms |      1.95x |

- INT8 chỉ nhanh hơn khi có kernel số nguyên.
- Riêng việc đổi sang ONNX Runtime đã nhanh hơn 1.95x (batch 1). Phần INT8 đóng góp thêm khoảng 1.5x (ORT: 1.55x / 1.47x; PyTorch: 1.78x / 1.49x ở batch 1 / 64).
- ONNX Runtime INT8 mất 1.28 điểm accuracy (91.25% → 89.97%), nhiều hơn PyTorch INT8 (−0.09).

## 3. Plan

Action items (hoàn thiện những gì tuần này còn dở):

- [ ] Chạy lại bài 1.1 và 1.2 với 3 seed để biết phương sai, vì bài 1.1 chỉ còn dư 0.31 điểm so với chỉ tiêu. Thêm mốc FP32 fine-tune 8 epoch để so công bằng với QAT.
- [ ] Bài 1.3: Thử tìm cách lấy lại 1.28 điểm accuracy cho ONNX Runtime INT8.
- [ ] SmolVLA: chạy thêm seed cho các cấu hình 85–93/100 và đo `--num-steps 5` + `--flash-attn 1`.

Đề xuất cho tuần tới:

- SmolVLA: lượng tử hóa cả vision tower (`--vision`), vì vision (~95 ms) chiếm hơn 2/3 độ trễ sau khi quantize LM.
- SmolVLA: xác nhận nguyên nhân bf16 chậm ở mức kernel (profiler trên Windows hoặc `ncu`).

## 4. Tài liệu

Repo: https://github.com/KuSenpai/modelopt101 (branch `main`). Notebook có output Colab (riêng cell PTX của bài 1.4 chưa lưu output, kết quả ghi trong `exercises/report.md`).

| Bài              | Code / notebook                                                                                                     | Báo cáo, log                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Notebook 01–05   | [examples/](https://github.com/KuSenpai/modelopt101/tree/main/examples)                                              | [report.md](https://github.com/KuSenpai/modelopt101/blob/main/report.md)                     |
| Bài 1.1          | [1.1_pipeline_cifar10.ipynb](https://github.com/KuSenpai/modelopt101/blob/main/exercises/1.1_pipeline_cifar10.ipynb) | [exercises/report.md](https://github.com/KuSenpai/modelopt101/blob/main/exercises/report.md) |
| Bài 1.2, 1.3     | [1.2_quant_cliff.ipynb](https://github.com/KuSenpai/modelopt101/blob/main/exercises/1.2_quant_cliff.ipynb)           | [exercises/report.md](https://github.com/KuSenpai/modelopt101/blob/main/exercises/report.md) |
| Bài 1.4 (kernel) | [1.4_triton_kernels.ipynb](https://github.com/KuSenpai/modelopt101/blob/main/exercises/1.4_triton_kernels.ipynb)     | [exercises/report.md](https://github.com/KuSenpai/modelopt101/blob/main/exercises/report.md) |
| TinyML Chat       | [apps/tinyml_chat/](https://github.com/KuSenpai/modelopt101/tree/main/apps/tinyml_chat)                              | [results/](https://github.com/KuSenpai/modelopt101/tree/main/apps/tinyml_chat/results)       |
| SmolVLA + vla.cpp | [apps/vla.cpp/](https://github.com/KuSenpai/modelopt101/tree/main/apps/vla.cpp)                                      | [results/](https://github.com/KuSenpai/modelopt101/tree/main/apps/vla.cpp/results)           |
