// Generate the reproduction report (Vietnamese) from my_cil/results/*.json
// Usage: node make_report.js   ->  BaoCao_TaiLap_BiCM-Prompt.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, LevelFormat, BorderStyle, PageNumber, Footer,
} = require("docx");

const RESULTS = path.join(__dirname, "..", "my_cil", "results");
const META = path.join(__dirname, "..", "my_cil", "data_houston", "paper_seed0", "meta.json");
const OUT = path.join(__dirname, "BaoCao_TaiLap_BiCM-Prompt.docx");

// ---------- data ----------
const PAPER = [ // Table IV, Houston2013, "Ours"
  { s: "Base", n: 6, OA: 96.96, AA: 97.55, K: 96.35, APD: null },
  { s: "Session 1", n: 9, OA: 93.14, AA: 94.09, K: 92.28, APD: 0.0528 },
  { s: "Session 2", n: 12, OA: 82.85, AA: 85.13, K: 81.26, APD: 0.1142 },
  { s: "Session 3", n: 15, OA: 77.39, AA: 82.09, K: 75.56, APD: 0.0576 },
];
const PAPER_AVG = { OA: 87.58, AA: 89.71, K: 86.36 };
const CLASS_NAMES = ["Healthy grass", "Stressed grass", "Synthetic grass", "Trees", "Soil", "Water",
  "Residential", "Commercial", "Road", "Highway", "Railway", "Parking lot 1", "Parking lot 2",
  "Tennis court", "Running track"];
const TABLE1_TEST = [1071, 1074, 517, 1064, 1062, 145, 1088, 1064, 1072, 1047, 1055, 1053, 289, 248, 480];

function loadRuns(prefix) {
  if (!fs.existsSync(RESULTS)) return [];
  return fs.readdirSync(RESULTS)
    .filter((f) => f.startsWith(prefix) && f.endsWith(".json"))
    .sort()
    .map((f) => ({ name: f.replace(".json", ""), ...JSON.parse(fs.readFileSync(path.join(RESULTS, f))) }));
}
const paperRuns = loadRuns("paper_seed");
const origRuns = loadRuns("original_seed");
const meta = fs.existsSync(META) ? JSON.parse(fs.readFileSync(META)) : null;

const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const std = (a) => (a.length < 2 ? 0 : Math.sqrt(a.reduce((s, x) => s + (x - mean(a)) ** 2, 0) / (a.length - 1)));
const f2 = (x) => (x === null || x === undefined ? "–" : x.toFixed(2));
const f4 = (x) => (x === null || x === undefined ? "–" : x.toFixed(4));
const ms = (a, d = 2) => (a.length === 0 ? "–" : a.length === 1 ? a[0].toFixed(d) : `${mean(a).toFixed(d)} ± ${std(a).toFixed(d)}`);

// ---------- styling helpers ----------
const FONT = "Times New Roman";
const PAGE_W = 11906, MARGIN = 1134, CONTENT_W = PAGE_W - 2 * MARGIN; // A4, 2 cm margins
const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };

const p = (text, opts = {}) => new Paragraph({
  spacing: { after: 120, line: 300 }, alignment: opts.align || AlignmentType.JUSTIFIED,
  children: (Array.isArray(text) ? text : [text]).map((t) => (typeof t === "string" ? new TextRun({ text: t }) : t)),
});
const b = (text) => new TextRun({ text, bold: true });
const i = (text) => new TextRun({ text, italics: true });
const code = (text) => new TextRun({ text, font: "Consolas", size: 20 });
const h1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
const h2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
const bullet = (parts, level = 0) => new Paragraph({
  numbering: { reference: "bullets", level }, spacing: { after: 60, line: 288 },
  children: (Array.isArray(parts) ? parts : [parts]).map((t) => (typeof t === "string" ? new TextRun(t) : t)),
});
const numbered = (parts) => new Paragraph({
  numbering: { reference: "numbers", level: 0 }, spacing: { after: 60, line: 288 },
  children: (Array.isArray(parts) ? parts : [parts]).map((t) => (typeof t === "string" ? new TextRun(t) : t)),
});
const codeBlock = (lines) => lines.map((l) => new Paragraph({
  spacing: { after: 0 }, shading: { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" },
  children: [code(l)],
}));
const caption = (text) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 120, after: 80 }, keepNext: true,
  children: [new TextRun({ text, bold: true, size: 22 })],
});

function table(header, rows, widthsRel, opts = {}) {
  const total = widthsRel.reduce((a, c) => a + c, 0);
  const widths = widthsRel.map((w) => Math.floor((w / total) * CONTENT_W));
  widths[widths.length - 1] += CONTENT_W - widths.reduce((a, c) => a + c, 0);
  const cell = (text, w, isHead, fill) => new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    shading: isHead ? { type: ShadingType.CLEAR, fill: "D9E2F3", color: "auto" }
      : fill ? { type: ShadingType.CLEAR, fill, color: "auto" } : undefined,
    children: [new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 0 },
      children: [new TextRun({ text: String(text), bold: isHead, size: 20 })],
    })],
  });
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, k) => cell(h, widths[k], true)) }),
      ...rows.map((r, ri) => new TableRow({
        children: r.map((c, k) => cell(c, widths[k], false, opts.rowFill ? opts.rowFill(ri) : undefined)),
      })),
    ],
  });
}

// ---------- sections ----------
const children = [];
children.push(new Paragraph({
  heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER,
  children: [new TextRun("BÁO CÁO TÁI LẬP KẾT QUẢ BiCM-Prompt")],
}));
children.push(p([i("BiCM-Prompt: Bidirectional Cross-Modal Prompt Tuning for Class-Incremental Learning on Multisource Remote Sensing Images")], { align: AlignmentType.CENTER }));
children.push(p("Y. Zou, J. Qu, W. Dong, Y. Li — IEEE Transactions on Image Processing, vol. 35, 2026. DOI: 10.1109/TIP.2026.3706299", { align: AlignmentType.CENTER }));
children.push(p(`Ngày cập nhật: ${new Date().toISOString().slice(0, 10)} — Phạm vi: dataset Houston 2013, phương pháp đề xuất ("Ours")`, { align: AlignmentType.CENTER }));

// 1
children.push(h1("1. Mục tiêu và phạm vi"));
children.push(p("Mục tiêu là chạy lại toàn bộ quy trình của bài báo, từ dữ liệu gốc → tiền xử lý → huấn luyện tăng dần → đánh giá, bằng mã nguồn công khai của tác giả, sau đó so sánh với số liệu trong Table IV của bài báo."));
children.push(p("Báo cáo này trình bày những phần đã chạy xong: chuẩn bị môi trường, dữ liệu Houston 2013, đối chiếu mã nguồn với bài báo, và kết quả phương pháp đề xuất trên Houston 2013. Các baseline, MUUFL, Augsburg và các thí nghiệm ablation chưa được chạy (xem mục 8)."));

// 2
children.push(h1("2. Mã nguồn và môi trường"));
children.push(bullet([b("Mã nguồn: "), "repo gốc github.com/Jiahuiqu/BiCMPT, được sao chép (không giữ lịch sử) sang github.com/VuHa123/CIL_Remoate_sensing."]));
children.push(bullet([b("Nhánh git: "), code("original-code"), " = mã gốc (chỉ thêm pyproject.toml, .gitignore); ", code("paper-config"), " = cấu hình theo bài báo + sửa lỗi (mục 4)."]));
children.push(bullet([b("Phần cứng: "), "NVIDIA GeForce RTX 3060 Laptop (6 GB), driver 575.64.03 (CUDA 12.9)."]));
children.push(bullet([b("Môi trường: "), "conda env ", code("cil_rs"), ", Python 3.10, PyTorch 2.5.1+cu121, timm 1.0.31, NumPy 2.2.6, scikit-learn 1.7.2, SciPy 1.15.3, imbalanced-learn. Đặt PYTHONNOUSERSITE=1 để tách khỏi gói trong ~/.local."]));
children.push(bullet([b("Lưu ý cài đặt: "), "repo gốc thiếu pyproject.toml nên lệnh pip install -e . trong README không chạy được; đã bổ sung. Khi cài timm, pip tự nâng PyTorch lên bản CUDA 13 không tương thích driver, phải ghim lại torch 2.5.1/cu121."]));
children.push(bullet([b("Kiểm thử: "), "5/5 unit test của repo đạt; chạy thử toàn bộ quy trình trên dữ liệu giả trước khi chạy dữ liệu thật."]));

// 3
children.push(h1("3. Dữ liệu"));
children.push(h2("3.1. Nguồn tải"));
children.push(p([b("Nguồn đã dùng: "), "bản Houston 2013 định dạng .mat (bản \"MMRS\", từ Multimodal-Remote-Sensing-Toolkit của likyoo) được phát hành tại ",
  code("https://github.com/songyz2019/rs-fusion-datasets-dist/releases/download/v1.0.0/houston2013-mmr.zip"),
  " (161 MB). Đây là nguồn mà thư viện Python rs-fusion-datasets dùng; dữ liệu gốc thuộc 2013 IEEE GRSS Data Fusion Contest (machinelearning.ee.uh.edu/?page_id=459)."]));
children.push(p("Toàn bộ file được kiểm tra bằng SHA-256 (khớp giá trị công bố trong rs-fusion-datasets):"));
children.push(table(["File", "Biến", "Kích thước", "SHA-256 (rút gọn)"], [
  ["houston2013-mmr.zip", "–", "161 MB", "7a2d719d…ae1fd762"],
  ["HSI.mat", "HSI", "349 × 1905 × 144", "6a0edba3…b3c6fd"],
  ["LiDAR.mat", "LiDAR", "349 × 1905", "7aa956e7…b35ad"],
  ["TRLabel.mat", "TRLabel", "2 832 pixel có nhãn", "96ce863e…2b527"],
  ["TSLabel.mat", "TSLabel", "12 197 pixel có nhãn", "46bd849d…a45"],
  ["gt.mat", "gt", "349 × 1905", "75ecccc0…6ae0"],
], [3, 1.4, 2.4, 2.6]));
children.push(p([b("Về các link bạn tìm được: "), "chưa dùng link nào trong số đó cho Houston. Nhận xét để dùng về sau:"]));
children.push(bullet([code("github.com/GatorSense/MUUFLGulfport"), " — nguồn chính thức của MUUFL Gulfport, phù hợp cho thí nghiệm MUUFL (rs-fusion-datasets cũng tải từ đây)."]));
children.push(bullet([code("kaggle.com/datasets/vkpanigrahi/houston2013-hsi-lidar-dataset"), " và ", code("github.com/YangHuihan219/Houston2013"), " — bản sao do cá nhân đăng lại, chưa kiểm tra nội dung; không cần thiết vì đã có bản kiểm chứng bằng checksum."]));
children.push(bullet([code("github.com/zhu-xlab/augsburg_Multimodal_Data_Set_MDaS"), " — đây là bộ MDaS (2023), ", b("có thể không phải"), " bộ Augsburg HSI–SAR mà bài báo dùng (332 × 485 pixel, 180 band HySpex + Sentinel-1, 7 lớp — bộ của Hong et al.). Cần kiểm tra trước khi dùng."]));

children.push(h2("3.2. Tiền xử lý theo bài báo"));
children.push(bullet("PCA giảm HSI từ 144 → 40 band (giữ 99,99 % phương sai), LiDAR giữ nguyên 1 kênh; chuẩn hoá min–max từng kênh."));
children.push(bullet("Cắt patch 11 × 11 quanh mỗi pixel có nhãn → mẫu 11 × 11 × 41."));
children.push(bullet("Gộp nhãn train + test của bộ chuẩn DFC2013 (2 832 + 12 197 = 15 029 pixel), chọn ngẫu nhiên 180 pixel/lớp làm train, phần còn lại làm test (seed của phép chia = seed của lần chạy)."));
children.push(p([b("Kiểm chứng: "), "số mẫu test thu được khớp chính xác từng lớp với Table I của bài báo (tổng 2 700 train / 12 329 test). Bộ chia chuẩn DFC2013 (2 832 train) không khớp mô tả 180 mẫu/lớp của bài báo."]));
const t1rows = CLASS_NAMES.map((n, k) => {
  const ours = meta ? meta.test_per_class[String(k + 1)] : "–";
  return [k + 1, n, meta ? meta.train_per_class[String(k + 1)] : "–", ours, TABLE1_TEST[k], ours === TABLE1_TEST[k] ? "✓" : "✗"];
});
children.push(caption("Bảng 1. Số mẫu mỗi lớp (seed 0) so với Table I"));
children.push(table(["#", "Lớp", "Train", "Test (tái lập)", "Test (bài báo)", "Khớp"], t1rows, [0.6, 2.6, 1, 1.5, 1.5, 0.8]));

// 4
children.push(h1("4. Đối chiếu mã nguồn với bài báo"));
children.push(p("Mã công khai không trùng với cấu hình được mô tả trong bài báo cho Houston 2013:"));
children.push(caption("Bảng 2. Khác biệt giữa mã gốc và bài báo"));
children.push(table(["Hạng mục", "Bài báo", "Mã gốc (train_incremental.py)", "Cấu hình tái lập"], [
  ["Chia lớp", "6 + 3 + 3 + 3", "7 + 3 + 3 + 2", "6 + 3 + 3 + 3"],
  ["Thứ tự lớp", "Thứ tự gốc (lớp 1–6 là base)", "Hoán vị seed 4 (base = lớp gốc 2,3,4,5,8,12,13)", "Thứ tự gốc"],
  ["Kênh HSI", "PCA 40", "144 band", "PCA 40"],
  ["Chia train/test", "180 mẫu/lớp ngẫu nhiên", "Đọc từ file .mat (không rõ)", "180 mẫu/lớp ngẫu nhiên"],
  ["Bộ nhớ replay", "30 mẫu, 2 mẫu/lớp ở cuối", "K = 30; base m = 6, sau đó m = K // s", "m = 30 // số lớp đã thấy (5, 3, 2, 2)"],
  ["Huấn luyện base", "Adam, lr 1e-3, batch 32", "Adam, lr 1e-4, batch 16", "Adam, lr 1e-3, batch 32"],
  ["Prompt pool / top-k / độ dài", "30 / 3 / 5", "30 / 3 / 5", "30 / 3 / 5"],
  ["Seed", "–", "Chỉ random.seed(11)", "random, numpy, torch"],
], [2, 2.4, 3, 2.6]));
children.push(h2("4.1. Lỗi trong mã gốc"));
children.push(p("Kiểm tra trực tiếp tập tham số mà optimizer cập nhật ở từng session cho thấy:"));
children.push(bullet([b("Backbone có được đóng băng từ session 1: "), "đúng như bài báo — ViT của cả hai nhánh có requires_grad = False; chỉ prompt, PGKA (TaskAware, trừ khối grouping), BN và fc được học."]));
children.push(bullet([b("Optimizer chỉ được tạo một lần trong __init__: "), "lớp phân loại fc (được tạo lại khi thêm lớp), các mạng chiếu hai chiều BiCMP (stage_mapping_net) và task token G (thậm chí không phải nn.Parameter) ", b("không bao giờ được cập nhật"), ". Như vậy mô hình gốc dùng fc khởi tạo ngẫu nhiên cố định và BiCMP không được huấn luyện."]));
children.push(bullet([b("Trọng số lớp cho OT: "), "được tính theo thứ tự đã xáo trộn của DataLoader nhưng lại tra bằng chỉ số của dataset, nên gán trọng số sai mẫu; đồng thời tính lại trên toàn bộ dữ liệu ở mỗi batch (rất chậm)."]));
children.push(p([b("Sửa: "), "cờ fix_optimizer (bật trong preset paper) tạo lại Adam + cosine scheduler ở mỗi session cho đúng các tham số đang requires_grad, đăng ký task token là tham số (token cũ đóng băng, token mới được học), và tính trọng số OT theo thứ tự dataset. Với cờ này, số tham số học được ở các session tăng dần là ≈ 0,850 M, sát với 0,843 M mà bài báo báo cáo — cho thấy tập tham số đã đúng ý đồ tác giả. Preset original giữ nguyên hành vi gốc để so sánh."]));

// 5
children.push(h1("5. Cấu hình chạy và giả định"));
children.push(bullet("Base: 30 epoch, Adam lr 1e-3, batch 32 (hiểu \"pretraining\" trong bài báo là session base, vì mã không có bước pretrain tương thích: scripts/pretrain_backbone.py dùng ViT 768 chiều, khác backbone 128 chiều của mô hình)."));
children.push(bullet("Session tăng dần: 35 epoch, Adam lr 1e-4, batch 16, weight decay 1e-4, cosine annealing — giữ mặc định của mã (bài báo không nêu)."));
children.push(bullet("Trọng số các hàm loss: giữ như mã (CE + 0,01·MI + sim + aux + distill, CE có trọng số OT)."));
children.push(bullet("Đánh giá: giống mã gốc — session base dùng bộ phân loại NME (trung bình exemplar), các session sau dùng fc; cả hai đều được ghi lại. APD theo công thức (20) của bài báo, trùng với hàm calculate_forgetting_rate của mã."));
children.push(bullet("Mỗi seed dùng một phép chia train/test ngẫu nhiên khác nhau (seed 0, 1, 2)."));

// 6
children.push(h1("6. Kết quả trên Houston 2013"));
const nS = PAPER.length;
if (paperRuns.length === 0) {
  children.push(p("Chưa có kết quả."));
} else {
  children.push(p(`Đã chạy xong ${paperRuns.length} seed với preset paper (${paperRuns.map((r) => r.name.replace("paper_", "")).join(", ")}). Mỗi lần chạy mất khoảng ${Math.round(mean(paperRuns.map((r) => r.total_seconds)) / 60)} phút trên RTX 3060.`));
  const rows = [];
  for (let s = 0; s < nS; s++) {
    const pick = (k) => paperRuns.map((r) => r.sessions[s] && r.sessions[s][k]).filter((x) => x !== undefined && x !== null);
    rows.push([PAPER[s].s, PAPER[s].n,
      ms(pick("OA")), f2(PAPER[s].OA), ms(pick("AA")), f2(PAPER[s].AA),
      ms(pick("Kappa")), f2(PAPER[s].K), s === 0 ? "–" : ms(pick("APD"), 4), f4(PAPER[s].APD)]);
  }
  const avg = (k) => paperRuns.map((r) => r.average[k]);
  rows.push(["Trung bình", "", ms(avg("OA")), f2(PAPER_AVG.OA), ms(avg("AA")), f2(PAPER_AVG.AA), ms(avg("Kappa")), f2(PAPER_AVG.K), "", ""]);
  children.push(caption(`Bảng 3. Tái lập (preset paper, ${paperRuns.length > 1 ? "trung bình ± độ lệch chuẩn qua " + paperRuns.length + " seed" : "1 seed"}) so với Table IV`));
  children.push(table(["Session", "Số lớp", "OA tái lập", "OA bài báo", "AA tái lập", "AA bài báo", "Kappa tái lập", "Kappa bài báo", "APD tái lập", "APD bài báo"],
    rows, [1.5, 0.8, 1.5, 1.1, 1.5, 1.1, 1.5, 1.1, 1.4, 1.1], { rowFill: (ri) => (ri === nS ? "F2F2F2" : undefined) }));

  children.push(caption("Bảng 4. Kết quả từng seed (OA / AA / Kappa, %)"));
  children.push(table(["Seed", ...PAPER.map((x) => x.s), "Trung bình OA", "Thời gian"],
    paperRuns.map((r) => [r.name.replace("paper_seed", ""),
      ...PAPER.map((_, s) => r.sessions[s] ? `${f2(r.sessions[s].OA)} / ${f2(r.sessions[s].AA)} / ${f2(r.sessions[s].Kappa)}` : "–"),
      f2(r.average.OA), `${Math.round(r.total_seconds / 60)} phút`]),
    [0.7, 2.1, 2.1, 2.1, 2.1, 1.3, 1.1]));

  children.push(caption("Bảng 5. OA của hai cách phân loại ở mỗi session (trung bình các seed, %)"));
  children.push(table(["Session", "fc (dùng để báo cáo, trừ base)", "NME (trung bình exemplar)"],
    PAPER.map((x, s) => [x.s, ms(paperRuns.map((r) => r.sessions[s].fc.OA)), ms(paperRuns.map((r) => r.sessions[s].nme.OA))]),
    [2, 3, 3]));

  // per-class at last session
  const last = nS - 1;
  const pc = CLASS_NAMES.map((n, k) => {
    const vals = paperRuns.map((r) => r.sessions[last].per_class_acc[String(k)] * 100);
    const s2 = k < 12 ? paperRuns.map((r) => r.sessions[2].per_class_acc[String(k)] * 100) : [];
    const s1 = k < 9 ? paperRuns.map((r) => r.sessions[1].per_class_acc[String(k)] * 100) : [];
    return [k + 1, n, s1.length ? ms(s1, 1) : "–", s2.length ? ms(s2, 1) : "–", ms(vals, 1)];
  });
  children.push(caption("Bảng 6. Độ chính xác từng lớp (%) sau Session 1, 2, 3"));
  children.push(table(["#", "Lớp", "Sau S1", "Sau S2", "Sau S3"], pc, [0.6, 2.6, 1.8, 1.8, 1.8]));
}

if (origRuns.length) {
  const r = origRuns[0];
  children.push(caption("Bảng 7. Mã gốc nguyên trạng (preset original: 7+3+3+2, 144 band, bộ chia chuẩn DFC2013)"));
  children.push(table(["Session", "Số lớp", "OA", "AA", "Kappa", "APD"],
    r.sessions.map((s) => [s.session === 0 ? "Base" : `Session ${s.session}`, s.seen_classes, f2(s.OA), f2(s.AA), f2(s.Kappa), s.session ? f4(s.APD) : "–"]),
    [1.5, 1, 1.3, 1.3, 1.3, 1.3]));
  children.push(p("Lưu ý: cấu hình này có số lớp mỗi session, thứ tự lớp, dữ liệu đầu vào và phép chia khác bài báo nên không so sánh trực tiếp được với Table IV; chỉ dùng để thấy hành vi của mã công khai."));
}

// 7
children.push(h1("7. Nhận xét"));
children.push(bullet([b("Session base và Session 1 tái lập được: "), "OA/AA/Kappa bằng hoặc cao hơn bài báo khoảng 1 điểm."]));
const gap = (s) => paperRuns.length ? (PAPER[s].OA - mean(paperRuns.map((r) => r.sessions[s].OA))).toFixed(1) : "?";
const apd2 = paperRuns.length ? mean(paperRuns.map((r) => r.sessions[2].APD)).toFixed(3) : "?";
children.push(bullet([b("Session 2 và 3 thấp hơn bài báo rõ rệt "), `(OA trung bình thấp hơn ${gap(2)} điểm ở Session 2 và ${gap(3)} điểm ở Session 3). Nguyên nhân chính là quên thảm hoạ khi thêm Highway, Railway, Parking lot 1: lớp Commercial và Road giảm mạnh ở Session 2 (APD trung bình ${apd2} so với 0,1142 của bài báo). Các lớp này giống nhau về phổ và độ cao, trong khi chỉ còn 2 exemplar/lớp. Độ dao động giữa các seed cũng lớn ở các session cuối.`]));
if (origRuns.length) children.push(bullet([b("Mã gốc nguyên trạng "), `(Bảng 7) cho OA cuối ${origRuns[0].sessions[origRuns[0].sessions.length - 1].OA.toFixed(2)} % và quên mạnh ngay từ Session 1 (APD ${origRuns[0].sessions[1].APD.toFixed(3)}), cũng thấp hơn nhiều so với Table IV.`]));
children.push(bullet([b("Bộ phân loại NME "), "cho OA cao hơn fc vài điểm ở Session 2–3 nhưng vẫn chưa đạt mức của bài báo."]));
children.push(bullet([b("Kết luận tạm thời: "), "mã công khai không phải là cấu hình đã tạo ra Table IV. Sau khi đưa về đúng mô tả của bài báo và sửa lỗi optimizer, kết quả khớp ở các session đầu nhưng vẫn thấp hơn ở các session cuối. Những yếu tố bài báo không công bố (epoch, lr, trọng số loss ở session tăng dần; giao thức đánh giá) là nghi vấn chính."]));

// 8
children.push(h1("8. Việc tiếp theo"));
children.push(numbered("Thử nghiệm độ nhạy cho các giả định chưa rõ (lr/epoch session tăng dần, đánh giá bằng NME) trên Houston."));
children.push(numbered("Mở issue trên GitHub của tác giả hỏi cấu hình chính xác của Table IV và báo lỗi optimizer (fc, BiCMP, task token không được cập nhật)."));
children.push(numbered("Mở rộng sang MUUFL (3 mẫu/lớp) và Augsburg (5 mẫu/lớp) — cần thêm bộ đọc dữ liệu và xác minh đúng phiên bản dataset."));
children.push(numbered("Các thí nghiệm còn lại trong danh sách 56 run: 6 baseline replay qua PyCIL, DualPrompt / Adam-adapter, ablation thành phần, top-k, độ dài prompt, prompt đơn modality, dung lượng bộ nhớ, thứ tự lớp."));

// 9
children.push(h1("9. Cách chạy lại"));
children.push(...codeBlock([
  "conda activate cil_rs && export PYTHONNOUSERSITE=1",
  "cd ~/BiCMPT/my_cil && git checkout paper-config",
  "python -m scripts.prepare_houston --raw-dir data_houston/raw/Houston2013 \\",
  "    --hsi-file HSI.mat --lidar-file LiDAR.mat --train-file TRLabel.mat --test-file TSLabel.mat \\",
  "    --pca 40 --split random --train-per-class 180 --seed 0 --out-dir data_houston/paper_seed0",
  "python -m scripts.run_cil --preset paper --data-dir data_houston/paper_seed0 --seed 0",
  "# mã gốc: python -m scripts.run_cil --preset original --data-dir data_houston/original_dfc",
  "# cập nhật báo cáo: cd ~/BiCMPT/report && node make_report.js",
]));

// ---------- document ----------
const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { font: FONT, size: 34, bold: true, color: "1F3864" }, paragraph: { spacing: { before: 0, after: 200 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 28, bold: true, color: "1F3864" }, paragraph: { spacing: { before: 280, after: 120 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 25, bold: true, color: "2F5496" }, paragraph: { spacing: { before: 200, after: 100 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 283 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 340 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Trang ", PageNumber.CURRENT], size: 18 })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, `(${paperRuns.length} paper runs, ${origRuns.length} original runs)`); });
