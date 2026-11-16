// Generate the reproduction report (Vietnamese) from my_cil/results/*.json
// Usage: node make_report.js   ->  BaoCao_TaiLap_BiCM-Prompt.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, LevelFormat, BorderStyle, PageNumber, Footer,
} = require("docx");

const ROOT = path.join(__dirname, "..", "my_cil");
const RESULTS = path.join(ROOT, "results");
const OUT = path.join(__dirname, "BaoCao_TaiLap_BiCM-Prompt.docx");

// ---------- data ----------
// Paper values: Table IV, row "Ours". Test counts: Tables I-III.
const DATASETS = [
  {
    key: "houston", title: "Houston 2013", prefix: "paper_seed", data: "data_houston", paperTable: "Table I",
    modality: "HSI + LiDAR (1 kênh)", split: "6 + 3 + 3 + 3", trainPerClass: 180, memory: "K = 30 (2 mẫu/lớp ở session cuối)",
    paper: [
      { OA: 96.96, AA: 97.55, K: 96.35, APD: null },
      { OA: 93.14, AA: 94.09, K: 92.28, APD: 0.0528 },
      { OA: 82.85, AA: 85.13, K: 81.26, APD: 0.1142 },
      { OA: 77.39, AA: 82.09, K: 75.56, APD: 0.0576 },
    ],
    classes: ["Healthy grass", "Stressed grass", "Synthetic grass", "Trees", "Soil", "Water", "Residential", "Commercial",
      "Road", "Highway", "Railway", "Parking lot 1", "Parking lot 2", "Tennis court", "Running track"],
    paperTest: [1071, 1074, 517, 1064, 1062, 145, 1088, 1064, 1072, 1047, 1055, 1053, 289, 248, 480],
  },
  {
    key: "muufl", title: "MUUFL Gulfport", prefix: "muufl_paper_seed", data: "data_muufl", paperTable: "Table II",
    modality: "HSI + LiDAR (1 kênh)", split: "5 + 2 + 2 + 2", trainPerClass: 100, memory: "K = 33 (3 mẫu/lớp ở session cuối)",
    paper: [
      { OA: 95.45, AA: 97.70, K: 92.19, APD: null },
      { OA: 77.96, AA: 73.10, K: 68.02, APD: 0.3510 },
      { OA: 69.96, AA: 72.16, K: 62.92, APD: 0.0807 },
      { OA: 73.97, AA: 67.02, K: 65.46, APD: 0.1003 },
    ],
    classes: ["Trees", "Mostly grass", "Mixed ground", "Dirt and sand", "Roads", "Water", "Building shadows", "Buildings",
      "Sidewalks", "Yellow curbs", "Cloth panels"],
    paperTest: [19674, 3545, 5765, 1467, 5599, 311, 1813, 5219, 1092, 71, 144],
  },
  {
    key: "augsburg", title: "Augsburg", prefix: "augsburg_paper_seed", data: "data_augsburg", paperTable: "Table III",
    modality: "HSI + SAR (4 kênh)", split: "3 + 2 + 2", trainPerClass: 150, memory: "K = 35 (5 mẫu/lớp ở session cuối)",
    paper: [
      { OA: 92.83, AA: 93.82, K: 88.43, APD: null },
      { OA: 85.99, AA: 67.48, K: 65.83, APD: 0.3748 },
      { OA: 75.87, AA: 58.73, K: 66.45, APD: 0.2431 },
    ],
    classes: ["Water", "Commercial Area", "Allotment", "Low Plants", "Industrial Area", "Residential Area", "Forest"],
    paperTest: [1380, 1495, 425, 26707, 3710, 30179, 13357],
  },
];

function loadRuns(prefix) {
  if (!fs.existsSync(RESULTS)) return [];
  return fs.readdirSync(RESULTS)
    .filter((f) => f.startsWith(prefix) && /seed\d+\.json$/.test(f))
    .sort()
    .map((f) => ({ name: f.replace(".json", ""), seed: f.match(/seed(\d+)/)[1], ...JSON.parse(fs.readFileSync(path.join(RESULTS, f))) }));
}
for (const d of DATASETS) {
  d.runs = loadRuns(d.prefix);
  const metaPath = path.join(ROOT, d.data, "paper_seed0", "meta.json");
  d.meta = fs.existsSync(metaPath) ? JSON.parse(fs.readFileSync(metaPath)) : null;
  d.paperAvg = ["OA", "AA", "K"].reduce((o, k) => ({ ...o, [k]: d.paper.reduce((a, s) => a + s[k], 0) / d.paper.length }), {});
}
const origRuns = loadRuns("original_seed");
const LIDAR_VARIANTS = [
  ["first_dem", "Độ cao first return (mặc định)", 1, "muufl_paper_seed0"],
  ["last_dem", "Độ cao last return", 1, "muufl_lidar_last_dem_seed0"],
  ["first_last_dem", "Độ cao first + last return", 2, "muufl_lidar_first_last_dem_seed0"],
  ["all4", "Độ cao + cường độ, first + last", 4, "muufl_lidar_all4_seed0"],
].map(([k, label, ch, file]) => {
  const f = path.join(RESULTS, file + ".json");
  return { k, label, ch, run: fs.existsSync(f) ? JSON.parse(fs.readFileSync(f)) : null };
});

const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const std = (a) => (a.length < 2 ? 0 : Math.sqrt(a.reduce((s, x) => s + (x - mean(a)) ** 2, 0) / (a.length - 1)));
const f2 = (x) => (x === null || x === undefined ? "–" : x.toFixed(2));
const f4 = (x) => (x === null || x === undefined ? "–" : x.toFixed(4));
const ms = (a, d = 2) => (a.length === 0 ? "–" : a.length === 1 ? a[0].toFixed(d) : `${mean(a).toFixed(d)} ± ${std(a).toFixed(d)}`);
const sname = (s) => (s === 0 ? "Base" : `Session ${s}`);

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
let tableNo = 0;
// Every caption states which table of the paper it corresponds to (or that it has none).
const caption = (text, paperRef) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 160, after: 80 }, keepNext: true,
  children: [new TextRun({ text: `Bảng ${++tableNo}. ${text}`, bold: true, size: 22 }),
    new TextRun({ text: ` [${paperRef}]`, italics: true, size: 20, color: "595959" })],
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
const done = DATASETS.filter((d) => d.runs.length).map((d) => `${d.title} (${d.runs.length} seed)`).join(", ");
children.push(p(`Ngày cập nhật: ${new Date().toISOString().slice(0, 10)} — Phạm vi: phương pháp đề xuất ("Ours") trên ${done}`, { align: AlignmentType.CENTER }));

// 1
children.push(h1("1. Mục tiêu và phạm vi"));
children.push(p("Mục tiêu là chạy lại toàn bộ quy trình của bài báo, từ dữ liệu gốc → tiền xử lý → huấn luyện tăng dần → đánh giá, bằng mã nguồn công khai của tác giả, sau đó so sánh với số liệu của bài báo."));
children.push(p("Thứ tự thực hiện: (1) phương pháp đề xuất trên cả ba bộ dữ liệu của bài báo — dòng \"Ours\" của Table IV (phần đã xong, trình bày trong báo cáo này); (2) sáu baseline có replay của Table IV qua PyCIL; (3) DualPrompt và Adam-adapter (Table V); (4) ablation (Table VI) và các phân tích khác."));
children.push(p([b("Cách đọc số bảng: "), "các bảng trong báo cáo được đánh số riêng (Bảng 1, 2, …). Cuối mỗi tiêu đề bảng có ghi trong ngoặc vuông bảng tương ứng của bài báo, hoặc \"không có trong bài báo\" nếu đó là phân tích bổ sung của người tái lập."]));

// 2
children.push(h1("2. Mã nguồn và môi trường"));
children.push(bullet([b("Mã nguồn: "), "repo gốc github.com/Jiahuiqu/BiCMPT, được sao chép (không giữ lịch sử) sang github.com/VuHa123/CIL_Remoate_sensing."]));
children.push(bullet([b("Nhánh git: "), code("original-code"), " = mã gốc (chỉ thêm pyproject.toml, .gitignore); ", code("paper-config"), " = cấu hình theo bài báo + sửa lỗi (mục 4) + hỗ trợ MUUFL/Augsburg."]));
children.push(bullet([b("Phần cứng: "), "NVIDIA GeForce RTX 3060 Laptop (6 GB), driver 575.64.03 (CUDA 12.9)."]));
children.push(bullet([b("Môi trường: "), "conda env ", code("cil_rs"), ", Python 3.10, PyTorch 2.5.1+cu121, timm 1.0.31, NumPy 2.2.6, scikit-learn 1.7.2, SciPy 1.15.3, imbalanced-learn. Đặt PYTHONNOUSERSITE=1 để tách khỏi gói trong ~/.local."]));
children.push(bullet([b("Lưu ý cài đặt: "), "repo gốc thiếu pyproject.toml nên lệnh pip install -e . trong README không chạy được; đã bổ sung. Khi cài timm, pip tự nâng PyTorch lên bản CUDA 13 không tương thích driver, phải ghim lại torch 2.5.1/cu121."]));
children.push(bullet([b("Kiểm thử: "), "5/5 unit test của repo đạt; chạy thử toàn bộ quy trình (1 epoch) trên mỗi bộ dữ liệu trước khi chạy thật."]));

// 3
children.push(h1("3. Dữ liệu"));
children.push(h2("3.1. Nguồn tải và cách xác minh"));
children.push(p("Bài báo không ghi link tải dữ liệu. Vì vậy dữ liệu được tải từ một bản phân phối công khai, rồi xác minh là đúng bộ dữ liệu bài báo dùng bằng cách đối chiếu kích thước ảnh, số band, số lớp và số pixel có nhãn của từng lớp với Table I–III. Nguồn tải chỉ là nơi lưu trữ; nội dung là bộ dữ liệu gốc."));
children.push(caption("Nguồn dữ liệu đã dùng", "không có trong bài báo"));
children.push(table(["Bộ dữ liệu", "Nguồn tải", "Nội dung", "Xác minh"], [
  ["Houston 2013", "github.com/songyz2019/rs-fusion-datasets-dist, release v1.0.0, houston2013-mmr.zip (161 MB). Dữ liệu gốc: 2013 IEEE GRSS Data Fusion Contest",
    "HSI 349×1905×144, LiDAR 1 kênh, 15 lớp", "SHA-256 khớp giá trị công bố; số mẫu test khớp Table I ở cả 15 lớp"],
  ["MUUFL Gulfport", "github.com/GatorSense/MUUFLGulfport (nguồn chính thức), MUUFLGulfportSceneLabels/muufl_gulfport_campus_1_hsi_220_label.mat (16 MB)",
    "HSI 325×220×64, LiDAR 2 lần thu × (độ cao, cường độ), 11 lớp, 53 687 pixel có nhãn", "Kích thước, số lớp, tổng pixel có nhãn khớp mô tả của bài báo; số train khớp Table II"],
  ["Augsburg", "github.com/songyz2019/rs-fusion-datasets-dist, release v1.0.0, augsburg-ouc.zip (52 MB). Bộ HSI–SAR của Hong et al.",
    "HSI 332×485×180 (HySpex), SAR 4 kênh (Sentinel-1), 7 lớp", "Số mẫu test khớp Table III ở 6/7 lớp và khớp tổng 77 244 (lớp Industrial: 3 701 so với 3 710 — nhiều khả năng lỗi đánh máy vì tổng vẫn khớp)"],
], [1.4, 3.4, 2.6, 2.8]));
children.push(p("SHA-256 (rút gọn) của các file đã dùng:"));
children.push(table(["File", "SHA-256"], [
  ["houston2013-mmr.zip", "7a2d719d…ae1fd762"],
  ["muufl_gulfport_campus_1_hsi_220_label.mat", "69420a72…feb59edc"],
  ["augsburg-ouc.zip", "4d853be6…3a7ab6ae"],
  ["Augsburg/augsburg_hsi.mat", "2f13839d…73946583"],
  ["Augsburg/augsburg_sar.mat", "cd1dbfad…e3f8e82b"],
  ["Augsburg/augsburg_gt.mat", "065df712…b9eaacbe"],
], [5, 4]));
children.push(p([b("Về các link đã được gợi ý: "), code("GatorSense/MUUFLGulfport"), " là nguồn chính thức và đã được dùng cho MUUFL. ",
  code("kaggle …/houston2013-hsi-lidar-dataset"), " và ", code("YangHuihan219/Houston2013"), " là bản đăng lại của cá nhân, không cần thiết vì đã có bản xác minh bằng checksum và Table I. ",
  code("zhu-xlab/augsburg_Multimodal_Data_Set_MDaS"), " là bộ MDaS (2023), ", b("khác"), " bộ Augsburg HSI–SAR 332×485×180, 7 lớp mà bài báo dùng, nên không dùng."]));

children.push(h2("3.2. Tiền xử lý theo bài báo"));
children.push(bullet("PCA giảm HSI về 40 thành phần (giữ ≥ 99,98 % phương sai); LiDAR/SAR giữ nguyên số kênh; chuẩn hoá min–max từng kênh."));
children.push(bullet("Cắt patch 11 × 11 quanh mỗi pixel có nhãn."));
children.push(bullet("Chọn ngẫu nhiên số pixel train mỗi lớp theo Table I–III (Houston 180, MUUFL 100, Augsburg 150), phần còn lại làm test. Seed của phép chia = seed của lần chạy (0, 1, 2)."));
children.push(bullet([b("Houston: "), "gộp nhãn train + test của bộ DFC2013 (2 832 + 12 197 pixel) trước khi chia lại."]));
children.push(bullet([b("Augsburg: "), "thứ tự lớp trong file (Forest, Residential, Industrial, Low Plants, Allotment, Commercial, Water) được đảo lại cho đúng thứ tự của Table III."]));
children.push(bullet([b("MUUFL — LiDAR: "), "file có 2 ảnh LiDAR (first return, last return), mỗi ảnh có 2 band (độ cao DEM, cường độ) = 4 kênh. Bài báo ghi LiDAR 11×11×1 nhưng không nói kênh nào; mặc định dùng độ cao first return. Ba lựa chọn khác được chạy thêm để so sánh (mục 6.4)."]));
children.push(bullet([b("MUUFL — tập test: "), "bài báo ghi 44 700 mẫu test, ít hơn số pixel còn lại sau khi lấy train (52 587, khoảng 85 %) ở mọi lớp, và không nói cách lọc. Tái lập dùng toàn bộ 52 587 pixel."]));

for (const d of DATASETS) {
  if (!d.meta) continue;
  const rows = d.classes.map((n, k) => {
    const ours = d.meta.test_per_class[String(k + 1)];
    return [k + 1, n, d.meta.train_per_class[String(k + 1)], ours, d.paperTest[k], ours === d.paperTest[k] ? "✓" : "✗"];
  });
  rows.push(["", "Tổng", Object.values(d.meta.train_per_class).reduce((a, c) => a + c, 0),
    Object.values(d.meta.test_per_class).reduce((a, c) => a + c, 0), d.paperTest.reduce((a, c) => a + c, 0), ""]);
  children.push(caption(`${d.title}: số mẫu mỗi lớp (seed 0)`, `đối chiếu ${d.paperTable}`));
  children.push(table(["#", "Lớp", "Train", "Test (tái lập)", "Test (bài báo)", "Khớp"], rows, [0.6, 2.6, 1, 1.5, 1.5, 0.8],
    { rowFill: (ri) => (ri === rows.length - 1 ? "F2F2F2" : undefined) }));
}

// 4
children.push(h1("4. Đối chiếu mã nguồn với bài báo"));
children.push(p("Mã công khai chỉ có kịch bản cho Houston 2013, và không trùng với cấu hình mô tả trong bài báo:"));
children.push(caption("Khác biệt giữa mã gốc và bài báo (Houston 2013)", "không có trong bài báo"));
children.push(table(["Hạng mục", "Bài báo", "Mã gốc (train_incremental.py)", "Cấu hình tái lập"], [
  ["Chia lớp", "6 + 3 + 3 + 3", "7 + 3 + 3 + 2", "6 + 3 + 3 + 3"],
  ["Thứ tự lớp", "Thứ tự gốc (lớp 1–6 là base)", "Hoán vị seed 4 (base = lớp gốc 2,3,4,5,8,12,13)", "Thứ tự gốc"],
  ["Kênh HSI", "PCA 40", "144 band", "PCA 40"],
  ["Chia train/test", "180 mẫu/lớp ngẫu nhiên", "Đọc từ file .mat (không rõ)", "180 mẫu/lớp ngẫu nhiên"],
  ["Bộ nhớ replay", "2 mẫu/lớp ở cuối", "K = 30; base m = 6, sau đó m = K // s", "m = 30 // số lớp đã thấy (5, 3, 2, 2)"],
  ["Huấn luyện base", "Adam, lr 1e-3, batch 32", "Adam, lr 1e-4, batch 16", "Adam, lr 1e-3, batch 32"],
  ["Prompt pool / top-k / độ dài", "30 / 3 / 5", "30 / 3 / 5", "30 / 3 / 5"],
  ["Seed", "–", "Chỉ random.seed(11)", "random, numpy, torch"],
], [2, 2.4, 3, 2.6]));
children.push(p("Với MUUFL và Augsburg, mã gốc cố định nhánh thứ hai có 1 kênh; đã thêm tham số aux_channels để nhận SAR 4 kênh. Cách chia lớp lấy theo Table II–III (MUUFL 5 + 2 + 2 + 2, Augsburg 3 + 2 + 2); dung lượng bộ nhớ chọn K = 33 và K = 35 để session cuối còn đúng 3 và 5 mẫu/lớp như bài báo."));
children.push(h2("4.1. Lỗi trong mã gốc"));
children.push(p("Kiểm tra trực tiếp tập tham số mà optimizer cập nhật ở từng session cho thấy:"));
children.push(bullet([b("Backbone có được đóng băng từ session 1: "), "đúng như bài báo — ViT của cả hai nhánh có requires_grad = False; chỉ prompt, PGKA (TaskAware, trừ khối grouping), BN và fc được học."]));
children.push(bullet([b("Optimizer chỉ được tạo một lần trong __init__: "), "lớp phân loại fc (được tạo lại khi thêm lớp), các mạng chiếu hai chiều BiCMP (stage_mapping_net) và task token G (thậm chí không phải nn.Parameter) ", b("không bao giờ được cập nhật"), ". Như vậy mô hình gốc dùng fc khởi tạo ngẫu nhiên cố định và BiCMP không được huấn luyện."]));
children.push(bullet([b("Trọng số lớp cho OT: "), "được tính theo thứ tự đã xáo trộn của DataLoader nhưng lại tra bằng chỉ số của dataset, nên gán trọng số sai mẫu; đồng thời tính lại trên toàn bộ dữ liệu ở mỗi batch (rất chậm)."]));
children.push(p([b("Sửa: "), "cờ fix_optimizer (bật trong preset paper) tạo lại Adam + cosine scheduler ở mỗi session cho đúng các tham số đang requires_grad, đăng ký task token là tham số (token cũ đóng băng, token mới được học), và tính trọng số OT theo thứ tự dataset. Với cờ này, số tham số học được ở các session tăng dần là ≈ 0,850 M, sát với 0,843 M mà bài báo báo cáo. Preset original giữ nguyên hành vi gốc để so sánh."]));

// 5
children.push(h1("5. Cấu hình chạy và giả định"));
children.push(bullet("Base: 30 epoch, Adam lr 1e-3, batch 32 (hiểu \"pretraining\" trong bài báo là session base, vì mã không có bước pretrain tương thích: scripts/pretrain_backbone.py dùng ViT 768 chiều, khác backbone 128 chiều của mô hình)."));
children.push(bullet("Session tăng dần: 35 epoch, Adam lr 1e-4, batch 16, weight decay 1e-4, cosine annealing — giữ mặc định của mã (bài báo không nêu)."));
children.push(bullet("Trọng số các hàm loss: giữ như mã (CE + 0,01·MI + sim + aux + distill, CE có trọng số OT)."));
children.push(bullet([b("Hai cách phân loại: "), "fc = lớp tuyến tính cuối mạng; NME = gán mẫu vào lớp có trung bình đặc trưng exemplar gần nhất (cách của iCaRL). Mã gốc dùng NME ở session base và fc ở các session sau; bài báo không nói dùng cách nào. Số liệu chính trong báo cáo theo mã gốc; NME được ghi kèm để thấy ảnh hưởng của lựa chọn này."]));
children.push(bullet("APD theo công thức (20) của bài báo, trùng với hàm calculate_forgetting_rate của mã."));
children.push(bullet("Mỗi bộ dữ liệu chạy 3 seed (0, 1, 2), mỗi seed dùng một phép chia train/test ngẫu nhiên khác nhau."));

// 6
children.push(h1("6. Kết quả phương pháp đề xuất"));
let sub = 0;
for (const d of DATASETS) {
  sub += 1;
  children.push(h2(`6.${sub}. ${d.title}`));
  const runs = d.runs;
  if (!runs.length) { children.push(p("Chưa có kết quả.")); continue; }
  const nS = d.paper.length;
  children.push(p(`Dữ liệu ${d.modality}, chia lớp ${d.split}, ${d.trainPerClass} mẫu train/lớp, bộ nhớ ${d.memory}. Đã chạy ${runs.length} seed; mỗi lần chạy khoảng ${Math.round(mean(runs.map((r) => r.total_seconds)) / 60)} phút trên RTX 3060.`));
  const rows = [];
  for (let s = 0; s < nS; s++) {
    const pick = (k) => runs.map((r) => r.sessions[s] && r.sessions[s][k]).filter((x) => x !== undefined && x !== null);
    rows.push([sname(s), runs[0].sessions[s].seen_classes,
      ms(pick("OA")), f2(d.paper[s].OA), ms(pick("AA")), f2(d.paper[s].AA),
      ms(pick("Kappa")), f2(d.paper[s].K), s === 0 ? "–" : ms(pick("APD"), 4), f4(d.paper[s].APD)]);
  }
  const avg = (k) => runs.map((r) => r.average[k]);
  rows.push(["Trung bình", "", ms(avg("OA")), f2(d.paperAvg.OA), ms(avg("AA")), f2(d.paperAvg.AA), ms(avg("Kappa")), f2(d.paperAvg.K), "", ""]);
  children.push(caption(`${d.title}: tái lập (trung bình ± độ lệch chuẩn qua ${runs.length} seed) so với bài báo`, "đối chiếu Table IV, dòng Ours"));
  children.push(table(["Session", "Số lớp", "OA tái lập", "OA bài báo", "AA tái lập", "AA bài báo", "Kappa tái lập", "Kappa bài báo", "APD tái lập", "APD bài báo"],
    rows, [1.5, 0.8, 1.5, 1.1, 1.5, 1.1, 1.5, 1.1, 1.4, 1.1], { rowFill: (ri) => (ri === nS ? "F2F2F2" : undefined) }));

  children.push(caption(`${d.title}: kết quả từng seed (OA / AA / Kappa, %)`, "không có trong bài báo"));
  children.push(table(["Seed", ...d.paper.map((_, s) => sname(s)), "TB OA"],
    runs.map((r) => [r.seed, ...d.paper.map((_, s) => r.sessions[s] ? `${f2(r.sessions[s].OA)} / ${f2(r.sessions[s].AA)} / ${f2(r.sessions[s].Kappa)}` : "–"), f2(r.average.OA)]),
    [0.7, ...d.paper.map(() => 2.2), 1.1]));

  children.push(caption(`${d.title}: OA theo hai cách phân loại fc và NME (trung bình các seed, %)`, "không có trong bài báo — phân tích bổ sung"));
  children.push(table(["Session", "fc", "NME", "OA bài báo"],
    d.paper.map((x, s) => [sname(s), ms(runs.map((r) => r.sessions[s].fc.OA)), ms(runs.map((r) => r.sessions[s].nme.OA)), f2(x.OA)]),
    [2, 3, 3, 2]));

  const header = ["#", "Lớp", ...d.paper.map((_, s) => `Sau ${sname(s)}`)];
  const pc = d.classes.map((n, k) => [k + 1, n, ...d.paper.map((_, s) => {
    const key = String(k);
    const vals = runs.map((r) => r.sessions[s].per_class_acc[key]).filter((x) => x !== undefined).map((x) => x * 100);
    return vals.length ? ms(vals, 1) : "–";
  })]);
  children.push(caption(`${d.title}: độ chính xác từng lớp (%, cách phân loại chính) — để tìm lớp bị quên`, "không có trong bài báo — phân tích bổ sung"));
  children.push(table(header, pc, [0.5, 2.4, ...d.paper.map(() => 1.6)]));
}

// LiDAR variants
const lv = LIDAR_VARIANTS.filter((v) => v.run);
sub += 1;
children.push(h2(`6.${sub}. MUUFL: thử các kênh LiDAR khác nhau`));
children.push(p("Cùng phép chia seed 0, chỉ thay kênh LiDAR đưa vào nhánh thứ hai. Đây là thí nghiệm bổ sung để kiểm tra giả định chọn kênh."));
if (lv.length) {
  children.push(caption("MUUFL seed 0: OA theo lựa chọn kênh LiDAR (%)", "không có trong bài báo — phân tích bổ sung"));
  const nS = DATASETS[1].paper.length;
  children.push(table(["Kênh LiDAR", "Số kênh", ...Array.from({ length: nS }, (_, s) => sname(s)), "TB OA", "TB AA", "TB Kappa"],
    lv.map((v) => [v.label, v.ch, ...v.run.sessions.map((s) => f2(s.OA)), f2(v.run.average.OA), f2(v.run.average.AA), f2(v.run.average.Kappa)]),
    [3, 0.9, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]));
  if (lv.length < LIDAR_VARIANTS.length) children.push(p(`Còn ${LIDAR_VARIANTS.length - lv.length} biến thể đang chạy.`));
} else {
  children.push(p("Đang chạy."));
}

if (origRuns.length) {
  const r = origRuns[0];
  sub += 1;
  children.push(h2(`6.${sub}. Mã gốc nguyên trạng trên Houston 2013`));
  children.push(caption("Mã gốc nguyên trạng (preset original: 7+3+3+2, 144 band, bộ chia DFC2013, seed 0)", "không so trực tiếp được với Table IV"));
  children.push(table(["Session", "Số lớp", "OA", "AA", "Kappa", "APD"],
    r.sessions.map((s) => [sname(s.session), s.seen_classes, f2(s.OA), f2(s.AA), f2(s.Kappa), s.session ? f4(s.APD) : "–"]),
    [1.5, 1, 1.3, 1.3, 1.3, 1.3]));
  children.push(p("Cấu hình này khác bài báo về số lớp mỗi session, thứ tự lớp, dữ liệu đầu vào và phép chia; chỉ dùng để thấy hành vi của mã công khai."));
}

// 7
children.push(h1("7. Nhận xét"));
const lastGap = (d) => {
  const s = d.paper.length - 1;
  return (mean(d.runs.map((r) => r.sessions[s].OA)) - d.paper[s].OA).toFixed(1);
};
const lastNmeGap = (d) => {
  const s = d.paper.length - 1;
  return (mean(d.runs.map((r) => r.sessions[s].nme.OA)) - d.paper[s].OA).toFixed(1);
};
for (const d of DATASETS.filter((x) => x.runs.length)) {
  const s0 = (mean(d.runs.map((r) => r.sessions[0].OA)) - d.paper[0].OA).toFixed(1);
  children.push(bullet([b(`${d.title}: `), `OA session base lệch ${s0} điểm so với bài báo; OA session cuối lệch ${lastGap(d)} điểm với fc và ${lastNmeGap(d)} điểm nếu dùng NME (số âm = thấp hơn bài báo).`]));
}
children.push(bullet([b("Xu hướng chung: "), "session base và session tăng dần đầu tiên tái lập được gần với bài báo; các session sau thấp hơn, chủ yếu do một vài lớp cũ bị quên mạnh (xem bảng độ chính xác từng lớp). Độ dao động giữa các seed lớn ở các session cuối."]));
children.push(bullet([b("Cách phân loại: "), "ở các session cuối, NME thường cho OA cao hơn fc. Bài báo không nói dùng cách nào; đây là một nghi vấn chính khi so sánh."]));
if (origRuns.length) children.push(bullet([b("Mã gốc nguyên trạng "), `cho OA cuối ${origRuns[0].sessions[origRuns[0].sessions.length - 1].OA.toFixed(2)} % trên Houston và quên mạnh ngay từ Session 1 (APD ${origRuns[0].sessions[1].APD.toFixed(3)}).`]));
children.push(bullet([b("Kết luận tạm thời: "), "mã công khai không phải là cấu hình đã tạo ra Table IV. Sau khi đưa về đúng mô tả của bài báo và sửa lỗi optimizer, kết quả khớp ở các session đầu nhưng vẫn lệch ở các session cuối. Những yếu tố bài báo không công bố (epoch, lr, trọng số loss ở session tăng dần; cách phân loại khi đánh giá) là nghi vấn chính."]));

// 8
children.push(h1("8. Việc tiếp theo"));
children.push(numbered("Sáu baseline có replay của Table IV (LwF, iCaRL, PodNet, WA, Foster, BEEF) qua PyCIL trên cả ba bộ dữ liệu."));
children.push(numbered("DualPrompt và Adam-adapter (Table V)."));
children.push(numbered("Ablation thành phần (Table VI) và các phân tích khác: top-k, độ dài prompt, prompt đơn modality, dung lượng bộ nhớ, thứ tự lớp."));
children.push(numbered("Thử nghiệm độ nhạy cho các giả định chưa rõ (lr/epoch session tăng dần, cách phân loại)."));
children.push(numbered("Cân nhắc mở issue trên GitHub của tác giả hỏi cấu hình chính xác của Table IV và báo lỗi optimizer."));

// 9
children.push(h1("9. Cách chạy lại"));
children.push(...codeBlock([
  "conda activate cil_rs && export PYTHONNOUSERSITE=1",
  "cd ~/BiCMPT/my_cil && git checkout paper-config",
  "# Houston 2013",
  "python -m scripts.prepare_houston --raw-dir data_houston/raw/Houston2013 \\",
  "    --hsi-file HSI.mat --lidar-file LiDAR.mat --train-file TRLabel.mat --test-file TSLabel.mat \\",
  "    --pca 40 --split random --train-per-class 180 --seed 0 --out-dir data_houston/paper_seed0",
  "python -m scripts.run_cil --preset paper --data-dir data_houston/paper_seed0 --seed 0",
  "# MUUFL / Augsburg",
  "python -m scripts.prepare_multisource --dataset muufl --raw-dir data_muufl/raw --out-dir data_muufl/paper_seed0 --seed 0",
  "python -m scripts.run_cil --preset paper --dataset muufl --data-dir data_muufl/paper_seed0 --seed 0",
  "python -m scripts.prepare_multisource --dataset augsburg --raw-dir data_augsburg/raw --out-dir data_augsburg/paper_seed0 --seed 0",
  "python -m scripts.run_cil --preset paper --dataset augsburg --data-dir data_augsburg/paper_seed0 --seed 0",
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

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log("wrote", OUT, DATASETS.map((d) => `${d.key}:${d.runs.length}`).join(" "), `lidar:${lv.length}`, `original:${origRuns.length}`);
});
