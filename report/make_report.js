// Generate the reproduction report (Vietnamese) from my_cil/results/*.json
// Usage: node make_report.js   ->  BaoCao_TaiLap_BiCM-Prompt.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, LevelFormat, BorderStyle, PageNumber, Footer, PageOrientation,
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

function loadRuns(prefix, dir = RESULTS) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir)
    .filter((f) => f.startsWith(prefix) && /seed\d+\.json$/.test(f))
    .sort()
    .map((f) => ({ name: f.replace(".json", ""), seed: f.match(/seed(\d+)/)[1], ...JSON.parse(fs.readFileSync(path.join(dir, f))) }));
}
for (const d of DATASETS) {
  d.runs = loadRuns(d.prefix);
  d.nmeRuns = loadRuns(d.prefix, path.join(RESULTS, "eval_nme"));
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

// ---------- Table IV (all methods) ----------
// Paper Table IV transcribed from page 7084: per dataset, per session [APD, OA, AA, Kappa].
const T4_METHODS = [
  { key: "lwf", name: "LwF" }, { key: "icarl", name: "iCaRL" }, { key: "podnet", name: "PodNet" },
  { key: "wa", name: "WA" }, { key: "beef", name: "BEEF" }, { key: "foster", name: "FOSTER" }, { key: "ours", name: "Ours (BiCM-Prompt)" },
];
const P = (apd, oa, aa, k) => ({ APD: apd, OA: oa, AA: aa, K: k });
const PAPER_T4 = {
  lwf: {
    houston: [P(null, 95.52, 96.25, 94.63), P(0.5696, 50.87, 50.27, 44.69), P(0.3152, 32.67, 32.21, 26.07), P(0.1504, 22.77, 22.57, 16.13)],
    muufl: [P(null, 90.13, 87.67, 83.30), P(0.6689, 63.32, 35.31, 34.88), P(0.2152, 52.42, 20.44, 19.28), P(0.0982, 42.95, 21.65, 37.49)],
    augsburg: [P(null, 91.92, 93.15, 86.98), P(0.7537, 89.09, 49.60, 70.10), P(0.0649, 67.53, 45.27, 53.48)],
  },
  icarl: {
    houston: [P(null, 96.30, 97.13, 95.40), P(0.1307, 89.37, 88.09, 87.84), P(0.1640, 75.33, 76.53, 72.89), P(0.1264, 65.28, 71.10, 62.78)],
    muufl: [P(null, 92.33, 93.81, 92.13), P(0.4560, 76.38, 62.80, 65.35), P(0.1527, 64.08, 58.59, 54.98), P(0.1130, 72.69, 56.44, 64.31)],
    augsburg: [P(null, 84.32, 86.93, 75.50), P(0.5299, 90.15, 59.74, 72.94), P(0.2878, 60.69, 48.25, 46.29)],
  },
  podnet: {
    houston: [P(null, 97.77, 98.19, 97.32), P(0.0645, 75.27, 77.14, 72.28), P(0.0285, 59.59, 60.49, 56.08), P(0.0032, 58.05, 62.74, 55.00)],
    muufl: [P(null, 95.61, 97.04, 92.40), P(0.2983, 81.90, 73.98, 75.42), P(0.1347, 70.52, 62.02, 63.99), P(0.1300, 54.19, 47.70, 45.44)],
    augsburg: [P(null, 91.81, 93.04, 86.78), P(0.5141, 67.22, 57.86, 40.28), P(0.1106, 44.29, 51.41, 32.08)],
  },
  wa: {
    houston: [P(null, 94.54, 95.52, 93.46), P(0.1731, 85.26, 84.46, 83.41), P(0.2860, 63.01, 65.53, 59.70), P(0.1463, 50.14, 57.45, 46.78)],
    muufl: [P(null, 94.74, 96.84, 91.00), P(0.5048, 78.13, 61.66, 68.22), P(0.2317, 55.20, 51.63, 43.28), P(0.2199, 59.96, 41.79, 45.76)],
    augsburg: [P(null, 93.65, 94.47, 89.74), P(0.6569, 88.72, 56.11, 68.26), P(0.3432, 59.49, 43.49, 42.79)],
  },
  beef: {
    houston: [P(null, 98.15, 98.50, 97.79), P(0.2147, 76.55, 77.77, 73.69), P(0.1528, 57.72, 62.57, 53.78), P(0.0801, 60.80, 67.72, 57.99)],
    muufl: [P(null, 95.53, 97.19, 92.30), P(0.1998, 55.08, 65.41, 43.96), P(0.1169, 43.44, 52.73, 32.79), P(0.0752, 49.95, 43.96, 37.52)],
    augsburg: [P(null, 91.04, 92.14, 85.51), P(0.5782, 90.55, 59.32, 73.72), P(0.1766, 73.25, 55.09, 62.01)],
  },
  foster: {
    houston: [P(null, 99.60, 99.67, 99.51), P(0.0067, 94.42, 95.15, 93.73), P(0.1358, 68.28, 70.90, 65.33), P(0.0335, 67.46, 71.85, 64.89)],
    muufl: [P(null, 95.99, 97.33, 93.06), P(0.0533, 84.57, 89.81, 79.31), P(0.2359, 64.16, 70.36, 57.58), P(0.0445, 70.30, 66.34, 64.40)],
    augsburg: [P(null, 92.16, 93.32, 87.34), P(0.1929, 81.23, 71.72, 57.56), P(0.1800, 58.88, 60.97, 46.22)],
  },
  ours: {
    houston: [P(null, 96.96, 97.55, 96.35), P(0.0528, 93.14, 94.09, 92.28), P(0.1142, 82.85, 85.13, 81.26), P(0.0576, 77.39, 82.09, 75.56)],
    muufl: [P(null, 95.45, 97.70, 92.19), P(0.3510, 77.96, 73.10, 68.02), P(0.0807, 69.96, 72.16, 62.92), P(0.1003, 73.97, 67.02, 65.46)],
    augsburg: [P(null, 92.83, 93.82, 88.43), P(0.3748, 85.99, 67.48, 65.83), P(0.2431, 75.87, 58.73, 66.45)],
  },
};
const BASELINE_DIR = path.join(RESULTS, "baselines");
const T4_SEED = "0";
const BASELINE_TAG = "_b32"; // batch 32 as stated in the paper; "" = PyCIL default batch 128
const baselineRun = (m, ds) => {
  const f = path.join(BASELINE_DIR, `${m}_adam_e1${BASELINE_TAG}_${ds}_seed${T4_SEED}.json`);
  return fs.existsSync(f) ? JSON.parse(fs.readFileSync(f)) : null;
};
const pilotSgd = fs.existsSync(path.join(BASELINE_DIR, "icarl_houston_seed0.json"))
  ? JSON.parse(fs.readFileSync(path.join(BASELINE_DIR, "icarl_houston_seed0.json"))) : null;

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
children.push(p(`Ngày cập nhật: ${new Date().toISOString().slice(0, 10)} — Phạm vi: phương pháp đề xuất ("Ours") trên ${done}; baseline PyCIL seed 0`, { align: AlignmentType.CENTER }));

// 1
children.push(h1("1. Mục tiêu và phạm vi"));
children.push(p("Mục tiêu là chạy lại toàn bộ quy trình của bài báo, từ dữ liệu gốc → tiền xử lý → huấn luyện tăng dần → đánh giá, bằng mã nguồn công khai của tác giả, sau đó so sánh với số liệu của bài báo."));
children.push(p("Thứ tự thực hiện: (1) phương pháp đề xuất trên cả ba bộ dữ liệu của bài báo — dòng \"Ours\" của Table IV (đã xong, 3 seed); (2) sáu baseline có replay của Table IV qua PyCIL (đang chạy, seed 0 trước; mục 6); (3) DualPrompt và Adam-adapter (Table V); (4) ablation (Table VI) và các phân tích khác."));
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
children.push(bullet([b("MUUFL — LiDAR: "), "file có 2 ảnh LiDAR (first return, last return), mỗi ảnh có 2 band (độ cao DEM, cường độ) = 4 kênh. Bài báo ghi LiDAR 11×11×1 nhưng không nói kênh nào; mặc định dùng độ cao first return. Ba lựa chọn khác được chạy thêm để so sánh (mục 7.4)."]));
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
children.push(bullet([b("Hai cách phân loại: "), "fc = lớp tuyến tính cuối mạng; NME = gán mẫu vào lớp có trung bình đặc trưng exemplar gần nhất (cách của iCaRL). Mã gốc dùng NME ở session base và fc ở các session sau; bài báo không nói dùng cách nào. Số liệu chính trong báo cáo theo mã gốc; kết quả khi dùng NME ở mọi session được trình bày riêng."]));
children.push(bullet([b("Bài báo có nhắc đến NME không: "), "không. Phần đánh giá và cài đặt không mô tả cách đưa ra dự đoán khi kiểm thử; bài báo chỉ nói hàm loss là cross-entropy (Eq. 1), tức huấn luyện qua lớp fc. \"Class mean\" chỉ xuất hiện ở phần tổng quan khi giới thiệu iCaRL. Các baseline chạy bằng PyCIL — thư viện này báo cả độ chính xác theo fc (\"CNN\") lẫn theo NME — nhưng bài báo không nói lấy số nào."]));
children.push(bullet("APD theo công thức (20) của bài báo, trùng với hàm calculate_forgetting_rate của mã."));
children.push(bullet("Mỗi bộ dữ liệu chạy 3 seed (0, 1, 2), mỗi seed dùng một phép chia train/test ngẫu nhiên khác nhau."));

// 6 — Table IV replica (landscape section)
const part1 = children.splice(0);
const LAND_W = 16838 - 2 * MARGIN;
const DS_KEYS = ["houston", "muufl", "augsburg"];
const dsByKey = Object.fromEntries(DATASETS.map((d) => [d.key, d]));
const T4_HEAD = { houston: "Houston2013 (2 mẫu/lớp)", muufl: "MUUFL (3 mẫu/lớp)", augsburg: "Augsburg (5 mẫu/lớp)" };
const METRICS = ["APD", "OA", "AA", "K"];

// cell value for the reproduction: kind = "main" (PyCIL "CNN"/fc; Ours: classifier of the code) or "nme"
function reproCell(m, ds, s, kind = "main") {
  let sess;
  if (m === "ours") {
    const r = dsByKey[ds].runs.find((x) => x.seed === T4_SEED);
    sess = r && r.sessions[s];
  } else {
    const r = baselineRun(m, ds);
    sess = r && r.sessions[s];
  }
  if (!sess) return null;
  const src = kind === "nme" ? sess.nme : sess;
  if (!src || src.OA === undefined || src.OA === null) return kind === "nme" ? P(null, null, null, null) : null;
  return P(s === 0 ? null : (src.APD === undefined ? null : src.APD), src.OA, src.AA, src.Kappa);
}
const paperCell = (m, ds, s) => (PAPER_T4[m][ds][s] || null);
const diffCell = (m, ds, s) => {
  const a = reproCell(m, ds, s), pp = paperCell(m, ds, s);
  if (!a || !pp) return null;
  return P(s === 0 ? null : a.APD - pp.APD, a.OA - pp.OA, a.AA - pp.AA, a.K - pp.K);
};
const paramsRepro = (m) => {
  if (m === "ours") return "≈0,85M";
  for (const ds of DS_KEYS) { const r = baselineRun(m, ds); if (r) return `${r.sessions[0].trainable_params_M.toFixed(2)}M`; }
  return "…";
};
const paramsPaper = (m) => (m === "ours" ? "0.843M" : "2.445M");

function t4Table(getCell, { params, signed = false, bold = true, missing = "…" }) {
  const rel = [1.5, 0.9, 1.0, ...Array(12).fill(0.85)];
  const tot = rel.reduce((a, c) => a + c, 0);
  const w = rel.map((x) => Math.floor((x / tot) * LAND_W));
  w[w.length - 1] += LAND_W - w.reduce((a, c) => a + c, 0);
  const SZ = 16;
  const mk = (text, width, o = {}) => new TableCell({
    borders, width: { size: width, type: WidthType.DXA }, columnSpan: o.colSpan, rowSpan: o.rowSpan,
    verticalAlign: "center", margins: { top: 30, bottom: 30, left: 40, right: 40 },
    shading: o.head ? { type: ShadingType.CLEAR, fill: "D9E2F3", color: "auto" } : o.fill ? { type: ShadingType.CLEAR, fill: o.fill, color: "auto" } : undefined,
    children: [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [new TextRun({ text: String(text), bold: o.head || o.bold, size: SZ })] })],
  });
  const fmt = (v, metric) => {
    if (v === null || v === undefined || Number.isNaN(v)) return null;
    const t = metric === "APD" ? v.toFixed(4) : v.toFixed(2);
    return signed && v > 0 ? `+${t}` : t;
  };
  // best value per (dataset, session, metric) among methods with a value (OA/AA/K: max), as in the paper
  const best = {};
  if (bold) for (const ds of DS_KEYS) for (let s = 0; s < 4; s++) for (const k of ["OA", "AA", "K"]) {
    const vals = T4_METHODS.map((mm) => getCell(mm.key, ds, s)).filter(Boolean).map((c) => c[k]);
    if (vals.length > 1) best[`${ds}${s}${k}`] = Math.max(...vals);
  }
  const head1 = new TableRow({ tableHeader: true, children: [
    mk("Phương pháp", w[0], { head: true, rowSpan: 2 }), mk("Tham số học", w[1], { head: true, rowSpan: 2 }), mk("Session", w[2], { head: true, rowSpan: 2 }),
    ...DS_KEYS.map((ds, j) => mk(T4_HEAD[ds], w[3 + 4 * j] * 4, { head: true, colSpan: 4 }))] });
  const head2 = new TableRow({ tableHeader: true, children: DS_KEYS.flatMap((ds, j) => ["APD", "OA", "AA", "Kappa"].map((h, q) => mk(h, w[3 + 4 * j + q], { head: true }))) });
  const rows = [head1, head2];
  for (const mm of T4_METHODS) {
    const fill = mm.key === "ours" ? "F2F2F2" : undefined;
    for (let s = 0; s < 4; s++) {
      const cells = [];
      if (s === 0) {
        cells.push(mk(mm.name, w[0], { rowSpan: 4, bold: mm.key === "ours", fill }));
        cells.push(mk(params(mm.key), w[1], { rowSpan: 4, fill }));
      }
      cells.push(mk(s === 0 ? "Base" : `Session${s}`, w[2], { fill }));
      DS_KEYS.forEach((ds, j) => {
        const nS = dsByKey[ds].paper.length;
        const c = s < nS ? getCell(mm.key, ds, s) : null;
        METRICS.forEach((k, q) => {
          let text;
          if (s >= nS || (k === "APD" && s === 0)) text = "-";
          else if (!c) text = missing;
          else { const t = fmt(c[k], k); text = t === null ? "-" : t; }
          const isBest = bold && c && k !== "APD" && best[`${ds}${s}${k}`] !== undefined && c[k] === best[`${ds}${s}${k}`];
          cells.push(mk(text, w[3 + 4 * j + q], { bold: isBest, fill }));
        });
      });
      rows.push(new TableRow({ cantSplit: true, children: cells }));
    }
  }
  return new Table({ width: { size: LAND_W, type: WidthType.DXA }, columnWidths: w, rows });
}

const nDone = T4_METHODS.filter((mm) => mm.key !== "ours").reduce((a, mm) => a + DS_KEYS.filter((ds) => baselineRun(mm.key, ds)).length, 0);
children.push(h1("6. So sánh với Table IV và Table V: baseline và phương pháp đề xuất trên ba bộ dữ liệu"));
children.push(p(`Bảng dưới đây có cùng bố cục với Table IV của bài báo: mỗi phương pháp một khối dòng (Base, Session1–3), mỗi bộ dữ liệu bốn cột APD / OA / AA / Kappa. Đây là kết quả tham chiếu với một seed (seed ${T4_SEED}) cho mọi phương pháp; phương pháp đề xuất đã có 3 seed (mục 7). Ô "…" là lần chạy chưa xong (hiện có ${nDone}/18 lần chạy baseline). Chữ đậm: giá trị cao nhất trong cột ở mỗi session (như bài báo, không đánh dấu APD).`));
children.push(h2("6.1. Cách chạy các baseline"));
children.push(bullet([b("Khung mã: "), "PyCIL (github.com/G-U-N/PyCIL, commit f3509b8, giấy phép MIT) — bài báo ghi các baseline được cài đặt trong PyCIL. Mã PyCIL giữ nguyên; script ", code("baselines/rs_pycil.py"), " nạp dữ liệu viễn thám, backbone và cách tính chỉ số vào PyCIL."]));
children.push(bullet([b("Dữ liệu: "), "đúng các phép chia và thứ tự lớp đã dùng cho phương pháp đề xuất (data_<bộ>/paper_seed0)."]));
children.push(bullet([b("Backbone: "), "bài báo chỉ nói các baseline \"dùng chung kiến trúc ViT\". Tái lập dùng hai nhánh ViT giống BiCM-Prompt (chiều 128, 4 lớp, 4 head, mỗi nhánh một modality), bỏ prompt; đặc trưng = ghép trung bình token của hai nhánh (256 chiều). Số tham số ≈ 1,63M, ít hơn 2,445M của bài báo — bài báo không mô tả chi tiết nên không khớp được con số này."]));
children.push(bullet([b("Bộ nhớ replay: "), "K = 30 / 33 / 35 chia đều cho các lớp đã học (2 / 3 / 5 mẫu/lớp ở session cuối), giống phương pháp đề xuất và đúng tiêu đề cột của Table IV."]));
children.push(bullet([b("Optimizer: "), "bài báo không nêu siêu tham số của baseline. Mặc định của PyCIL (SGD, lr 0,1) được thiết kế cho CNN trên CIFAR và không hội tụ với ViT ở đây (bảng thử nghiệm bên dưới), nên mọi baseline dùng Adam, lr 1e-3, batch 32 — đúng cấu hình huấn luyện bài báo nêu (\"Adam optimizer, initial learning rate 0.001, batch size 32\") — với số epoch và lịch giảm lr mặc định của PyCIL cho từng phương pháp (lr ở session tăng dần được giảm theo đúng tỉ lệ PyCIL dùng)."]));
children.push(bullet([b("Cách phân loại: "), "bảng chính dùng lớp fc (PyCIL gọi là \"CNN\"), là số PyCIL báo cáo mặc định; bảng NME trình bày riêng. Với FOSTER, PyCIL đánh giá mạng teacher (hai backbone) sau mỗi session; mạng student đã nén chỉ dùng từ session sau — giữ nguyên như PyCIL."]));
children.push(bullet([b("Lỗi đã sửa (PodNet): "), "bản đầu của script chỉ quy đổi lr truyền trực tiếp cho optimizer, bỏ sót lr khai báo trong từng nhóm tham số mà PodNet dùng ở các session tăng dần, nên PodNet bị huấn luyện với Adam lr 0,1 (gấp 100 lần) và sụp ngay ở Session1 (OA Houston 26 %). Đã sửa; các kết quả PodNet cũ bị loại (đổi tên *.INVALID_lr_bug.json) và chạy lại. Các phương pháp khác không dùng lr theo nhóm nên không bị ảnh hưởng."]));
children.push(bullet([b("BEEF — kết quả đáng ngờ: "), "trên Augsburg, ở giai đoạn mở rộng (expansion) độ chính xác train chỉ 12–29 % và lớp mới học rất kém (Low Plants 11,5 %, Industrial 25,9 % sau Session1), dẫn tới OA Session1 = 18 % so với 90,55 % của bài báo. Nghi vấn: loss năng lượng của BEEF (sinh mẫu bằng SGLD) hoặc lr không phù hợp với ViT/dữ liệu viễn thám. Sẽ kiểm tra riêng sau khi có đủ seed 0."]));
children.push(bullet([b("Ours: "), "kết quả seed 0 của phương pháp đề xuất (mục 7), cách phân loại theo mã gốc (NME ở Base, fc ở các session sau). Tham số học ≈ 0,85M ở các session tăng dần."]));
if (pilotSgd) {
  const paF = path.join(BASELINE_DIR, "icarl_adam_e1_houston_seed0.json"); // same batch 128 as the SGD pilot
  const pa = fs.existsSync(paF) ? JSON.parse(fs.readFileSync(paF)) : null;
  children.push(caption("Chọn optimizer cho baseline: iCaRL, Houston 2013, seed 0, batch 128 (OA / AA / Kappa, fc)", "không có trong bài báo"));
  children.push(table(["Session", "SGD lr 0,1 (mặc định PyCIL)", "Adam lr 1e-3 (đã chọn)", "Bài báo (iCaRL)"],
    pilotSgd.sessions.map((s, k) => [sname(k), `${f2(s.OA)} / ${f2(s.AA)} / ${f2(s.Kappa)}`,
      pa ? `${f2(pa.sessions[k].OA)} / ${f2(pa.sessions[k].AA)} / ${f2(pa.sessions[k].Kappa)}` : "…",
      `${f2(PAPER_T4.icarl.houston[k].OA)} / ${f2(PAPER_T4.icarl.houston[k].AA)} / ${f2(PAPER_T4.icarl.houston[k].K)}`]),
    [1.3, 2.6, 2.6, 2.6]));
}
{
  const b128 = (m, ds) => { const f = path.join(BASELINE_DIR, `${m}_adam_e1_${ds}_seed0.json`); return fs.existsSync(f) ? JSON.parse(fs.readFileSync(f)) : null; };
  const r128 = b128("lwf", "augsburg"), r32 = baselineRun("lwf", "augsburg");
  children.push(p([b("Batch size: "), "lần chạy đầu dùng batch 128 mặc định của PyCIL. Trên Augsburg, session base chỉ có 450 mẫu train (4 bước cập nhật/epoch) nên mô hình học chưa đủ: LwF đạt OA base 69,70 % (độ chính xác train 79 %), so với 91,92 % trong bài báo. Chuyển sang batch 32 như bài báo nêu, OA base tăng lên 86,33 %. Vì vậy toàn bộ baseline được chạy lại với batch 32; kết quả batch 128 (Houston: LwF, iCaRL, PodNet, WA, FOSTER; Augsburg: LwF) được giữ trong results/baselines/*_adam_e1_*.json để đối chiếu."]));
  if (r128) {
    children.push(caption("Ảnh hưởng của batch size: LwF, Augsburg, seed 0 (OA / AA / Kappa)", "không có trong bài báo"));
    children.push(table(["Session", "Batch 128", "Batch 32", "Bài báo (LwF)"], r128.sessions.map((s, k) => [sname(k),
      `${f2(s.OA)} / ${f2(s.AA)} / ${f2(s.Kappa)}`,
      r32 && r32.sessions[k] ? `${f2(r32.sessions[k].OA)} / ${f2(r32.sessions[k].AA)} / ${f2(r32.sessions[k].Kappa)}` : "…",
      `${f2(PAPER_T4.lwf.augsburg[k].OA)} / ${f2(PAPER_T4.lwf.augsburg[k].AA)} / ${f2(PAPER_T4.lwf.augsburg[k].K)}`]), [1.3, 2.6, 2.6, 2.6]));
  }
}
children.push(p([b("Số epoch: "), "bài báo không nêu số epoch cho phương pháp nào. Các baseline giữ mặc định của PyCIL (thiết kế cho CIFAR-100/ResNet), dài hơn nhiều so với phương pháp đề xuất:"]));
children.push(caption("Số epoch theo session (mặc định của mã; Session1 = Session2 = Session3)", "không có trong bài báo"));
children.push(table(["Phương pháp", "Base", "Mỗi session tăng dần", "Lịch lr"], [
  ["Ours (BiCM-Prompt)", "30", "35", "cosine; lr 1e-3 (base), 1e-4 (tăng dần)"],
  ["LwF", "200", "250", "giảm ở mốc cố định"],
  ["iCaRL", "200", "170", "giảm ở mốc cố định"],
  ["WA", "200", "170", "giảm ở mốc cố định"],
  ["PodNet", "160", "160 + 20 fine-tune cân bằng", "cosine"],
  ["FOSTER", "200", "170 mở rộng + 130 nén", "cosine"],
  ["BEEF", "200", "170 mở rộng + 60 gộp", "cosine"],
], [2.2, 1, 2.8, 3]));
const part1b = children.splice(0);

children.push(h2("6.2. Bảng kiểu Table IV"));
children.push(caption(`Kết quả tái lập trên ba bộ dữ liệu, seed ${T4_SEED}, phân loại bằng fc`, "cùng bố cục Table IV"));
children.push(t4Table((m, ds, s) => reproCell(m, ds, s), { params: paramsRepro }));
children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
children.push(caption("Số liệu của bài báo (chép lại từ Table IV)", "Table IV"));
children.push(t4Table(paperCell, { params: paramsPaper }));
children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
children.push(caption("Chênh lệch tái lập − bài báo (điểm %; APD: số dương = quên nhiều hơn bài báo)", "so với Table IV"));
children.push(t4Table(diffCell, { params: () => "", signed: true, bold: false }));
children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
children.push(caption(`Kết quả tái lập, seed ${T4_SEED}, phân loại bằng NME (LwF không lưu exemplar nên không có NME; APD theo NME của Ours chưa có vì các lần chạy chính được làm trước khi ghi chỉ số này — sẽ bổ sung khi chạy lại --eval nme)`, "cùng bố cục Table IV — giả định bài báo dùng NME"));
children.push(t4Table((m, ds, s) => reproCell(m, ds, s, "nme"), { params: paramsRepro }));
// ---- Table V replica (Last / Average) ----
const PAPER_T5 = { // [APD, OA, AA, Kappa]; Average rows have no APD
  dualprompt: { params: "0.815M", houston: [P(0.0330, 74.02, 77.75, 72.00), P(null, 83.77, 85.63, 81.18)], muufl: [P(0.0288, 70.66, 62.60, 62.92), P(null, 78.15, 73.02, 71.19)], augsburg: [P(0.2022, 59.13, 59.24, 49.12), P(null, 73.84, 67.83, 58.82)] },
  adamadapter: { params: "0.924M", houston: [P(0.0091, 73.06, 77.73, 70.93), P(null, 82.02, 84.65, 80.28)], muufl: [P(0.0196, 67.84, 66.45, 60.00), P(null, 76.95, 76.72, 70.04)], augsburg: [P(0.2482, 52.22, 57.46, 40.95), P(null, 68.97, 69.44, 53.80)] },
  ours: { params: "0.843M", houston: [P(0.0576, 77.39, 82.09, 75.56), P(null, 87.58, 89.71, 86.36)], muufl: [P(0.1003, 73.97, 67.02, 65.46), P(null, 79.33, 77.54, 72.14)], augsburg: [P(0.2431, 75.87, 58.73, 66.45), P(null, 84.89, 73.34, 73.57)] },
};
// values per (dataset, row 0 = Last / 1 = Average): arrays over seeds, or paper scalars
function oursT5(ds, row, kind) {
  const d = dsByKey[ds];
  const src = kind === "nme" ? (d.nmeRuns.length ? d.nmeRuns : d.runs) : d.runs;
  if (!src.length) return null;
  const n = d.paper.length - 1;
  const g = (sess, k) => (kind === "nme" ? sess.nme && sess.nme[k] : sess[k]);
  if (row === 0) {
    let apd = src.map((r) => g(r.sessions[n], "APD")).filter((x) => x !== undefined && x !== null);
    if (apd.length < src.length) apd = null; // incomplete (older runs did not record NME APD)
    return { APD: apd, OA: src.map((r) => g(r.sessions[n], "OA")), AA: src.map((r) => g(r.sessions[n], "AA")), K: src.map((r) => g(r.sessions[n], "Kappa")) };
  }
  const av = (k) => src.map((r) => mean(r.sessions.map((ss) => g(ss, k))));
  return { APD: null, OA: av("OA"), AA: av("AA"), K: av("Kappa") };
}
function t5Table(rowsSpec) {
  const rel = [1.9, 0.8, 0.9, ...Array(12).fill(0.85)];
  const tot = rel.reduce((a, c) => a + c, 0);
  const w = rel.map((x) => Math.floor((x / tot) * LAND_W));
  w[w.length - 1] += LAND_W - w.reduce((a, c) => a + c, 0);
  const mk = (text, width, o = {}) => new TableCell({
    borders, width: { size: width, type: WidthType.DXA }, columnSpan: o.colSpan, rowSpan: o.rowSpan, verticalAlign: "center",
    margins: { top: 30, bottom: 30, left: 40, right: 40 },
    shading: o.head ? { type: ShadingType.CLEAR, fill: "D9E2F3", color: "auto" } : o.fill ? { type: ShadingType.CLEAR, fill: o.fill, color: "auto" } : undefined,
    children: [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [new TextRun({ text: String(text), bold: o.head || o.bold, size: 16 })] })],
  });
  const fmtV = (v, k, signed) => {
    if (v === null || v === undefined) return "-";
    const dg = k === "APD" ? 4 : 2;
    if (Array.isArray(v)) return v.length ? (v.length === 1 ? v[0].toFixed(dg) : `${mean(v).toFixed(dg)} ±${std(v).toFixed(dg)}`) : "-";
    const t = v.toFixed(dg);
    return signed && v > 0 ? `+${t}` : t;
  };
  const rows = [
    new TableRow({ tableHeader: true, children: [mk("Phương pháp", w[0], { head: true, rowSpan: 2 }), mk("Tham số học", w[1], { head: true, rowSpan: 2 }), mk("Session", w[2], { head: true, rowSpan: 2 }),
      ...DS_KEYS.map((ds, j) => mk(T4_HEAD[ds], w[3 + 4 * j] * 4, { head: true, colSpan: 4 }))] }),
    new TableRow({ tableHeader: true, children: DS_KEYS.flatMap((ds, j) => ["APD", "OA", "AA", "Kappa"].map((h, q) => mk(h, w[3 + 4 * j + q], { head: true }))) }),
  ];
  for (const spec of rowsSpec) {
    [0, 1].forEach((row) => {
      const cells = [];
      if (row === 0) { cells.push(mk(spec.name, w[0], { rowSpan: 2, bold: spec.bold, fill: spec.fill })); cells.push(mk(spec.params, w[1], { rowSpan: 2, fill: spec.fill })); }
      cells.push(mk(row === 0 ? "Last" : "Average", w[2], { fill: spec.fill }));
      DS_KEYS.forEach((ds, j) => {
        const c = spec.get(ds, row);
        METRICS.forEach((k, q) => cells.push(mk(row === 1 && k === "APD" ? "-" : c ? fmtV(c[k], k, spec.signed) : "…", w[3 + 4 * j + q], { fill: spec.fill })));
      });
      rows.push(new TableRow({ cantSplit: true, children: cells }));
    });
  }
  return new Table({ width: { size: LAND_W, type: WidthType.DXA }, columnWidths: w, rows });
}
const nSeeds = Math.min(...DS_KEYS.map((ds) => dsByKey[ds].runs.length));
const paperT5 = (m) => (ds, row) => PAPER_T5[m][ds][row];
const diffT5 = (ds, row) => {
  const o = oursT5(ds, row, "main"), pp = PAPER_T5.ours[ds][row];
  if (!o) return null;
  return { APD: row === 0 && o.APD && o.APD.length ? mean(o.APD) - pp.APD : null, OA: mean(o.OA) - pp.OA, AA: mean(o.AA) - pp.AA, K: mean(o.K) - pp.K };
};
children.push(h2("6.3. Bảng kiểu Table V: phương pháp đề xuất (Last / Average)"));
children.push(p(`Cùng bố cục với Table V của bài báo: dòng Last = session cuối (APD của session cuối), dòng Average = trung bình OA/AA/Kappa qua tất cả các session. Kết quả tái lập của phương pháp đề xuất là trung bình ± độ lệch chuẩn qua ${nSeeds} seed. Dual-prompt và Adam-adapter chưa được tái lập (việc tiếp theo), nên hiện chỉ có số liệu của bài báo.`));
children.push(caption(`Phương pháp đề xuất: tái lập (${nSeeds} seed) so với bài báo, cùng hai phương pháp tinh chỉnh tham số của bài báo`, "cùng bố cục Table V"));
children.push(t5Table([
  { name: "Dual-prompt (tái lập)", params: "…", get: () => null },
  { name: "Adam-adapter (tái lập)", params: "…", get: () => null },
  { name: "Ours — tái lập, phân loại theo mã (NME ở Base, fc sau đó)", params: "≈0,85M", bold: true, fill: "F2F2F2", get: (ds, row) => oursT5(ds, row, "main") },
  { name: "Ours — tái lập, NME ở mọi session", params: "≈0,85M", get: (ds, row) => oursT5(ds, row, "nme") },
  { name: "Chênh lệch tái lập (theo mã) − bài báo", params: "", signed: true, get: diffT5 },
  { name: "Dual-prompt (bài báo)", params: PAPER_T5.dualprompt.params, get: paperT5("dualprompt") },
  { name: "Adam-adapter (bài báo)", params: PAPER_T5.adamadapter.params, get: paperT5("adamadapter") },
  { name: "Ours (bài báo)", params: PAPER_T5.ours.params, bold: true, get: paperT5("ours") },
]));
children.push(p("Ghi chú: APD theo NME của các lần chạy chính chưa được ghi (ô \"-\"), sẽ có khi chạy lại --eval nme. Với MUUFL, nếu dùng 2 hoặc 4 kênh LiDAR thay vì 1 kênh như bài báo ghi, Average OA của seed 0 tăng lên 79,08 / 79,79 (mục 7.4)."));
const part2 = children.splice(0);

// 7
children.push(h1("7. Kết quả phương pháp đề xuất"));
let sub = 0;
for (const d of DATASETS) {
  sub += 1;
  children.push(h2(`7.${sub}. ${d.title}`));
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

  // NME at every session: prefer dedicated --eval nme reruns, else the NME metrics recorded in the main runs.
  const nmeSrc = d.nmeRuns.length ? d.nmeRuns : runs;
  const nmeApd = (s) => nmeSrc.map((r) => r.sessions[s].nme.APD).filter((x) => x !== undefined && x !== null);
  const nmeRows = d.paper.map((x, s) => {
    const apd = nmeApd(s);
    return [sname(s), ms(runs.map((r) => r.sessions[s].fc.OA)),
      ms(nmeSrc.map((r) => r.sessions[s].nme.OA)), ms(nmeSrc.map((r) => r.sessions[s].nme.AA)), ms(nmeSrc.map((r) => r.sessions[s].nme.Kappa)),
      s === 0 ? "–" : apd.length ? ms(apd, 4) + (apd.length < nmeSrc.length ? ` (${apd.length} seed)` : "") : "chờ",
      `${f2(x.OA)} / ${f2(x.AA)} / ${f2(x.K)}`, f4(x.APD)];
  });
  const avgN = (k) => nmeSrc.map((r) => mean(r.sessions.map((ss) => ss.nme[k])));
  nmeRows.push(["Trung bình", ms(runs.map((r) => mean(r.sessions.map((ss) => ss.fc.OA)))), ms(avgN("OA")), ms(avgN("AA")), ms(avgN("Kappa")), "",
    `${f2(d.paperAvg.OA)} / ${f2(d.paperAvg.AA)} / ${f2(d.paperAvg.K)}`, ""]);
  children.push(caption(`${d.title}: đánh giá bằng NME ở mọi session (${nmeSrc.length} seed${d.nmeRuns.length ? ", chạy lại với --eval nme" : ""}) so với bài báo`,
    "đối chiếu Table IV, dòng Ours — giả định bài báo dùng NME"));
  children.push(table(["Session", "OA fc", "OA NME", "AA NME", "Kappa NME", "APD NME", "Bài báo OA / AA / Kappa", "APD bài báo"],
    nmeRows, [1.4, 1.4, 1.4, 1.4, 1.4, 1.5, 2.2, 1.1], { rowFill: (ri) => (ri === d.paper.length ? "F2F2F2" : undefined) }));

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
children.push(h2(`7.${sub}. MUUFL: thử các kênh LiDAR khác nhau`));
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
  children.push(h2(`7.${sub}. Mã gốc nguyên trạng trên Houston 2013`));
  children.push(caption("Mã gốc nguyên trạng (preset original: 7+3+3+2, 144 band, bộ chia DFC2013, seed 0)", "không so trực tiếp được với Table IV"));
  children.push(table(["Session", "Số lớp", "OA", "AA", "Kappa", "APD"],
    r.sessions.map((s) => [sname(s.session), s.seen_classes, f2(s.OA), f2(s.AA), f2(s.Kappa), s.session ? f4(s.APD) : "–"]),
    [1.5, 1, 1.3, 1.3, 1.3, 1.3]));
  children.push(p("Cấu hình này khác bài báo về số lớp mỗi session, thứ tự lớp, dữ liệu đầu vào và phép chia; chỉ dùng để thấy hành vi của mã công khai."));
}

// 7
children.push(h1("8. Nhận xét"));
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
  children.push(bullet([b(`${d.title}: `), `OA session base lệch ${s0 > 0 ? "+" : ""}${s0} điểm so với bài báo; OA session cuối lệch ${lastGap(d)} điểm với fc và ${lastNmeGap(d)} điểm nếu dùng NME (số âm = thấp hơn bài báo).`]));
}
children.push(bullet([b("Xu hướng chung: "), "Houston khớp bài báo ở Base và Session 1. MUUFL và Augsburg thấp hơn bài báo 5–6 điểm ngay từ Base (ít mẫu train hơn: 100 và 150 mẫu/lớp), nhưng OA ở Session 1 gần bài báo. Ở các session sau, cả ba bộ đều thấp hơn bài báo, chủ yếu do một vài lớp cũ bị quên mạnh (xem bảng độ chính xác từng lớp); độ dao động giữa các seed cũng lớn hơn."]));
children.push(bullet([b("Cách phân loại: "), "dùng NME ở mọi session làm tăng OA ở các session cuối trên cả ba bộ (Houston Session 3 +3,8; MUUFL Session 2 +13, Session 3 +8; Augsburg Session 2 +9 điểm), nhưng làm giảm mạnh ở Augsburg Session 1 (−11,5 điểm). Ngay cả với NME, OA session cuối vẫn thấp hơn bài báo 6–10 điểm. Vậy lựa chọn fc/NME chỉ giải thích được một phần chênh lệch; phần còn lại nhiều khả năng do các siêu tham số không được công bố (epoch, lr, trọng số loss ở session tăng dần)."]));
if (origRuns.length) children.push(bullet([b("Mã gốc nguyên trạng "), `cho OA cuối ${origRuns[0].sessions[origRuns[0].sessions.length - 1].OA.toFixed(2)} % trên Houston và quên mạnh ngay từ Session 1 (APD ${origRuns[0].sessions[1].APD.toFixed(3)}).`]));
children.push(bullet([b("Kết luận tạm thời: "), "mã công khai không phải là cấu hình đã tạo ra Table IV. Sau khi đưa về đúng mô tả của bài báo và sửa lỗi optimizer, kết quả khớp ở các session đầu nhưng vẫn lệch ở các session cuối. Những yếu tố bài báo không công bố (epoch, lr, trọng số loss ở session tăng dần; cách phân loại khi đánh giá) là nghi vấn chính."]));

{
  // baseline observations (seed 0, batch 32); numbers read from the result files
  const last = (m, ds) => { const r = baselineRun(m, ds); return r ? r.sessions[r.sessions.length - 1] : null; };
  const sOA = (x, k = "OA") => (x ? x[k].toFixed(2) : "…");
  const nOA = (x) => (x && x.nme && x.nme.OA != null ? x.nme.OA.toFixed(2) : "…");
  const pl = (m, ds) => { const a = PAPER_T4[m][ds]; return a[a.length - 1].OA.toFixed(2); };
  children.push(h2("Baseline (mục 6, seed 0, batch 32)"));
  children.push(bullet([b("LwF và WA "), `khớp bài báo tốt nhất: OA session cuối trên Houston LwF ${sOA(last("lwf", "houston"))} (bài báo ${pl("lwf", "houston")}), WA ${sOA(last("wa", "houston"))} (bài báo ${pl("wa", "houston")}); trên Augsburg WA ${sOA(last("wa", "augsburg"))} (bài báo ${pl("wa", "augsburg")}).`]));
  children.push(bullet([b("FOSTER "), `chỉ khớp bài báo khi đánh giá bằng NME: OA session cuối trên Houston fc ${sOA(last("foster", "houston"))} / NME ${nOA(last("foster", "houston"))} so với ${pl("foster", "houston")} của bài báo; trên Augsburg fc ${sOA(last("foster", "augsburg"))} / NME ${nOA(last("foster", "augsburg"))} so với ${pl("foster", "augsburg")}. Với FOSTER, mạng teacher sau mỗi session gần như không quên lớp cũ (APD ≈ 0) nhưng lớp fc phân loại kém lớp mới. Đây là dấu hiệu bài báo có thể đã dùng NME, ít nhất cho FOSTER.`]));
  children.push(bullet([b("PodNet "), `(sau khi sửa lỗi lr) cao hơn bài báo trên Houston: OA session cuối ${sOA(last("podnet", "houston"))} so với ${pl("podnet", "houston")}. Trên Augsburg OA/Kappa các session tăng dần cao hơn nhưng AA thấp hơn bài báo, và session base thấp hơn ~10 điểm (bộ phân loại cosine nhiều proxy của PodNet học kém hơn trên 450 mẫu).`]));
  children.push(bullet([b("iCaRL "), `khớp bài báo ở Base và Session1 trên Houston nhưng quên đột ngột ở Session2 (APD 0,64), OA session cuối ${sOA(last("icarl", "houston"))} (NME ${nOA(last("icarl", "houston"))}) so với ${pl("icarl", "houston")}. Trên Augsburg khá sát bài báo.`]));
  children.push(bullet([b("BEEF "), "trên Augsburg học lớp mới rất kém ở giai đoạn mở rộng (xem mục 6.1); trên MUUFL Session1 bình thường. Cần kiểm tra thêm trước khi kết luận."]));
  children.push(bullet([b("Lưu ý chung: "), "session base của LwF, iCaRL và WA giống hệt nhau vì cùng seed và cùng huấn luyện bằng cross-entropy; trong bài báo các giá trị này khác nhau (ví dụ Augsburg 84,32–93,65), nên nhiều khả năng mỗi baseline được chạy với seed hoặc cấu hình khác nhau. Trên Augsburg, OA bị chi phối bởi Low Plants và Residential (≈ 73 % tập test), nên AA và APD phản ánh khả năng chống quên tốt hơn OA."]));
}
// 8
children.push(h1("9. Việc tiếp theo"));
children.push(numbered("Hoàn tất seed 0 của sáu baseline trên cả ba bộ dữ liệu, sau đó chạy seed 1–2; kiểm tra riêng PodNet (kém bài báo rõ rệt trên Houston: chưng cất POD-spatial trên lưới token 11×11 và lr là nghi vấn)."));
children.push(numbered("Chạy lại đánh giá NME ở mọi session và seed 1–2 cho biến thể LiDAR 2 kênh của MUUFL (đã tạm dừng để ưu tiên baseline)."));
children.push(numbered("DualPrompt và Adam-adapter (Table V)."));
children.push(numbered("Ablation thành phần (Table VI) và các phân tích khác: top-k, độ dài prompt, prompt đơn modality, dung lượng bộ nhớ, thứ tự lớp."));
children.push(numbered("Thử nghiệm độ nhạy cho các giả định chưa rõ (lr/epoch session tăng dần, cách phân loại)."));
children.push(numbered("Cân nhắc mở issue trên GitHub của tác giả hỏi cấu hình chính xác của Table IV và báo lỗi optimizer."));

// 9
children.push(h1("10. Cách chạy lại"));
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
  "# baseline PyCIL (cần clone G-U-N/PyCIL vào baselines/PyCIL, commit f3509b8)",
  "cd ~/BiCMPT/baselines && python rs_pycil.py --method icarl --dataset houston --seed 0 --optimizer adam",
  "#   --method: lwf | icarl | podnet | wa | foster | beef; --dataset: houston | muufl | augsburg",
  "# cập nhật báo cáo: cd ~/BiCMPT/report && node make_report.js",
]));

// ---------- document ----------
const PORTRAIT = { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } };
const footer = () => new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Trang ", PageNumber.CURRENT], size: 18 })] })] });
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
  sections: [
    { properties: { page: PORTRAIT }, footers: { default: footer() }, children: [...part1, ...part1b] },
    { properties: { page: { ...PORTRAIT, size: { width: PAGE_W, height: 16838, orientation: PageOrientation.LANDSCAPE } } }, footers: { default: footer() }, children: part2 },
    { properties: { page: PORTRAIT }, footers: { default: footer() }, children },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log("wrote", OUT, DATASETS.map((d) => `${d.key}:${d.runs.length}`).join(" "), `lidar:${lv.length}`, `original:${origRuns.length}`);
});
