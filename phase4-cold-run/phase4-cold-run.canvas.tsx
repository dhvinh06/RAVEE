import { Callout, H1, H2, Row, Stack, Stat, Table, Text } from "cursor/canvas";

const report = {
  "status": "PASS",
  "caption": "Solve4X Scene 01 và 02 · Camera 2 · 15 FPS · 8 cụm mỗi scene · Seed 0 · 28/11/2026.",
  "perfRows": [
    {
      "name": "M0 · RAVEE_org",
      "scenes": [
        {
          "total": "2.334,80",
          "stages": "Tiền xử lý 29,65 · IDT 2.031,93 · FV 272,02 · TW-FINCH 1,20"
        },
        {
          "total": "1.045,51",
          "stages": "Tiền xử lý 13,00 · IDT 895,05 · FV 137,09 · TW-FINCH 0,37"
        }
      ],
      "total": "3.380,31",
      "time": "56 phút 20 giây",
      "speed": "1,00×",
      "resources": "— / — / —"
    },
    {
      "name": "M1 · RAVEE_optz RAW_ONLY",
      "scenes": [
        {
          "total": "1.820,33",
          "stages": "Tiền xử lý 33,38 · IDT 1.605,55 · FV 180,27 · TW-FINCH 1,13"
        },
        {
          "total": "970,41",
          "stages": "Tiền xử lý 11,88 · IDT 862,93 · FV 95,28 · TW-FINCH 0,33"
        }
      ],
      "total": "2.790,74",
      "time": "46 phút 31 giây",
      "speed": "1,21×",
      "resources": "— / 2,80 / —"
    },
    {
      "name": "E1 · CPU W=6 + FV opt",
      "scenes": [
        {
          "total": "750,71",
          "stages": "Tiền xử lý 20,05 · IDT 673,07 · FV 56,48 · TW-FINCH 1,11"
        },
        {
          "total": "434,22",
          "stages": "Tiền xử lý 12,58 · IDT 375,79 · FV 45,42 · TW-FINCH 0,44"
        }
      ],
      "total": "1.184,94",
      "time": "19 phút 45 giây",
      "speed": "2,85×",
      "resources": "4,68 / 1,69 / —"
    },
    {
      "name": "E2 · CUDA W=6 + FV opt",
      "scenes": [
        {
          "total": "547,23",
          "stages": "Tiền xử lý 21,44 · IDT 465,28 · FV 59,20 · TW-FINCH 1,31"
        },
        {
          "total": "310,58",
          "stages": "Tiền xử lý 12,32 · IDT 251,12 · FV 46,80 · TW-FINCH 0,35"
        }
      ],
      "total": "857,81",
      "time": "14 phút 18 giây",
      "speed": "3,94×",
      "resources": "5,65 / 1,63 / 2,07"
    }
  ],
  "qRows": [
    [
      "Raw ARI",
      "0,2109",
      "0,2084 ± 0,0204",
      "0,1983",
      "0,2039 ± 0,0268"
    ],
    [
      "Oracle ARI",
      "0,5790",
      "0,4912 ± 0,0603",
      "0,4595",
      "0,4785 ± 0,0644"
    ],
    [
      "Oracle Accuracy",
      "75,24%",
      "70,67% ± 4,02 đpt",
      "66,29%",
      "68,99% ± 4,53 đpt"
    ],
    [
      "Số giây đúng / 614",
      "462 / 614",
      "433,90 ± 24,68",
      "407 / 614",
      "423,60 ± 27,81"
    ]
  ],
  "details": [
    {
      "name": "E2 · Scene 01",
      "ram": "5,65 / 1,63 / 2,07",
      "intermediate": "1.109,39 MB",
      "trajectory": "Khớp",
      "fv": "Khớp",
      "fvHash": "a1302d5e03288f9f5f605fb643740198bc9dab143f5b251d3245c8eaaf4d851b",
      "oldFvHash": "a1302d5e03288f9f5f605fb643740198bc9dab143f5b251d3245c8eaaf4d851b",
      "gpu": "51 °C · 975 mẫu",
      "power": "10:01:42 → 10:11:01 · AC 1 → 1"
    },
    {
      "name": "E2 · Scene 02",
      "ram": "3,50 / 0,74 / 2,07",
      "intermediate": "480,70 MB",
      "trajectory": "Khớp",
      "fv": "Khớp",
      "fvHash": "0ccc65df5a04c06882ee3309ae53610eeb4c5297bc293ff3427f4800c5a0014c",
      "oldFvHash": "0ccc65df5a04c06882ee3309ae53610eeb4c5297bc293ff3427f4800c5a0014c",
      "gpu": "51 °C · 547 mẫu",
      "power": "10:11:05 → 10:16:19 · AC 1 → 1"
    },
    {
      "name": "E1 · Scene 01",
      "ram": "4,68 / 1,69 / —",
      "intermediate": "1.145,79 MB",
      "trajectory": "Khớp",
      "fv": "Khớp",
      "fvHash": "6d26b9742bb226d7cbf45225e904395fa162c7aae451fa9d1e6e3e1c1db29538",
      "oldFvHash": "6d26b9742bb226d7cbf45225e904395fa162c7aae451fa9d1e6e3e1c1db29538",
      "gpu": null,
      "power": "10:19:24 → 10:32:01 · AC 1 → 1"
    },
    {
      "name": "E1 · Scene 02",
      "ram": "2,48 / 0,77 / —",
      "intermediate": "497,30 MB",
      "trajectory": "Khớp",
      "fv": "Khớp",
      "fvHash": "682aecc7de2a12b6df4b2be8b90be89a46924551e3ace2b01fb34e63f11f36bd",
      "oldFvHash": "682aecc7de2a12b6df4b2be8b90be89a46924551e3ace2b01fb34e63f11f36bd",
      "gpu": null,
      "power": "10:32:06 → 10:39:24 · AC 1 → 1"
    }
  ],
  "checks": [
    [
      "Đủ bốn công đoạn ở cả bốn lượt",
      "Đạt"
    ],
    [
      "AC=1 và High performance ở đầu và cuối từng scene",
      "Đạt"
    ],
    [
      "Nghỉ 180,02 giây giữa E2 và E1",
      "Đạt"
    ],
    [
      "Docker RAM < 6,5 GiB, FV RAM < 2,5 GiB, CUDA VRAM < 3,5 GiB",
      "Đạt"
    ],
    [
      "E2 đạt 3,94× so với ngưỡng 3,0×",
      "Đạt"
    ],
    [
      "Tái lập chất lượng seed 0 của Giai đoạn 3",
      "Đạt"
    ]
  ],
  "hashNote": "FV SHA-256 khớp 4/4. Số cụm từng cấp TW-FINCH khớp 4/4. E1 khớp 9/9 điểm seed 0 đã lưu. Mốc CUDA Bước 2 trích được gồm thống kê 10 seed, SHA-256 và số cụm từng cấp.",
  "sources": "Thời gian M0/M1 lấy từ benchmark Solve4X ngày 26/11/2026. Mốc E1 10 seed dùng FV3 CPU W=6. Mốc E2 10 seed dùng FV Bước 2 CUDA W=6.",
  "scope": "Thời gian là tổng wall-clock của bốn công đoạn. RAM Docker lấy từ đỉnh cgroup, RAM FV lấy mẫu working set Python mỗi 100 ms và VRAM toàn GPU lấy mẫu bằng nvidia-smi mỗi khoảng 0,5 giây. Các cột tài nguyên lấy đỉnh cao nhất của hai scene.",
  "qualityNote": "SD là độ lệch chuẩn mẫu. Số giây đúng của mốc 10 seed được tính từ Oracle Accuracy × 614. Đối chiếu tái lập dùng mốc seed 0.",
  "speedNote": "M1 → E1 nhanh hơn 2,36×. E1 → E2 nhanh thêm 1,38×.",
  "freed": "3,23 GB"
};

export default function Phase4ColdRun() {
  return (
    <Stack gap={22} style={{ padding: 24, maxWidth: 1160, margin: "0 auto" }}>
      <Stack gap={8}>
        <H1>RAVEE · Giai đoạn 4 · Trụ 1</H1>
        <Text tone="secondary">{report.caption}</Text>
      </Stack>
      <Row gap={28} wrap>
        <Stat value={report.status} label="Kết quả nghiệm thu" />
        <Stat value={report.perfRows[3].speed} label="E2 so với RAVEE_org" />
        <Stat value={report.perfRows[3].time} label="Tổng thời gian E2" />
      </Row>
      <Stack gap={10}>
        <H2>Bảng 1 · Hiệu năng End-to-End Cold Run</H2>
        <Table headers={["Cấu hình", "Scene 01 (s)", "Scene 02 (s)", "Tổng (s)", "So với M0", "RAM Docker / FV / VRAM (GiB)"]}
          rows={report.perfRows.map(row => [row.name, ...row.scenes.map(scene =>
            <Stack gap={4}><Text weight="semibold">{scene.total}</Text><Text size="small" tone="secondary">{scene.stages}</Text></Stack>
          ), row.total, row.speed, row.resources])} />
        <Text size="small" tone="secondary">{report.scope} {report.speedNote}</Text>
      </Stack>
      <Stack gap={10}>
        <H2>Bảng 2 · Chất lượng phân đoạn gộp trên 614 giây</H2>
        <Table headers={["Chỉ số", "E1 seed 0", "E1 10 seed", "E2 seed 0", "E2 10 seed"]} rows={report.qRows} />
        <Text size="small" tone="secondary">{report.qualityNote}</Text>
        <Callout tone="neutral">{report.hashNote}</Callout>
      </Stack>
      <Stack gap={10}>
        <H2>Điều kiện nghiệm thu</H2>
        <ul style={{ margin: 0, paddingInlineStart: 20 }}>
          {report.checks.map(([label, outcome]) => <li key={label}><Text as="span"><strong>{outcome}.</strong> {label}</Text></li>)}
        </ul>
        <Text size="small" tone="secondary">Đã dọn {report.freed} file trung gian. {report.sources}</Text>
      </Stack>
    </Stack>
  );
}
