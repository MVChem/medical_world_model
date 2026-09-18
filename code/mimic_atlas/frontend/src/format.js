export const fmt = (n) => (n == null ? "—" : Number(n).toLocaleString("en-US"));
export const when = (s) =>
  s ? String(s).replace("T", " ").slice(0, 19) : "无事件时刻";
export const time = (s) =>
  s ? Date.parse(String(s).replace(" ", "T").replace(/Z$/, "") + "Z") : NaN;
export const iso = (n) => new Date(n).toISOString().replace(/Z$/, "");
export const hours = (n) =>
  Number(n) < 72
    ? Number(n).toFixed(1) + " h"
    : (Number(n) / 24).toFixed(1) + " d";
export const esc = (v) =>
  String(v ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
export const labelNames = {
  present: "阳性",
  absent: "阴性",
  uncertain: "不确定",
  not_mentioned: "未提及",
};
