import { useRef, useState } from "react";
import { when } from "../format";

export default function ImageViewer({
  patient,
  studies,
  packet,
  quality,
  offline,
}) {
  const [single, setSingle] = useState(0),
    [zoom, setZoom] = useState(1),
    [brightness, setBrightness] = useState(1),
    [contrast, setContrast] = useState(1),
    [inverted, setInverted] = useState(false);
  const [offset, setOffset] = useState({ x: 0, y: 0 }),
    [large, setLarge] = useState(null);
  const dialog = useRef(null),
    drag = useRef(null);
  const selected = studies.map(
    (s, i) =>
      s.images.find(
        (im) =>
          im.dicom_id ===
          packet?.[i ? "future_state" : "current_state"].image.dicom_id,
      ) ||
      s.images[single] ||
      s.images[0],
  );
  const url = (im, full = false) =>
    offline || im.url?.startsWith("data:")
      ? im.url
      : full
        ? im.full_url || `/api/images/${im.dicom_id}?size=1800&quality=88`
        : `/api/images/${im.dicom_id}?size=${quality}&format=webp&quality=${quality === "512" ? 60 : 78}`;
  function reset() {
    setZoom(1);
    setBrightness(1);
    setContrast(1);
    setInverted(false);
    setOffset({ x: 0, y: 0 });
  }
  return (
    <>
      <div
        id="image-grid"
        className={"image-grid" + (studies.length === 1 ? " single" : "")}
      >
        {studies.map((study, i) => {
          const im = selected[i];
          return (
            <div
              key={study.study_id}
              className={"image-card " + (i ? "future" : "current")}
            >
              <div className="image-caption">
                <span className="role">
                  {i
                    ? "T₁ / OBSERVED FOLLOW-UP"
                    : studies.length > 1
                      ? "T₀ / CURRENT STATE"
                      : "SELECTED STUDY"}
                </span>
                <time>{when(im?.timestamp || study.timestamp)}</time>
              </div>
              <div
                className="image-stage"
                data-side={i}
                style={{ touchAction: zoom > 1 ? "none" : "pan-y" }}
              >
                <div className="image-tags">
                  {study.study_id}
                  <br />
                  {im?.view || "无影像"}
                </div>
                {im?.available ? (
                  <>
                    <img
                      key={url(im)}
                      className="scan"
                      data-side={i}
                      src={url(im)}
                      decoding="async"
                      alt={`患者 ${patient.subject_id}，检查 ${study.study_id}，${im.view} 胸片`}
                      draggable="false"
                      style={{
                        transform: `translate(${offset.x}px,${offset.y}px) scale(${zoom})`,
                        filter: `brightness(${brightness}) contrast(${contrast}) invert(${inverted ? 1 : 0})`,
                      }}
                      onLoad={(e) => e.currentTarget.classList.add("loaded")}
                      onError={(e) => {
                        e.currentTarget.hidden = true;
                        e.currentTarget.nextElementSibling.textContent =
                          "影像读取失败";
                      }}
                      onPointerDown={(e) => {
                        if (zoom <= 1) return;
                        e.preventDefault();
                        e.currentTarget.setPointerCapture(e.pointerId);
                        drag.current = {
                          x: e.clientX - offset.x,
                          y: e.clientY - offset.y,
                        };
                      }}
                      onPointerMove={(e) => {
                        if (drag.current)
                          setOffset({
                            x: e.clientX - drag.current.x,
                            y: e.clientY - drag.current.y,
                          });
                      }}
                      onPointerUp={() => {
                        drag.current = null;
                      }}
                      onPointerCancel={() => {
                        drag.current = null;
                      }}
                    />
                    <span className="image-loading">读取影像…</span>
                    <button
                      className="expand-image"
                      data-expand={i}
                      aria-label={"放大" + (i ? "随访" : "当前") + "胸片"}
                      onClick={() => {
                        setLarge(im);
                        dialog.current.showModal();
                      }}
                    >
                      ⛶
                    </button>
                  </>
                ) : (
                  <div className="image-unavailable">
                    此影像文件在本地不可用
                  </div>
                )}
              </div>
              <div className="image-bottom">
                <span>
                  {im?.available
                    ? `${im.columns ?? "—"} × ${im.rows ?? "—"} px`
                    : "IMAGE UNAVAILABLE"}
                </span>
                {studies.length === 1 && study.images.length > 1 ? (
                  <select
                    id="single-image"
                    aria-label="选择投照影像"
                    value={single}
                    onChange={(e) => {
                      setSingle(Number(e.target.value));
                      reset();
                    }}
                  >
                    {study.images.map((m, j) => (
                      <option key={m.dicom_id} value={j}>
                        {m.view} · {m.dicom_id.slice(0, 8)}
                        {m.available ? "" : " · 缺失"}
                      </option>
                    ))}
                  </select>
                ) : (
                  <span>
                    {i ? "随访观测 / 评估用" : "当前影像"} · MIMIC-CXR
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
      <div className="image-tools">
        <span className="tool-label">同步显示</span>
        <label>
          缩放{" "}
          <input
            id="zoom"
            type="range"
            min="1"
            max="3"
            step="0.1"
            value={zoom}
            onChange={(e) => {
              const n = Number(e.target.value);
              setZoom(n);
              if (n === 1) setOffset({ x: 0, y: 0 });
            }}
          />
          <output id="zoom-value">{zoom.toFixed(1)}×</output>
        </label>
        <label>
          亮度{" "}
          <input
            id="brightness"
            type="range"
            min="0.5"
            max="1.8"
            step="0.05"
            value={brightness}
            onChange={(e) => setBrightness(e.target.value)}
          />
        </label>
        <label>
          对比度{" "}
          <input
            id="contrast"
            type="range"
            min="0.5"
            max="2"
            step="0.05"
            value={contrast}
            onChange={(e) => setContrast(e.target.value)}
          />
        </label>
        <button
          id="invert"
          aria-pressed={inverted}
          onClick={() => setInverted(!inverted)}
        >
          反相
        </button>
        <button id="reset-image" onClick={reset}>
          重置
        </button>
        <span className="viewer-help">放大后可拖动 · 点击 ⛶ 查看大图</span>
      </div>
      <dialog ref={dialog} id="image-dialog">
        <div className="dialog-toolbar">
          <span id="dialog-caption">
            {large?.dicom_id} · {large?.view}
          </span>
          <button id="close-dialog" onClick={() => dialog.current.close()}>
            关闭 ×
          </button>
        </div>
        {large && (
          <img id="dialog-image" src={url(large, true)} alt="放大的胸部影像" />
        )}
      </dialog>
    </>
  );
}
