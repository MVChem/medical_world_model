import { useState } from "react";
import { useResource } from "../api";
import { fmt } from "../format";

function Bundle({ bundle, onOpen }) {
  const [open, setOpen] = useState(false),
    [page, setPage] = useState(1);
  const { data, error } = useResource(
    open ? `/api/exports/${bundle.id}/cases?page=${page}` : null,
  );
  return (
    <section className="panel export-bundle">
      <div className="panel-heading">
        <div>
          <div className="eyebrow">
            {bundle.linked ? "CXR + IV" : "CXR EXPORT"}
          </div>
          <h3>{bundle.name}</h3>
          <p className="muted">
            {bundle.collection} · {fmt(bundle.cases)} 个片段
          </p>
        </div>
        <button className="button" onClick={() => setOpen(!open)}>
          {open ? "收起片段 ↑" : "浏览片段 ↓"}
        </button>
      </div>
      <div className="export-downloads">
        {bundle.files?.map((file) => (
          <a
            key={file.name}
            className="button"
            href={`/api/exports/${bundle.id}/files/${file.name}`}
            download
          >
            {file.name} ↓{" "}
            <span className="muted">{(file.bytes / 1024).toFixed(1)} KB</span>
          </a>
        ))}
      </div>
      {(bundle.error || error) && (
        <div className="notice">{bundle.error || error}</div>
      )}
      {open && (
        <>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>患者</th>
                  <th>当前 → 随访</th>
                  <th>Split</th>
                  <th>分类</th>
                  <th>导出时住院匹配</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {data?.rows.map((row) => (
                  <tr key={row.transition_id}>
                    <td>{row.subject_id}</td>
                    <td>
                      {row.source_study} → {row.target_study}
                    </td>
                    <td>{row.split}</td>
                    <td>{row.category || "—"}</td>
                    <td>{row.linkage_status || "未导出 IV"}</td>
                    <td>
                      <button
                        className="button"
                        data-export-transition={row.transition_id}
                        onClick={() =>
                          onOpen(row.subject_id, row.transition_id)
                        }
                      >
                        在患者页面查看 ↗
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!data && <p className="empty-state">读取片段目录…</p>}
          </div>
          <div className="observation-pagination">
            <button disabled={page <= 1} onClick={() => setPage(page - 1)}>
              ←
            </button>
            <span>
              {page} / {Math.max(1, Math.ceil((data?.total || 0) / 25))}
            </span>
            <button
              disabled={!data || page * 25 >= data.total}
              onClick={() => setPage(page + 1)}
            >
              →
            </button>
          </div>
        </>
      )}
    </section>
  );
}
export default function Exports({ onOpen }) {
  const { data, error } = useResource("/api/exports");
  return (
    <div id="exports-content">
      <section className="page-heading">
        <div>
          <div className="eyebrow">EXPORT LIBRARY</div>
          <h1>导出记录</h1>
          <p>
            已有病例与训练清单统一存放在这里，点击片段可进入当前患者工作区。
          </p>
        </div>
        <span className="subtle-badge">
          {fmt(data?.bundles.length)} 个导出批次
        </span>
      </section>
      <p className="table-note">
        文件保留导出时的内容；患者工作区读取当前原始数据。旧 HTML
        作为归档下载，影像、报告和临床记录通过新的患者页面浏览。
      </p>
      {error && <div className="notice">{error}</div>}
      {data?.bundles.map((bundle) => (
        <Bundle key={bundle.id} bundle={bundle} onOpen={onOpen} />
      ))}
      {data && !data.bundles.length && (
        <div className="empty-state">暂无导出记录。</div>
      )}
    </div>
  );
}
