import { fmt } from "../format";

export default function DatasetIntro({ catalog, compact = false, offline = false }) {
  const c = catalog.cohort?.counts || {};
  return (
    <details className="dataset-intro" id="dataset-intro" open={!compact}>
      <summary>
        <span>数据集简介与统计口径</span>
        <small>MIMIC-CXR 影像 · MIMIC-IV 临床 · 患者如何关联</small>
      </summary>
      <div className="dataset-intro-body">
        <div className="dataset-sources">
          <article>
            <h3>MIMIC-CXR · 胸部 X 光与放射学报告</h3>
            <p>
              包含患者多次胸片检查及对应的放射学报告。本项目使用
              MIMIC-CXR-JPG 2.0.0 的 JPG 影像、检查元数据和结构化标签，
              并从本地 MIMIC-CXR 读取报告。标签由报告自动提取。
            </p>
            <p className="dataset-source-count" id="intro-cxr-count">
              {fmt(c.cxr_patients)} 位患者 · {fmt(c.studies)} 次检查 · {fmt(c.images)} 张影像
            </p>
            <a href="https://physionet.org/content/mimic-cxr-jpg/2.0.0/" target="_blank" rel="noreferrer">
              MIMIC-CXR-JPG 官方说明 ↗
            </a>
          </article>
          <article>
            <h3>MIMIC-IV · 电子病历与临床事件</h3>
            <p>
              本项目使用 MIMIC-IV 3.1，包括患者基本信息、住院与 ICU 记录、
              检验、生命体征、微生物、处方、给药、医嘱、诊断和操作等。
              每位患者的数据覆盖不同，并非人人都有住院、ICU 或胸片记录。
            </p>
            <p className="dataset-source-count" id="intro-iv-count">
              {fmt(c.iv_patients)} 位患者 · {fmt(c.admissions)} 次住院 · {fmt(c.icu_stays)} 次 ICU 入住
            </p>
            <a href="https://physionet.org/content/mimiciv/3.1/" target="_blank" rel="noreferrer">
              MIMIC-IV 官方说明 ↗
            </a>
          </article>
        </div>
        <div className="dataset-definition">
          <h3>“全量患者” = CXR 与 IV 的患者并集</h3>
          <p id="intro-relationship">
            <strong>两者的患者集合部分重叠、互不包含。</strong>
            在当前本地 MIMIC-CXR-JPG 2.0.0 与 MIMIC-IV 3.1 中，
            CXR 有 IV 中没有的患者，IV 也有 CXR 中没有的患者；
            同时出现在两库中的患者构成交集。
          </p>
          <p>
            将本地两份数据按患者标识 <code>subject_id</code> 合并去重。
            同时出现在两份数据中的患者只计算一次，合计 <strong>{fmt(c.patients)}</strong> 人。
          </p>
          <dl className="dataset-coverage" aria-label="去重后的患者组成">
            {[
              ["同时有 IV 和 CXR（交集）", c.matched_patients],
              ["只有 IV", c.iv_only],
              ["只有 CXR", c.cxr_only],
              ["合并去重（并集）", c.patients],
            ].map(([label, value]) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd>{fmt(value)}</dd>
              </div>
            ))}
          </dl>
          <p>
            JPG / PNG 是影像文件格式。两库按 <code>subject_id</code> 关联患者，
            与图片格式无关。
          </p>
          <p>
            一个患者可以有多次住院、多次检查；一次检查也可以有多张不同投照的影像。
            <code>subject_id</code> 标识患者，<code>hadm_id</code> 标识住院，
            <code>study_id</code> 标识影像检查，<code>dicom_id</code> 标识单张影像。
            报告按检查关联。
          </p>
          <p>
            <strong>患者关联与住院匹配分开判断：</strong>
            同时有 IV 和 CXR 表示患者标识一致；具体检查能否关联某次住院，还需核对检查时间与住院区间。
            未匹配或有歧义的记录会保留其状态。
          </p>
          <p>
            <strong>可浏览范围不等于实际训练集。</strong>
            “影像配对”展示当前规则下的候选；实际训练范围以任务筛选和导出的训练清单为准。
            列表中的 train / validate / test 是官方 CXR 数据划分。
          </p>
        </div>
        <p className="dataset-stat-note">
          {offline
            ? "以上为导出时的本地全库统计；本离线 HTML 仅包含当前导出的片段与已嵌入记录。"
            : "以上人数与记录数来自当前本地全库目录，不随列表筛选变化。"}
        </p>
      </div>
    </details>
  );
}
