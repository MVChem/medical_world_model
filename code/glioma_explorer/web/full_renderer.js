/* Complete local volume renderer. Assets contain all voxels, never sampled slices. */
window.AtlasOffline = (() => {
  const patients = new Map(),
    packed = new Map(),
    volumes = new Map(),
    pending = new Map();
  let tables;
  window.__ATLAS_PATIENT__ = (id, value) => patients.set(id, value);
  window.__ATLAS_VOLUME__ = (id, value) => packed.set(id, value);
  window.__ATLAS_TABLES__ = (_id, value) => {
    tables = value;
  };
  const base = () =>
    window.__ATLAS_SNAPSHOT__?.asset_base || "/preview-assets/";
  async function script(path) {
    return new Promise((resolve, reject) => {
      const element = document.createElement("script");
      element.src = base() + path;
      element.onload = () => {
        element.remove();
        resolve();
      };
      element.onerror = () => {
        element.remove();
        reject(
          new Error(
            "无法读取本地数据文件：" +
              path +
              "。请保留 HTML 旁的 full_data 目录。",
          ),
        );
      };
      document.head.appendChild(element);
    });
  }
  async function record(dataset, pid) {
    if (!patients.has(pid)) {
      const path =
        window.__ATLAS_SNAPSHOT__?.patient_index?.[pid] ||
        `${dataset}/${pid}/patient.js`;
      await script(path);
    }
    if (!patients.has(pid)) throw new Error("患者数据文件未正确加载：" + pid);
    return patients.get(pid);
  }
  async function patient(dataset, pid, query) {
    const p = await record(dataset, pid),
      all = p.image.timepoints;
    const selected = query.has("first")
      ? [
          Number(query.get("first")),
          ...(query.has("second") ? [Number(query.get("second"))] : []),
        ]
      : all.slice(0, 2);
    if (
      !selected.length ||
      selected.some((t) => !all.includes(t)) ||
      new Set(selected).size !== selected.length
    )
      throw new Error("该时间点没有本地影像");
    const image = { ...p.image, timepoints: selected };
    for (const field of ["volumes_ml", "mask_available", "label_values"])
      image[field] = selected.map((t) => p.image[field][all.indexOf(t)]);
    return { ...p, image };
  }
  async function volume(description) {
    const key = description.key;
    if (volumes.has(key)) {
      const value = volumes.get(key);
      volumes.delete(key);
      volumes.set(key, value);
      return value;
    }
    if (pending.has(key)) return pending.get(key);
    const promise = (async () => {
      await script(description.asset);
      const p = packed.get(key);
      packed.delete(key);
      if (!p) throw new Error("影像文件内容缺失：" + description.name);
      const bytes = Uint8Array.from(atob(p.data), (c) => c.charCodeAt(0));
      if (typeof DecompressionStream === "undefined")
        throw new Error(
          "此浏览器不支持影像解压，请使用新版 Chrome、Edge 或 Safari。",
        );
      const stream = new Blob([bytes])
        .stream()
        .pipeThrough(new DecompressionStream("gzip"));
      const array = new Uint8Array(await new Response(stream).arrayBuffer());
      if (array.length !== p.shape.reduce((a, b) => a * b, 1))
        throw new Error("影像体素数不匹配：" + p.name);
      const value = { array, shape: p.shape, spacing: p.spacing, kind: p.kind };
      volumes.set(key, value);
      // Retain only a few whole volumes while moving through the 501 patients.
      while (volumes.size > 8) volumes.delete(volumes.keys().next().value);
      return value;
    })();
    pending.set(key, promise);
    try {
      return await promise;
    } finally {
      pending.delete(key);
    }
  }
  const palette = [
    [0, 0, 0],
    [184, 161, 227],
    [101, 198, 187],
    [239, 184, 97],
    [226, 145, 163],
  ];
  function renderVolume(v, mask, plane, index, opacity, labelView = false) {
    const axis = { axial: 2, coronal: 1, sagittal: 0 }[plane];
    if (axis == null || index < 0 || index >= v.shape[axis])
      throw new Error("切片超出影像范围");
    const axes = [0, 1, 2].filter((a) => a !== axis),
      width = v.shape[axes[0]],
      height = v.shape[axes[1]];
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d"),
      pixels = context.createImageData(width, height);
    for (let row = 0; row < height; row++)
      for (let col = 0; col < width; col++) {
        const pos = [0, 0, 0];
        pos[axis] = index;
        pos[axes[0]] = col;
        pos[axes[1]] = height - 1 - row;
        const at = (pos[0] * v.shape[1] + pos[1]) * v.shape[2] + pos[2],
          out = (row * width + col) * 4;
        const gray = v.array[at],
          label = mask?.array[at] || 0;
        const labelColor = palette[gray] || [200, 200, 200];
        const color = palette[label] || [200, 200, 200];
        for (let channel = 0; channel < 3; channel++)
          pixels.data[out + channel] = labelView
            ? gray
              ? labelColor[channel]
              : 0
            : label
              ? Math.floor(gray * (1 - opacity) + color[channel] * opacity)
              : gray;
        pixels.data[out + 3] = 255;
      }
    context.putImageData(pixels, 0, 0);
    const physicalWidth = width * v.spacing[axes[0]],
      physicalHeight = height * v.spacing[axes[1]];
    const scale = 640 / Math.max(physicalWidth, physicalHeight),
      output = document.createElement("canvas");
    output.width = Math.max(1, Math.round(physicalWidth * scale));
    output.height = Math.max(1, Math.round(physicalHeight * scale));
    const outputContext = output.getContext("2d");
    outputContext.imageSmoothingEnabled = false;
    outputContext.drawImage(canvas, 0, 0, output.width, output.height);
    return output.toDataURL("image/png");
  }
  async function render(
    p,
    sequence,
    plane,
    index,
    timepoint,
    overlay,
    opacity,
  ) {
    const find = (key) => p.files.find((f) => f.key === key);
    const image = find(p.volume_keys[`${sequence}:${timepoint}`]);
    if (!image) throw new Error("缺少该 MRI 序列");
    const mask = overlay ? find(p.volume_keys[`seg:${timepoint}`]) : null;
    const [v, m] = await Promise.all([
      volume(image),
      mask ? volume(mask) : null,
    ]);
    return renderVolume(v, m, plane, index, opacity);
  }
  async function renderFile(file, plane, index) {
    return renderVolume(
      await volume(file),
      null,
      plane,
      index,
      0,
      file.kind === "label",
    );
  }
  async function rawTables(dataset) {
    if (!tables) await script("tables.js");
    return tables[dataset];
  }
  return {
    patient,
    record,
    render,
    renderFile,
    rawTables,
    renderVolume,
    cacheSize: () => volumes.size,
  };
})();
