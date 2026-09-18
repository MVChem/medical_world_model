import { useEffect, useState } from "react";
export async function api(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : `读取失败 (${response.status})`,
    );
  }
  return response.json();
}
export function useResource(url, refresh = 0) {
  const [result, setResult] = useState({ url: null, data: null, error: "" });
  useEffect(() => {
    if (!url) {
      // Release inactive data and do not replay an expired response on re-enable.
      setResult({ url: null, data: null, error: "" });
      return;
    }
    const controller = new AbortController();
    let timer;
    async function read() {
      try {
        const data = await api(url, { signal: controller.signal });
        if (!controller.signal.aborted) setResult({ url, data, error: "" });
      } catch (error) {
        if (!controller.signal.aborted)
          setResult({ url, data: null, error: error.message });
      } finally {
        if (refresh && !controller.signal.aborted)
          timer = setTimeout(read, refresh);
      }
    }
    read();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [url, refresh]);
  return result.url === url ? result : { data: null, error: "" };
}
export function useDebounced(value, delay = 250) {
  const [current, setCurrent] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setCurrent(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
  return current;
}
export async function download(url, name) {
  const response = await fetch(url);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw Error(body.detail || "导出失败");
  }
  saveBlob(await response.blob(), name);
}
export function saveBlob(blob, name) {
  const url = URL.createObjectURL(blob),
    anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}
