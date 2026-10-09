export function describeFailure(error: unknown, action: string, target: string): string {
  const detail = error instanceof Error ? error.message : "Unknown error";
  const place = safeTarget(target);
  if (/failed to fetch|load failed|networkerror|network request failed/i.test(detail)) {
    return `${action} never reached ${place}. The browser said "${detail}". The site may be down, or the browser blocked that address.`;
  }
  return `${action} failed: ${detail}`;
}

function safeTarget(target: string): string {
  try {
    const url = new URL(target, "http://local.invalid");
    if (url.protocol === "http:" || url.protocol === "https:") return `${url.origin}${url.pathname}`;
  } catch {
    return target.split("?")[0] ?? target;
  }
  return target.split("?")[0] ?? target;
}

export async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const method = init?.method ?? "GET";
  const path = url.split("?")[0] ?? url;
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch (error) {
    throw new Error(describeFailure(error, `${method} ${path}`, url));
  }
  const text = await res.text();
  let data: { error?: string } = {};
  if (text) {
    try {
      data = JSON.parse(text) as { error?: string };
    } catch {
      data = {};
    }
  }
  if (!res.ok) {
    const detail = data.error || text.replace(/\s+/g, " ").slice(0, 180) || res.statusText;
    throw new Error(`${method} ${path} returned ${res.status}: ${detail}`);
  }
  return data as T;
}
