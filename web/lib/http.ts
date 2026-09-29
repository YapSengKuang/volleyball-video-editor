export class HttpError extends Error {
  status: number;
  retryAfter?: number;

  constructor(message: string, status: number, retryAfter?: number) {
    super(message);
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

export function json(data: unknown, status = 200, extra?: HeadersInit): Response {
  return Response.json(data, {
    status,
    headers: { "cache-control": "no-store", ...Object.fromEntries(new Headers(extra)) },
  });
}

export function errorResponse(error: unknown): Response {
  if (error instanceof HttpError) {
    return json(
      { error: error.message },
      error.status,
      error.retryAfter ? { "retry-after": String(error.retryAfter) } : undefined,
    );
  }
  console.error(error);
  return json({ error: "Something went wrong." }, 500);
}

export function clientIp(req: Request): string {
  const forwarded = req.headers.get("x-forwarded-for") ?? req.headers.get("x-real-ip");
  if (forwarded) return forwarded.split(",")[0]?.trim() || "local";
  return "local";
}

export async function readJson(req: Request, maxBytes: number): Promise<unknown> {
  const declared = Number(req.headers.get("content-length") ?? "0");
  if (Number.isFinite(declared) && declared > maxBytes) {
    throw new HttpError("Request is too large.", 413);
  }
  const raw = await req.arrayBuffer();
  if (raw.byteLength > maxBytes) throw new HttpError("Request is too large.", 413);
  try {
    return JSON.parse(new TextDecoder().decode(raw));
  } catch {
    throw new HttpError("Request body must be JSON.", 400);
  }
}

const UUID =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export function assertUuid(id: string): void {
  if (!UUID.test(id)) throw new HttpError("Unknown game.", 404);
}
