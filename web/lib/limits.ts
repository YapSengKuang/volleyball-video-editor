import { pool } from "./db";

export async function hitLimit(
  key: string,
  action: string,
  windowSeconds: number,
  limit: number,
): Promise<boolean> {
  const result = await pool.query<{ count: number }>(
    `INSERT INTO rate_limits (rate_key, action, window_start, count)
     VALUES ($1, $2, to_timestamp(floor(extract(epoch from now()) / $3) * $3), 1)
     ON CONFLICT (rate_key, action, window_start)
     DO UPDATE SET count = rate_limits.count + 1
     RETURNING count`,
    [key, action, windowSeconds],
  );
  return result.rows[0].count > limit;
}

export async function peekLimit(key: string, action: string, windowSeconds: number): Promise<number> {
  const result = await pool.query<{ count: number }>(
    `SELECT count FROM rate_limits
     WHERE rate_key = $1 AND action = $2
       AND window_start = to_timestamp(floor(extract(epoch from now()) / $3) * $3)`,
    [key, action, windowSeconds],
  );
  return result.rows[0]?.count ?? 0;
}
