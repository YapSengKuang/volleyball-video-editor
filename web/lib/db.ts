import { Pool } from "pg";
import { config } from "./config";

const globalForPg = globalThis as unknown as { pool?: Pool };

export const pool =
  globalForPg.pool ??
  new Pool({
    connectionString: config.databaseUrl,
    max: 10,
  });

if (process.env.NODE_ENV !== "production") {
  globalForPg.pool = pool;
}
