CREATE TABLE jobs (
  id UUID PRIMARY KEY,
  status TEXT NOT NULL,
  filename TEXT NOT NULL,
  content_type TEXT NOT NULL,
  size_bytes BIGINT NOT NULL,
  output_bytes BIGINT,
  source_key TEXT NOT NULL,
  output_key TEXT,
  merged_key TEXT,
  merged_bytes BIGINT,
  upload_id TEXT,
  duration_seconds DOUBLE PRECISION,
  eta_seconds DOUBLE PRECISION,
  phase TEXT,
  error TEXT,
  warning TEXT,
  client_ip TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at TIMESTAMPTZ,
  expires_at TIMESTAMPTZ
);

CREATE UNIQUE INDEX one_upload_per_ip ON jobs (client_ip) WHERE status = 'uploading';
CREATE INDEX jobs_status_idx ON jobs (status, created_at);
CREATE INDEX jobs_expires_idx ON jobs (expires_at);

CREATE TABLE segments (
  id UUID PRIMARY KEY,
  job_id UUID NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
  start_seconds DOUBLE PRECISION NOT NULL,
  end_seconds DOUBLE PRECISION NOT NULL,
  keep BOOLEAN NOT NULL,
  origin TEXT NOT NULL CHECK (origin IN ('auto', 'user')),
  position INT NOT NULL
);

CREATE INDEX segments_job_idx ON segments (job_id, position);

CREATE TABLE rate_limits (
  rate_key TEXT NOT NULL,
  action TEXT NOT NULL,
  window_start TIMESTAMPTZ NOT NULL,
  count INT NOT NULL,
  PRIMARY KEY (rate_key, action, window_start)
);
