import {
  CompleteMultipartUploadCommand,
  CreateMultipartUploadCommand,
  DeleteObjectCommand,
  HeadObjectCommand,
  PutBucketCorsCommand,
  S3Client,
  UploadPartCommand,
  GetObjectCommand,
  type CompletedPart,
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { del, issueSignedToken, presignUrl } from "@vercel/blob";
import { config } from "./config";

function makeClient(endpoint: string): S3Client {
  return new S3Client({
    endpoint,
    region: config.s3Region,
    forcePathStyle: true,
    credentials: {
      accessKeyId: config.s3AccessKey,
      secretAccessKey: config.s3SecretKey,
    },
    requestChecksumCalculation: "WHEN_REQUIRED",
    responseChecksumValidation: "WHEN_REQUIRED",
  });
}

const internal = makeClient(config.s3Endpoint);
const presign = makeClient(config.s3PublicEndpoint);

const uploadOrigins = [
  "http://localhost:3010",
  "http://127.0.0.1:3010",
  "https://volleyball-rally-editor.vercel.app",
  ...(process.env.WEB_ORIGIN ?? "").split(",").map((origin) => origin.trim()).filter(Boolean),
];

let corsReady: Promise<void> | null = null;

function ensureUploadCors(): Promise<void> {
  if (config.useBlob) return Promise.resolve();
  corsReady ??= internal
    .send(
      new PutBucketCorsCommand({
        Bucket: config.s3Bucket,
        CORSConfiguration: {
          CORSRules: [
            {
              AllowedOrigins: [...new Set(uploadOrigins)],
              AllowedMethods: ["GET", "PUT", "HEAD"],
              AllowedHeaders: ["*"],
              ExposeHeaders: ["ETag", "Content-Length", "Content-Range", "Accept-Ranges"],
              MaxAgeSeconds: 3600,
            },
          ],
        },
      }),
    )
    .then(() => undefined)
    .catch((error: unknown) => {
      corsReady = null;
      throw error;
    });
  return corsReady;
}

export async function createUpload(key: string, contentType: string): Promise<string> {
  await ensureUploadCors();
  const created = await internal.send(
    new CreateMultipartUploadCommand({
      Bucket: config.s3Bucket,
      Key: key,
      ContentType: contentType,
    }),
  );
  if (!created.UploadId) throw new Error("Storage did not start the upload.");
  return created.UploadId;
}

export async function signPart(
  key: string,
  uploadId: string,
  partNumber: number,
): Promise<string> {
  return getSignedUrl(
    presign,
    new UploadPartCommand({
      Bucket: config.s3Bucket,
      Key: key,
      UploadId: uploadId,
      PartNumber: partNumber,
    }),
    { expiresIn: 60 * 60 },
  );
}

export async function completeUpload(
  key: string,
  uploadId: string,
  parts: CompletedPart[],
): Promise<void> {
  await internal.send(
    new CompleteMultipartUploadCommand({
      Bucket: config.s3Bucket,
      Key: key,
      UploadId: uploadId,
      MultipartUpload: { Parts: parts },
    }),
  );
}

export async function deleteObject(key: string): Promise<void> {
  await internal.send(new DeleteObjectCommand({ Bucket: config.s3Bucket, Key: key }));
}

export async function deleteStored(key: string | null): Promise<void> {
  if (!key) return;
  if (key.includes(".blob.vercel-storage.com")) {
    if (!process.env.BLOB_READ_WRITE_TOKEN) return;
    await del(key).catch(() => undefined);
    return;
  }
  await deleteObject(key).catch(() => undefined);
}

export async function objectSize(key: string): Promise<number> {
  const head = await internal.send(
    new HeadObjectCommand({ Bucket: config.s3Bucket, Key: key }),
  );
  return Number(head.ContentLength ?? 0);
}

export async function signGet(
  key: string,
  expires: number,
  downloadName?: string,
  contentType?: string,
): Promise<string> {
  if (key.startsWith("https://") && key.includes(".blob.vercel-storage.com")) {
    const pathname = new URL(key).pathname.slice(1);
    const validUntil = Date.now() + expires * 1000;
    const signed = await issueSignedToken({
      pathname,
      operations: ["get"],
      validUntil,
    });
    const presigned = await presignUrl(signed, {
      access: "private",
      operation: "get",
      pathname,
      validUntil,
    });
    return presigned.presignedUrl;
  }
  const filename = downloadName?.replace(/["\r\n]/g, "") ?? "";
  return getSignedUrl(
    presign,
    new GetObjectCommand({
      Bucket: config.s3Bucket,
      Key: key,
      ResponseContentType: contentType,
      ResponseContentDisposition: filename ? `attachment; filename="${filename}"` : undefined,
    }),
    { expiresIn: expires },
  );
}
