import {
  CompleteMultipartUploadCommand,
  CreateMultipartUploadCommand,
  DeleteObjectCommand,
  HeadObjectCommand,
  S3Client,
  UploadPartCommand,
  GetObjectCommand,
  type CompletedPart,
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
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

export async function createUpload(key: string, contentType: string): Promise<string> {
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
