import { getSegments, publicJob, requireJob } from "@/lib/jobs";
import { assertUuid, errorResponse, json } from "@/lib/http";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(_req: Request, context: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await context.params;
    assertUuid(id);
    const job = await requireJob(id);
    const segments = await getSegments(id);
    return json(await publicJob(job, segments));
  } catch (error) {
    return errorResponse(error);
  }
}
