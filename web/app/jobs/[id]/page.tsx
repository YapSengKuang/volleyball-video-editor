"use client";

import { useParams } from "next/navigation";
import { JobView } from "./job-view";

export default function JobPage() {
  const params = useParams<{ id: string }>();
  const id = Array.isArray(params.id) ? params.id[0] : params.id;
  if (!id) return null;
  return <JobView id={id} />;
}
