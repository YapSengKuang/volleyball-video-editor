import type { ReactNode } from "react";

export function InfoPage({ title, children }: { title: string; children: ReactNode }) {
  return (
    <article className="panel info">
      <h1>{title}</h1>
      {children}
    </article>
  );
}
