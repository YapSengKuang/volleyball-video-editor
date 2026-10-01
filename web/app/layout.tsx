import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Volleyball downtime editor",
  description: "Cut warmups, whistle waits, and ball chasing out of a game film.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <header className="top">
            <Link className="brand" href="/">
              <span className="mark" aria-hidden="true" />
              Rally cut
            </Link>
            <p className="lede">Keep the rallies. Drop the waiting.</p>
          </header>
          {children}
          <footer className="site-footer">
            <Link href="/about">About</Link>
            <Link href="/how-it-works">How it works</Link>
            <Link href="/privacy">Privacy</Link>
            <Link href="/terms">Terms</Link>
            <Link href="/cookies">Cookies</Link>
          </footer>
        </div>
      </body>
    </html>
  );
}
