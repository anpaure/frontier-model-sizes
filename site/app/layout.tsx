import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { headers } from "next/headers";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const origin = `${protocol}://${host}`;
  const title = "Frontier estimates";
  const description = "Compare ten current frontier parameter estimates in an interactive forest plot and reweight every evidence channel live.";
  const image = `${origin}/og-v3.png`;
  return {
    metadataBase: new URL(origin),
    title,
    description,
    icons: { icon: [{ url: "/og-v3.png", type: "image/png" }] },
    openGraph: { title, description, type: "website", url: origin, images: [{ url: image, width: 1568, height: 1003, alt: "Frontier estimates forest plot" }] },
    twitter: { card: "summary_large_image", title, description, images: [image] },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>{children}</body>
    </html>
  );
}
