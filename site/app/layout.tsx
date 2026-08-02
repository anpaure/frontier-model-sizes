import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });
const publicUrl = "https://anpaure.github.io/frontier-model-sizes/";
const socialImage = `${publicUrl}og-v4.png`;

export const dynamic = "force-static";

export const metadata: Metadata = {
  metadataBase: new URL(publicUrl),
  title: "Frontier model size estimates",
  description: "Explore evidence-backed parameter-count estimates for current frontier AI models and reweight every evidence channel live.",
  icons: { icon: [{ url: `${publicUrl}favicon.svg`, type: "image/svg+xml" }] },
  openGraph: {
    title: "Frontier model size estimates",
    description: "Explore evidence-backed parameter-count estimates for current frontier AI models and reweight every evidence channel live.",
    type: "website",
    url: publicUrl,
    images: [{
      url: socialImage,
      width: 1200,
      height: 630,
      alt: "Frontier model size estimates page showing the interactive forest plot",
    }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Frontier model size estimates",
    description: "Explore evidence-backed parameter-count estimates for current frontier AI models and reweight every evidence channel live.",
    images: [socialImage],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>{children}</body>
    </html>
  );
}
