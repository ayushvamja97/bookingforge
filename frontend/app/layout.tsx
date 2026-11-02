import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BookingForge · Hotel Intelligence Research",
  description:
    "Explore reproducible NumPy neural-network research on historical hotel reservation outcomes. A local, synthetic-data-only demo.",
  robots: { index: false, follow: false },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
