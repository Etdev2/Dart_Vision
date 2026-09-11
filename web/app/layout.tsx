import type { Metadata, Viewport } from 'next';
import './globals.css';
import Chrome from '@/components/Chrome';

export const metadata: Metadata = {
  title: 'Dart Vision',
  description: 'Single-camera dart scoring.',
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  viewportFit: 'cover',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Chrome />
        <main>{children}</main>
      </body>
    </html>
  );
}
