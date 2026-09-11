import type { Metadata, Viewport } from 'next';
import './globals.css';
import Chrome from '@/components/Chrome';
import OfflineReady from '@/components/OfflineReady';

export const metadata: Metadata = {
  title: 'Dart Vision',
  description: 'Single-camera dart scoring.',
  manifest: '/manifest.webmanifest',
  appleWebApp: { capable: true, title: 'Dart Vision', statusBarStyle: 'black-translucent' },
  icons: { apple: '/apple-touch-icon.png' },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  viewportFit: 'cover',
  themeColor: '#0f172a',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <OfflineReady />
        <Chrome />
        <main>{children}</main>
      </body>
    </html>
  );
}
