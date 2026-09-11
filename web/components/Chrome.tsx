'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const TABS = [
  { href: '/', label: 'Mount' },
  { href: '/match/', label: 'Match' },
];

export default function Chrome() {
  const pathname = usePathname();
  return (
    <header className="chrome">
      <h1>Dart Vision</h1>
      <nav className="tabs">
        {TABS.map(({ href, label }) => (
          <Link
            key={href}
            href={href}
            className="tab"
            // Static export serves /match/ ; the trailing slash may or may not
            // survive the router, so match on the segment rather than equality.
            data-active={String(href === '/' ? pathname === '/' : pathname.startsWith('/match'))}
          >
            {label}
          </Link>
        ))}
      </nav>
    </header>
  );
}
