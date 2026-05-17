/** @type {import('next').NextConfig} */
// NEXT_PUBLIC_API_URL is set in .env.production (Vercel) or .env.local (dev).
// See frontend/.env.production for the Render backend URL.
const nextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
