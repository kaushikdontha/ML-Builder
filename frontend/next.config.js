/** @type {import('next').NextConfig} */
const nextConfig = {
    async rewrites() {
        let backendUrl = process.env.BACKEND_URL || 'http://127.0.0.1:8000';
        if (!backendUrl.startsWith('http')) {
            backendUrl = `http://${backendUrl}`;
        }
        backendUrl = backendUrl.replace(/\/$/, '');
        return [
            {
                source: '/api/:path*',
                destination: `${backendUrl}/api/:path*`,
            },
        ]
    },
};

module.exports = nextConfig;
