import "./globals.css";
import Providers from "./providers";

export const metadata = {
  title: "HydroCast",
  description: "AI-powered early warning dashboard for waterborne disease outbreaks in Maharashtra",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
