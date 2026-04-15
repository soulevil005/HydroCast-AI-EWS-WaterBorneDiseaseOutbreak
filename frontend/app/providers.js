"use client";

import { Toaster } from "react-hot-toast";
import { AuthProvider } from "../context/AuthContext";
import AuthGuard from "../components/auth-guard";
import TopLoadingBar from "../components/top-loading-bar";

export default function Providers({ children }) {
  return (
    <AuthProvider>
      <TopLoadingBar />
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 3200,
          style: {
            background: "rgba(9,18,33,0.92)",
            color: "#f5f8ff",
            border: "1px solid rgba(146,165,199,0.16)",
            backdropFilter: "blur(14px)",
          },
        }}
      />
      <AuthGuard>{children}</AuthGuard>
    </AuthProvider>
  );
}
