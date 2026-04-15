"use client";

import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "../context/AuthContext";

const protectedPaths = ["/dashboard", "/risk-map"];

export default function AuthGuard({ children }) {
  const { ready, isAuthenticated } = useAuth();
  const pathname = usePathname();
  const router = useRouter();

  useEffect(() => {
    if (!ready) return;
    if (protectedPaths.includes(pathname) && !isAuthenticated) {
      router.replace("/login");
    }
    if (pathname === "/login" && isAuthenticated) {
      router.replace("/dashboard");
    }
  }, [isAuthenticated, pathname, ready, router]);

  if (!ready) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-command px-6 text-white">
        <div className="glass-panel w-full max-w-md rounded-[1.6rem] p-8 text-center">
          <div className="orbitron text-3xl font-bold">HydroCast</div>
          <div className="mt-4 text-slate-300">Preparing secure workspace...</div>
          <div className="mt-6 h-2 rounded-full bg-white/10">
            <div className="h-2 w-2/3 animate-pulse rounded-full bg-gradient-to-r from-cyan-400 to-violet-500" />
          </div>
        </div>
      </div>
    );
  }

  if (protectedPaths.includes(pathname) && !isAuthenticated) {
    return null;
  }

  return children;
}
