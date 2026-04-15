"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "../context/AuthContext";

export default function Page() {
  const router = useRouter();
  const { ready, isAuthenticated } = useAuth();

  useEffect(() => {
    if (!ready) return;
    router.replace(isAuthenticated ? "/dashboard" : "/login");
  }, [isAuthenticated, ready, router]);

  return null;
}
