"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import toast from "react-hot-toast";
import { authenticateUser, clearSession, createUser, getActiveSession, persistSession } from "../lib/auth";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [ready, setReady] = useState(false);
  const router = useRouter();

  useEffect(() => {
    const activeSession = getActiveSession();
    if (activeSession?.isAuthenticated) {
      setUser(activeSession);
    }
    setReady(true);
  }, []);

  const value = useMemo(
    () => ({
      user,
      ready,
      isAuthenticated: Boolean(user?.isAuthenticated),
      signup: async ({ name, email, password }) => {
        const result = createUser({ name, email, password });
        if (!result.ok) {
          toast.error(result.message);
          return result;
        }
        toast.success("Account created. You can log in now.");
        return result;
      },
      login: async ({ email, password, remember }) => {
        const result = authenticateUser({ email, password });
        if (!result.ok) {
          toast.error("Invalid credentials");
          return result;
        }
        persistSession(result.user, remember);
        setUser(result.user);
        toast.success(`Welcome back, ${result.user.name}`);
        router.push("/dashboard");
        return result;
      },
      logout: () => {
        clearSession();
        setUser(null);
        toast.success("Logged out");
        router.push("/login");
      },
    }),
    [ready, router, user],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
}
